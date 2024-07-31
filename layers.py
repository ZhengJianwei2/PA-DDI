import math
import datetime

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv,SAGPooling,global_add_pool, GATConv, LayerNorm
from torch_scatter import scatter


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores
        return attentions

class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    
    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)
      
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        
        rels = rels.view(-1, self.n_features, self.n_features)
        # print(heads.size(),rels.size(),tails.size())
        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
          scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))
       
        return scores 
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


# intra rep
class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim,32,2)
    
    def forward(self,data):
        input_feature,edge_index = data.x_emb, data.edge_index
        input_feature = F.elu(input_feature)
        intra_rep = self.intra(input_feature,edge_index)
        return intra_rep

# inter rep
class InterGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim,input_dim),32,2)
    
    def forward(self,h_data,t_data,b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x_emb)
        t_input = F.elu(t_data.x_emb)
        t_rep = self.inter((h_input,t_input),edge_index)
        h_rep = self.inter((t_input,h_input),edge_index[[1,0]])
        return h_rep,t_rep


class PositionEmbbeding(nn.Module):
    def __init__(self, input_dims, nhid):
        super().__init__()
        self.input_dim = input_dims
        self.norm = LayerNorm(128)
        self.ego_sub_emb = nn.Linear(16, nhid)
        self.global_emb = nn.Linear(20, nhid)
        self.cut_sub_emb = nn.Linear(16, nhid)
        self.merge = nn.Linear(55+nhid+nhid, nhid)

    def forward(self, data):
        ego_sub_embedding = data.ego_RWPE
        global_embedding = data.glo_RWPE
        # cut_sub_embedding = data.cut_RWPE
        # subgraph_x_index = data.subgraph_x_index

        ego_sub_embedding = self.ego_sub_emb(ego_sub_embedding)
        global_embedding = self.global_emb(global_embedding)
        # cut_sub_embedding = self.cut_sub_emb(cut_sub_embedding)

        # cut_sub_embedding_trans = torch.zeros_like(cut_sub_embedding).to(cut_sub_embedding)
        # cut_sub_embedding_trans[subgraph_x_index] = cut_sub_embedding
        ego_sub_embedding_pooled = scatter(ego_sub_embedding, data.subgraphs_batch, dim=0, reduce="add")

        # x_cut = self.merge2(torch.cat((data.x_emb, cut_sub_embedding_trans, global_embedding), dim=-1))

        x_emb = torch.cat((data.x_emb, ego_sub_embedding_pooled, global_embedding), dim=-1)
        x_emb = self.merge(x_emb)
        x_emb = self.norm(x_emb, data.batch)
        return x_emb



