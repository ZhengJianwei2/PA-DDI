import torch

from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                Set2Set,
                                )

from model_v2.layers import (
                    CoAttentionLayer, 
                    RESCAL, 
                    IntraGraphAttention,
                    InterGraphAttention,
                    PositionEmbbeding
                    )

class PA_DDI(nn.Module):
    def __init__(self, cfg, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = cfg.model.n_atom_feats
        self.hidd_dim = cfg.model.n_atom_hid
        self.rel_total = cfg.model.rel_total
        self.kge_dim = cfg.model.kge_dim
        self.n_blocks = len(blocks_params)
        
        self.initial_norm = PositionEmbbeding(self.in_features, 128)
        self.blocks = []
        self.net_norms = ModuleList()
        in_features = 128
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = Bi_MPNN(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads
        
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples

        h_data.x_emb = self.initial_norm(h_data)
        t_data.x_emb = self.initial_norm(t_data)
        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x_emb = F.elu(self.net_norms[i](h_data.x_emb, h_data.batch))
            t_data.x_emb = F.elu(self.net_norms[i](t_data.x_emb, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        kge_heads = repr_h
        kge_tails = repr_t
        # print(kge_heads.size(), kge_tails.size(), rels.size())
        attentions = self.co_attention(kge_heads, kge_tails)
        # attentions = None
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        return scores

#intra+inter
class Bi_MPNN(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        self.intraAtt = IntraGraphAttention(head_out_feats * n_heads)
        self.interAtt = InterGraphAttention(head_out_feats * n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, h_data, t_data, b_graph):
        h_data.x_emb = self.feature_conv(h_data.x_emb, h_data.edge_index)
        t_data.x_emb = self.feature_conv(t_data.x_emb, t_data.edge_index)

        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x_emb = h_rep
        t_data.x_emb = t_rep

        # readout
        h_att_x_emb, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores = self.readout(h_data.x_emb,
                                                                                                   h_data.edge_index,
                                                                                                   batch=h_data.batch)
        t_att_x_emb, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores = self.readout(t_data.x_emb,
                                                                                                   t_data.edge_index,
                                                                                                   batch=t_data.batch)

        h_global_graph_emb = global_add_pool(h_att_x_emb, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x_emb, t_att_batch)

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb



