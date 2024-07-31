import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch_geometric.data import Data, Batch
import numpy as np
from collections import defaultdict
import random
from torch.utils.data import random_split
from PA_DDI.config import cfg, update_cfg


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

def get_bipartite_graph(mol_graph_1,mol_graph_2):
    x1 = np.arange(0,len(mol_graph_1.x))
    x2 = np.arange(0,len(mol_graph_2.x))
    edge_list = torch.LongTensor(np.meshgrid(x1,x2))
    edge_list = torch.stack([edge_list[0].reshape(-1),edge_list[1].reshape(-1)])
    return edge_list




class DrugDataset(Dataset):
    def __init__(self, csv_file, pt_file,disjoint_split=True):
        self.train_data = pd.read_csv(csv_file)
        self.pt_data = torch.load(pt_file)
        df_all_pos_ddi = self.train_data
        self.all_pos_tup = [(h, t, r) for h, t, r in zip(df_all_pos_ddi['d1'], df_all_pos_ddi['d2'], df_all_pos_ddi['type'])]

        self.ALL_TRUE_H_WITH_TR = defaultdict(list)
        self.ALL_TRUE_T_WITH_HR = defaultdict(list)

        self.FREQ_REL = defaultdict(int)
        self.ALL_H_WITH_R = defaultdict(dict)
        self.ALL_T_WITH_R = defaultdict(dict)
        self.ALL_TAIL_PER_HEAD = {}
        self.ALL_HEAD_PER_TAIL = {}

        for h, t, r in self.all_pos_tup:
            self.ALL_TRUE_H_WITH_TR[(t, r)].append(h)
            self.ALL_TRUE_T_WITH_HR[(h, r)].append(t)
            self.FREQ_REL[r] += 1.0
            self.ALL_H_WITH_R[r][h] = 1
            self.ALL_T_WITH_R[r][t] = 1

        for t, r in self.ALL_TRUE_H_WITH_TR:
            self.ALL_TRUE_H_WITH_TR[(t, r)] = np.array(list(set(self.ALL_TRUE_H_WITH_TR[(t, r)])))
        for h, r in self.ALL_TRUE_T_WITH_HR:
            self.ALL_TRUE_T_WITH_HR[(h, r)] = np.array(list(set(self.ALL_TRUE_T_WITH_HR[(h, r)])))

        for r in self.FREQ_REL:
            self.ALL_H_WITH_R[r] = np.array(list(self.ALL_H_WITH_R[r].keys()))
            self.ALL_T_WITH_R[r] = np.array(list(self.ALL_T_WITH_R[r].keys()))
            self.ALL_HEAD_PER_TAIL[r] = self.FREQ_REL[r] / len(self.ALL_T_WITH_R[r])
            self.ALL_TAIL_PER_HEAD[r] = self.FREQ_REL[r] / len(self.ALL_H_WITH_R[r])
        if disjoint_split:
            d1, d2, *_ = zip(*self.all_pos_tup)
            self.drug_ids = np.array(list(set(d1 + d2)))
        else:
            ids = list(self.pt_data.keys())
            self.drug_ids = np.array(ids)


    def __len__(self):
        return len(self.all_pos_tup)

    def __getitem__(self, idx):
        return self.all_pos_tup[idx]

    def collate_fn(self, batch):
        pos_rels = []
        pos_h_samples = []
        pos_t_samples = []
        pos_b_samples = []

        neg_rels = []
        neg_h_samples = []
        neg_t_samples = []
        neg_b_samples = []

        for d1, d2, rel in batch:
            pos_rels.append(rel)
            d1_mol = self.pt_data[d1]
            d2_mol = self.pt_data[d2]
            pos_b_graph = self._create_b_graph(get_bipartite_graph(d1_mol, d2_mol), d1_mol.x_emb, d2_mol.x_emb)

            pos_h_samples.append(d1_mol)
            pos_t_samples.append(d2_mol)
            pos_b_samples.append(pos_b_graph)

            neg_rels.append(rel)
            neg_head, neg_tail = self.__normal_batch(d1, d2, rel, 1)
            if len(neg_head) > 0:
                neg_h_graph = self.pt_data[neg_head[0]]
                neg_t_graph = self.pt_data[d2]
                neg_b_graph = self._create_b_graph(get_bipartite_graph(neg_h_graph, neg_t_graph), neg_h_graph.x_emb,
                                                   neg_t_graph.x_emb)
            else:
                neg_h_graph = self.pt_data[d1]
                neg_t_graph = self.pt_data[neg_tail[0]]
                neg_b_graph = self._create_b_graph(get_bipartite_graph(neg_h_graph, neg_t_graph), neg_h_graph.x_emb,
                                                   neg_t_graph.x_emb)

            neg_h_samples.append(neg_h_graph)
            neg_t_samples.append(neg_t_graph)
            neg_b_samples.append(neg_b_graph)

        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_b_samples = Batch.from_data_list(pos_b_samples)
        pos_rels = torch.LongTensor(pos_rels).unsqueeze(0)

        neg_h_samples = Batch.from_data_list(neg_h_samples)
        neg_t_samples = Batch.from_data_list(neg_t_samples)
        neg_b_samples = Batch.from_data_list(neg_b_samples)
        neg_rels = torch.LongTensor(neg_rels).unsqueeze(0)

        pos_tri = (pos_h_samples, pos_t_samples, pos_rels, pos_b_samples)
        neg_tri = (neg_h_samples, neg_t_samples, neg_rels, neg_b_samples)

        return pos_tri, neg_tri


    def _create_b_graph(self,edge_index,x_s, x_t):
        return BipartiteData(edge_index,x_s,x_t)

    def __corrupt_ent(self, other_ent, r, other_ent_with_r_dict, max_num=1):
        corrupted_ents = []
        current_size = 0
        while current_size < max_num:
            candidates = np.random.choice(self.drug_ids, (max_num - current_size) * 2)
            mask = np.isin(candidates, other_ent_with_r_dict[(other_ent, r)], assume_unique=True, invert=True)
            corrupted_ents.append(candidates[mask])
            current_size += len(corrupted_ents[-1])

        if corrupted_ents != []:
            corrupted_ents = np.concatenate(corrupted_ents)

        return np.asarray(corrupted_ents[:max_num])

    def __corrupt_head(self, t, r, n=1):
        return self.__corrupt_ent(t, r, self.ALL_TRUE_H_WITH_TR, n)

    def __corrupt_tail(self, h, r, n=1):
        return self.__corrupt_ent(h, r, self.ALL_TRUE_T_WITH_HR, n)

    def __normal_batch(self, h, t, r, neg_size):
        neg_size_h = 0
        neg_size_t = 0
        prob = self.ALL_TAIL_PER_HEAD[r] / (self.ALL_TAIL_PER_HEAD[r] + self.ALL_HEAD_PER_TAIL[r])
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        return self.__corrupt_head(t, r, neg_size_h), self.__corrupt_tail(h, r, neg_size_t)



def filter_data(csv_file, pt_file):
    pt_data = torch.load(pt_file)
    df = pd.read_csv(csv_file)

    filtered_df = df[df['d1'].map(lambda x: x.split('$')[0] in pt_data) &
                     df['d2'].map(lambda x: x.split('$')[0] in pt_data) &
                     df['Neg samples'].map(lambda x: x.split('$')[0] in pt_data)]

    return filtered_df

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)
def get_dataloader(cfg):
    pt_path = cfg.data.pt_path
    train_csv_path = cfg.data.train_ddi
    train_dataset = DrugDataset(csv_file=train_csv_path, pt_file=pt_path)
    train_data_loader = DrugDataLoader(train_dataset, batch_size=1024, shuffle=True)

    valid_csv_path = cfg.data.valid_ddi
    valid_dataset = DrugDataset(csv_file=valid_csv_path, pt_file=pt_path, disjoint_split=False)
    valid_data_loader = DrugDataLoader(valid_dataset, batch_size=1024)


    test_csv_path = cfg.data.test_ddi
    test_dataset = DrugDataset(csv_file=test_csv_path, pt_file=pt_path, disjoint_split=False)
    test_data_loader = DrugDataLoader(test_dataset, batch_size=1024)

    return train_data_loader, valid_data_loader, test_data_loader

def get_inductive_dataloader(cfg):
    pt_path = cfg.data.pt_path
    train_csv_path = cfg.data.train_ddi
    train_dataset = DrugDataset(csv_file=train_csv_path, pt_file=pt_path)
    train_data_loader = DrugDataLoader(train_dataset, batch_size=1024, shuffle=True)

    s1_csv_path = cfg.data.valid_ddi
    s1_dataset = DrugDataset(csv_file=s1_csv_path, pt_file=pt_path)
    s1_data_loader = DrugDataLoader(s1_dataset, batch_size=1024)


    s2_csv_path = cfg.data.test_ddi
    s2_dataset = DrugDataset(csv_file=s2_csv_path, pt_file=pt_path)
    s2_data_loader = DrugDataLoader(s2_dataset, batch_size=1024)

    return train_data_loader, s1_data_loader, s2_data_loader

if __name__ == '__main__':
    # csv_path = "./data/drugbank/fold0/train2.csv"
    # pt_path = "./data/drugbank/processed/t3/drug_graph_data.pt"
    #
    # # filtered_data = filter_data(csv_path, pt_path)
    # # filtered_data.to_csv('./data/drugbank/fold0/train.csv', index=False)
    #
    #
    # drug_dataset = DrugDataset(csv_file=csv_path, pt_file=pt_path)
    # train_data_loader = DataLoader(drug_dataset, batch_size=1024, shuffle=True)
    # pos = 0
    # neg = 0
    # for i, batch in enumerate(train_data_loader):
    #     if  len(batch) > 1:
    #         pos_tri, neg_tri = batch
    #         if pos< 3:
    #             print(f"Batch: {i + 1}")
    #             print(f"Positive Triplets: {pos_tri}")
    #             print(f"Negative Triplets: {neg_tri}")
    #         pos+=1
    #         print(i)
    #     else:
    #         neg+=1
    #         print(i)
    #     if i>3 :
    #         break
    # print(f'pos:{pos}, neg:{neg}')
    cfg.merge_from_file('./model_v1/drugbank.yaml')
    cfg = update_cfg(cfg)
    train_loader, valid_loader, test_loader = get_dataloader(cfg)

    for i, batch in enumerate(train_loader):
        print(i)
        pos_tri, neg_tri = batch
        device = 'cuda:0'
        pos_tri = [tensor.to(device=device) for tensor in pos_tri]
        neg_tri = [tensor.to(device=device) for tensor in neg_tri]




