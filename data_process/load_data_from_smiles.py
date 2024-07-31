import itertools
import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem
import pandas as pd
import numpy as np


def one_of_k_encoding_unk1(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    res = allowable_set.index(x)
    return res


def atom_features_emb1(atom):
    results = one_of_k_encoding_unk1(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ])
    return torch.tensor(results)

def one_of_k_encoding_unk2(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features_emb2(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk2(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(),
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk2(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk2(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)

def get_mol_edge_list_and_feat_mtx(mol_graph):
    n_features1 = [(atom.GetIdx(), atom_features_emb1(atom)) for atom in mol_graph.GetAtoms()]
    n_features1.sort()
    _, n_features1 = zip(*n_features1)
    n_features1 = torch.stack(n_features1)

    n_features2 = [(atom.GetIdx(), atom_features_emb2(atom)) for atom in mol_graph.GetAtoms()]
    n_features2.sort()
    _, n_features2 = zip(*n_features2)
    n_features2 = torch.stack(n_features2)

    edge_list = []
    edge_attr_list = []
    for b in mol_graph.GetBonds():
        edge_list.append([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
        edge_attr_list.append(b.GetBondTypeAsDouble())

    edge_list = torch.LongTensor(edge_list)
    edge_attr_list = torch.FloatTensor(edge_attr_list)

    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) > 0 else edge_list
    undirected_edge_attr_list = torch.cat([edge_attr_list, edge_attr_list], dim=0) if len(
    edge_attr_list) > 0 else edge_attr_list
    return undirected_edge_list, undirected_edge_attr_list, n_features1, n_features2


def get_bipartite_graph(mol_graph_1,mol_graph_2):
    x1 = np.arange(0,len(mol_graph_1.GetAtoms()))
    x2 = np.arange(0,len(mol_graph_2.GetAtoms()))
    edge_list = torch.LongTensor(np.meshgrid(x1,x2))
    edge_list = torch.stack([edge_list[0].reshape(-1),edge_list[1].reshape(-1)])
    return edge_list


def create_data_dict(drug_id_mol_graph_tup):
    data_dict = {
        drug_id: {
            'x': n_features1,
            'x_emb': n_features2,
            'edge_index': undirected_edge_list.t().contiguous(),
            'edge_attr': undirected_edge_attr_list
        }
        for drug_id, (undirected_edge_list, undirected_edge_attr_list, n_features1, n_features2) in drug_id_mol_graph_tup.items()
    }
    graph_data_objects = {k: Data(**v) for k, v in data_dict.items()}
    return graph_data_objects

def load_data_from_smiles(path):
    df_drugs_smiles = pd.read_csv(path)

    DRUG_TO_INDX_DICT = {drug_id: indx for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}

    drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in
                             zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]

    drug_to_mol_graph = {id: Chem.MolFromSmiles(smiles.strip()) for id, smiles in
                         zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])}

    # Gettings information and features of atoms
    ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])
    AVAILABLE_ATOM_SYMBOLS = list(
        {a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
    AVAILABLE_ATOM_DEGREES = list(
        {a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
    AVAILABLE_ATOM_TOTAL_HS = list(
        {a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
    max_valence = max(
        a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
    max_valence = max(max_valence, 9)
    AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

    MAX_ATOM_FC = abs(np.max(
        [a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
    MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
    MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in
                                  itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
    MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0

    MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol)
                              for drug_id, mol in drug_id_mol_graph_tup}

    MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}

    MOL_EDGE_LIST_FEAT_MTX = create_data_dict(MOL_EDGE_LIST_FEAT_MTX)

    return MOL_EDGE_LIST_FEAT_MTX


if __name__ == '__main__':
    data = load_data_from_smiles('../../data/drugbank/test.csv')
    print(data)