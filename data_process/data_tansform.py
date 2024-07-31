import torch
from PA_DDI.config import cfg
from PA_DDI.data_process.transform import SubgraphsTransform, SubgraphsData
import os.path as osp
import os
from load_data_from_smiles import load_data_from_smiles



def create_dataset(cfg, data):
    # No need to do offline transformation
    cfg.processed_dir = './data/drugbank/processed/'
    transform = SubgraphsTransform(cfg)

    if os.path.exists(cfg.processed_dir):
        print(f'loading data from {cfg.processed_dir}')
        smiles_graph_data = torch.load(osp.join(cfg.processed_dir, 'drug_graph_data.pt'))
    else:
        print(f'processing data')
        smiles_graph_data = {}
        for key, value in data.items():
            value = transform.transform(value)
            smiles_graph_data[key] = value
            # print(f"Error occurred with key: {key}")
            # print(f"Exception: {str(e)}")
        os.makedirs(cfg.processed_dir)
        torch.save(smiles_graph_data, osp.join(cfg.processed_dir, 'drug_graph_data.pt'))

    return smiles_graph_data

def process_data():
    path = "./data/drugbank/processed/drug_graph_data.pt"
    smiles_graph_data = torch.load(path)
    processed_data = {}
    keys_to_keep = ['x', 'edge_attr', 'x_emb', 'subgraphs_batch', 'ego_RWPE', 'glo_RWPE', 'edge_index']
    for key, data in smiles_graph_data.items():
        new_data = SubgraphsData()
        for k in keys_to_keep:
            if k in data:
                setattr(new_data, k, data[k])
        processed_data[key] = new_data

    out_path = "./data/drugbank/processed/processed_data.pt"
    torch.save(processed_data, out_path)
    print(processed_data)



if __name__ == '__main__':
    cfg.merge_from_file('./model_v1/drugbank.yaml')
    data = load_data_from_smiles('./data/drugbank/drug_smiles.csv')
    # data = load_data_from_smiles('./data/drugbank/test.csv')
    data = create_dataset(cfg, data)
    # for key, value in data.items():
    #     x = value.subgraphs_batch
    #     if not x.dtype == torch.int64:
    #         print('Shape of x:', x.shape)
    #         print('Type of x:', x.dtype)
    #         print(key)
    print(data)

    # process_data()
