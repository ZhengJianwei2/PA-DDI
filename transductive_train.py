from PA_DDI import models, custom_loss
from PA_DDI.config import cfg, update_cfg
from datetime import datetime
from PA_DDI.data_process.dataloader import get_dataloader
import torch
import numpy as np
from torch import optim
from tqdm import tqdm
from sklearn import metrics
import time

def do_compute(batch, device, model):
    '''
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap

def test(test_data_loader,model, device):
    test_probas_pred = []
    test_ground_truth = []
    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_acc, test_auc_roc, test_f1, test_precision,test_recall,test_int_ap, test_ap = do_compute_metrics(test_probas_pred, test_ground_truth)
    print('\n')
    print('============================== Test Result ==============================')
    print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')


if __name__ == '__main__':
    # get config
    cfg.merge_from_file('./PA_DDI/drugbank_trans.yaml')
    cfg = update_cfg(cfg)
    train_loader, valid_loader, test_loader = get_dataloader(cfg)
    device = 'cuda:1'
    model = models.PA_DDI(cfg, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])
    model.to(device)
    nc_loss = custom_loss.SigmoidLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    max_acc = 0
    pkl_name = 'drugbank_trans.pkl'
    val_loss_rec = []
    val_acc_rec = []
    print('training')
    for epoch in range(cfg.train.epochs):
        print(f'epoch {epoch} start')
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []


        for batch in tqdm(train_loader, desc="Training batches"):
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = nc_loss(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_loader)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_f1, train_precision, train_recall, train_int_ap, train_ap = do_compute_metrics(
                train_probas_pred, train_ground_truth)

            for batch in valid_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = nc_loss(p_score, n_score)
                val_loss += loss.item() * len(p_score)

            val_loss /= len(valid_loader)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_f1, val_precision, val_recall, val_int_ap, val_ap = do_compute_metrics(
                val_probas_pred, val_ground_truth)
            if val_acc > max_acc:
                max_acc = val_acc
                torch.save(model, pkl_name)

        if scheduler:
            # print('scheduling')
            scheduler.step()

        print(f'Epoch: {epoch} train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'
              f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
        print(
            f'\t\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, train_precision: {train_precision:.4f}, val_precision: {val_precision:.4f}')
        print(f'epoch {epoch} finished')
        val_loss_rec.append(val_loss)
        val_acc_rec.append(val_acc)

    val_loss_rec = np.array(val_loss_rec)
    np.save('base_val_loss_rec.npy', val_loss_rec)
    val_acc_rec = np.array(val_acc_rec)
    np.save('base_val_acc_rec.npy', val_acc_rec)

    test_model = torch.load(pkl_name)
    model.to(device)
    test(test_loader, test_model, device)