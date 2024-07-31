from PA_DDI import models, custom_loss
from PA_DDI.config import cfg, update_cfg
from datetime import datetime
from PA_DDI.data_process.dataloader import get_inductive_dataloader
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


def test(s1_data_loader, s2_data_loader, model1, model2):
    s1_probas_pred = []
    s1_ground_truth = []

    s2_probas_pred = []
    s2_ground_truth = []
    with torch.no_grad():
        for batch in s1_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model=model1)
            s1_probas_pred.append(probas_pred)
            s1_ground_truth.append(ground_truth)

        s1_probas_pred = np.concatenate(s1_probas_pred)
        s1_ground_truth = np.concatenate(s1_ground_truth)
        s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred,
                                                                                                  s1_ground_truth)

        for batch in s2_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model=model2)
            s2_probas_pred.append(probas_pred)
            s2_ground_truth.append(ground_truth)

        s2_probas_pred = np.concatenate(s2_probas_pred)
        s2_ground_truth = np.concatenate(s2_ground_truth)
        s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred,
                                                                                                  s2_ground_truth)

    print('\n')
    print('============================== Best Result ==============================')
    print(
        f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_f1: {s1_f1:.4f}, s1_precision: {s1_precision:.4f},s1_recall: {s1_recall:.4f},s1_int_ap: {s1_int_ap:.4f},s1_ap: {s1_ap:.4f}')
    print(
        f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_f1: {s2_f1:.4f}, s2_precision: {s2_precision:.4f},s2_recall: {s2_recall:.4f},s2_int_ap: {s2_int_ap:.4f},s2_ap: {s2_ap:.4f}')


if __name__ == '__main__':
    # get config
    cfg.merge_from_file('./PA_DDI/drugbank_in.yaml')
    cfg = update_cfg(cfg)
    train_loader, s1_loader, s2_loader = get_inductive_dataloader(cfg)
    device = 'cuda:7'
    model = models.PA_DDI(cfg, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])
    model.to(device)
    nc_loss = custom_loss.SigmoidLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    max_acc = 0
    s1_pkl_name = 'drugbank_inductive_v2_s1.pkl'
    s2_pkl_name = 'drugbank_inductive_v2_s2.pkl'

    s1_acc_max = 0
    s2_acc_max = 0
    for i in range(1, cfg.train.epochs):
        print(f'epoch {i} start')
        start = time.time()
        train_loss = 0
        s1_loss = 0
        s2_loss = 0

        train_probas_pred = []
        train_ground_truth = []

        s1_probas_pred = []
        s1_ground_truth = []

        s2_probas_pred = []
        s2_ground_truth = []

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

            for batch in s1_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                s1_probas_pred.append(probas_pred)
                s1_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = nc_loss(p_score, n_score)
                s1_loss += loss.item() * len(p_score)

            s1_loss /= len(s1_loader)
            s1_probas_pred = np.concatenate(s1_probas_pred)
            s1_ground_truth = np.concatenate(s1_ground_truth)
            s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred,
                                                                                                      s1_ground_truth)

            for batch in s2_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                s2_probas_pred.append(probas_pred)
                s2_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = nc_loss(p_score, n_score)
                s2_loss += loss.item() * len(p_score)

            s2_loss /= len(s2_loader)
            s2_probas_pred = np.concatenate(s2_probas_pred)
            s2_ground_truth = np.concatenate(s2_ground_truth)
            s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred,
                                                                                                      s2_ground_truth)

            if s1_acc > s1_acc_max:
                s1_acc_max = s1_acc
                torch.save(model, s1_pkl_name)
            if s2_acc>s2_acc_max:
                s2_acc_max = s2_acc
                torch.save(model, s2_pkl_name)

        if scheduler:
            # print('scheduling')
            scheduler.step()

        print(
            f'Epoch: {i} (train_loss: {train_loss:.4f}, s1_loss: {s1_loss:.4f},s2_loss: {s2_loss:.4f}')
        print(
            f'\t\ttrain_acc: {train_acc:.4f}, train_roc: {train_auc_roc:.4f},train_precision: {train_precision:.4f},train_recall:{train_recall:.4f}')
        print(
            f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_precision: {s1_precision:.4f}, s1_recall: {s1_recall:.4f}')
        print(
            f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_precision: {s2_precision:.4f}, s2_recall: {s2_recall:.4f}')

    test_model1 = torch.load(s1_pkl_name)
    test_model2 = torch.load(s2_pkl_name)
    test(s1_loader, s2_loader, test_model1, test_model2)