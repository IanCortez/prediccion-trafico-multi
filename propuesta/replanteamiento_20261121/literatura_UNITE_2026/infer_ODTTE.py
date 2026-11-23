import math
import pandas as pd
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.tte import collate_fn, TTEData, UNITE
from util import AttrDict, cal_regression_metric, do_eval, find_k_hop_neighbors, build_graph
import random
import time
import logging

import os
import argparse
import pickle

import torch
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from utils.map import RoadNetworkMapFull


def infer(model, iterator, device, tf_ratio):
    preds = []
    truths = []
    preds_eta = []
    truths_eta = []
    preds_eta_seg = []
    truths_eta_seg = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            o_seg, d_seg, depart, lengths, seqs, trg_labels, trg_times, day, discrete_depart, seg_seqs, tt_label = batch

            o_seg = o_seg.to(device, non_blocking=True)
            d_seg = d_seg.to(device, non_blocking=True)
            depart = depart.to(device, non_blocking=True)
            day = day.to(device, non_blocking=True)
            discrete_depart = discrete_depart.to(device, non_blocking=True)

            seqs = seqs.permute(1, 0).to(device, non_blocking=True)
            trg_labels = trg_labels.permute(1, 0, 2).float().to(device, non_blocking=True)
            trg_times = trg_times.permute(1, 0).to(device, non_blocking=True)

            outputs_tte, outputs_prob, outputs_id, actual_length = model(o_seg, d_seg, depart, lengths, seqs, trg_times, day, discrete_depart, teacher_forcing_ratio=tf_ratio)

            outputs_id = outputs_id.permute(1, 0).detach().cpu().tolist()
            outputs_tte = outputs_tte.permute(1, 0).detach().cpu().tolist()
            for item, item_eta, real_eta, length in zip(outputs_id, outputs_tte, tt_label, actual_length.detach().cpu().tolist()):
                preds.append(item[:length])
                preds_eta.append(sum(item_eta[:length]))
                truths_eta.append(sum(real_eta))
                preds_eta_seg.append(item_eta[:length])
                truths_eta_seg.append(real_eta)

            truths.extend(seg_seqs)

    return preds, truths, preds_eta, truths_eta, preds_eta_seg, truths_eta_seg


def main():
    parser = argparse.ArgumentParser(description='TTE')
    parser.add_argument('--workspace', type=str, default="data/hrb1")
    parser.add_argument('--city', type=str, default='harbin')
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--tf_ratio', type=float, default=1.0)
    parser.add_argument('--decay_flag', action='store_true')
    parser.add_argument('--lambda1', type=int, default=10)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--small', action='store_true', default=True)
    parser.add_argument('--num_experts', type=int, default=8)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--sin_flag', action='store_true')
    parser.add_argument('--hist_flag', action='store_true')
    parser.add_argument('--observed_flag', action='store_true')
    parser.add_argument('--no_tim_flag', action='store_false')
    parser.add_argument('--graph_flag', action='store_true')
    parser.add_argument('--candi_future_flag', action='store_true')
    parser.add_argument('--scalability', type=float, default=1.0)

    opts = parser.parse_args()
    print(opts)

    device = torch.device(f"cuda:{opts.gpu_id}" if torch.cuda.is_available() else 'cpu')
    print(f"Use GPU: cuda {opts.gpu_id}")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)
    print('multi_task device', device)

    model_save_path = opts.model_old_path
    load_pretrained_flag = True

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=os.path.join(model_save_path, 'log.txt'),
                        filemode='a')

    city = opts.city
    if city in ["chengdu"]:
        zone_range = [30.6443, 104.0288, 30.7416, 104.1375]
        utc = 8
    elif city in ["xian"]:
        zone_range = [34.2060, 108.9058, 34.2825, 109.0049]
        utc = 8
    elif city in ["harbin"]:
        zone_range = [45.7104, 126.5769, 45.7992, 126.7084]
        utc = 0
    else:
        raise NotImplementedError

    workspace = opts.workspace
    rn = RoadNetworkMapFull(os.path.join(workspace, "roadnet"), zone_range=zone_range, unit_length=50)

    args = AttrDict()
    args_dict = {
        'device': device,
        'num_layers': opts.num_layers,
        'heads': opts.heads,

        # model params
        'hid_dim': opts.hid_dim,
        'id_emb_dim': 16,
        'dropout': 0.1,

        'gat_num_hidden': opts.hid_dim // 4,
        'gat_num_layers': 2,
        'gat_heads': [4, 4, 1],
        'edge_feature_dim': 64,
        'node_feature_dim': 64,

        'slot_size': 900,
        'window_size': 2,

        'id_size': rn.valid_edge_cnt_one,
        'utc': utc,

        'lambda1': opts.lambda1,
        'lambda2': opts.lambda2,
        'n_epochs': opts.epochs,
        'batch_size': opts.batch_size,
        'learning_rate': opts.lr,
        "lr_step": 2,
        "lr_decay": 0.8,
        'tf_ratio': opts.tf_ratio,
        'decay_flag': opts.decay_flag,
        'decay_ratio': 0.9,
        'clip': 1,
        'log_step': 1,

        'knbr': 2,

        'decoder_tim_flag': False,
        'decoder_des_flag': False,
        'rnn_tim_flag': opts.no_tim_flag,
        'decoder_future_flag': False,
        'attn_type': 'dot',

        'da_norm_flag': False,
        'direct_flag': True,
        'candi_hist_flag': False,
        'traffic_flag': False,
        'cos_threshold': (-math.sqrt(2)/2),

        'num_experts': opts.num_experts,
        'topk': opts.topk,
        'sin_flag': opts.sin_flag,
        'rnn_dist_flag': opts.hist_flag,
        'decoder_observed_flag': opts.observed_flag,
        'graph_flag': opts.graph_flag,
        'candi_future_flag': opts.candi_future_flag,
    }
    args.update(args_dict)
    print(args)
    logging.info(args_dict)

    node_res, _, edge_res, _, turning_res = pickle.load(open(os.path.join(workspace, "roadnet.pkl"), "rb"))
    args.node_attrs = np.array(node_res, dtype=np.float32)
    args.node_size = len(node_res)
    args.node_fea_dim = len(node_res[0])
    args.seg_attrs = np.array(edge_res, dtype=np.float32)
    args.seg_size = len(edge_res)
    args.seg_fea_dim = len(edge_res[0])
    edge_bp = []
    edge_ep = []
    edge_cord = []
    for item in rn.edgeCord:
        edge_bp.append(item[:2])
        edge_ep.append(item[-2:])
        idx1 = np.array(range(0, len(item), 2), dtype=int)
        idx2 = idx1 + 1
        tmp = np.array(item, dtype=np.float32)
        x = tmp[idx1]
        y = tmp[idx2]
        edge_cord.append([x.mean().item(), y.mean().item()])
    args.edge_cord = np.array(edge_cord, dtype=np.float32)
    args.edge_bp = np.array(edge_bp, dtype=np.float32)
    args.edge_ep = np.array(edge_ep, dtype=np.float32)

    road_speeds = pickle.load(open(os.path.join(workspace, "speed_profile.pkl"), "rb"))
    tt_distribution = pickle.load(open(os.path.join(workspace, "tt_distribution.pkl"), "rb"))
    G = pickle.load(open(os.path.join(workspace, "road_graph_wtime"), "rb"))
    edgeNbrs = []
    edge_nbrs = []
    nbr_mask = []
    ls_nbr3 = []
    mask_nbr3 = []
    ls_nbr1 = []
    mask_nbr1 = []
    edge_index = [torch.tensor([[0],[0]])]
    for i in range(len(G.nodes)):
        item = list(G.neighbors(i))
        if len(item) == 0:
            item = [i]
        edgeNbrs.append(item)
        edge_nbrs.append(torch.tensor(item, dtype=torch.int64))
        nbr_mask.append(torch.ones(len(item), dtype=torch.float32))
        succ_segs = list(find_k_hop_neighbors(G, i, k=args.knbr))
        if len(succ_segs) == 0:
            succ_segs = [i]
        ls_nbr3.append(torch.tensor(succ_segs, dtype=torch.int64))
        mask_nbr3.append(torch.ones(len(succ_segs), dtype=torch.float32))
        succ = list(G.successors(i))
        predec = list(G.predecessors(i))
        nbr1 = [i] + predec + succ
        ls_nbr1.append(torch.tensor(nbr1, dtype=torch.int64))
        mask_nbr1.append(torch.ones(len(nbr1), dtype=torch.float32))
        e_src, e_dst = build_graph(predec, succ, args.window_size, 1)
        edge_index.append(torch.tensor([e_src, e_dst]))
    args.edge_nbrs = edge_nbrs
    args.nbr_mask = nbr_mask
    rn.edgeNbrs = edgeNbrs
    args.ls_nbr3 = ls_nbr3
    args.mask_nbr3 = mask_nbr3
    args.ls_nbr1 = ls_nbr1
    args.mask_nbr1 = mask_nbr1
    args.road_speeds = road_speeds
    args.tt_distribution = tt_distribution
    args.edge_index = edge_index

    dam_data = pd.read_csv(os.path.join(workspace, "csm_all.txt"), sep=" ", header=None, names=['row', 'col', 'value'])
    dam_data['row'] += 1
    dam_data['col'] += 1
    if args.da_norm_flag:
        max_dam = dam_data['value'].max()
        dam_data['value'] = dam_data['value'] / max_dam * 10
    args.dam_data = dam_data.to_numpy(dtype=np.float32)

    test_trajs = pickle.load(open(os.path.join(workspace, 'test_trajs.pkl'), "rb"))
    if opts.small:
        test_trajs = test_trajs[:100000]
    test_dataset = TTEData(test_trajs, rn, args)
    logging.info('testing dataset shape: ' + str(len(test_dataset)))

    test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x), num_workers=opts.num_worker, pin_memory=True)

    model = torch.load(os.path.join(model_save_path, 'val-best-model.pt'), map_location=device)

    print('==> Model Loaded')

    print("==> Starting Prediction for ODTTE...")
    start_time = time.time()
    preds, truths, preds_eta, truths_eta, preds_eta_seg, truths_eta_seg = infer(model, test_iterator, device, tf_ratio=-1)
    end_time = time.time()
    epoch_secs = end_time - start_time
    print('Time: ' + str(epoch_secs) + 's')
    logging.info('Inference Time: {}, {}, {}'.format(end_time - start_time, (end_time - start_time) / len(test_dataset) * 1000, len(test_dataset) / (end_time - start_time)))
    pickle.dump((preds, truths, preds_eta, truths_eta), open(os.path.join(model_save_path, 'infer_odtte.pkl'), "wb"))

    prec, rec, f1, jac, reach = do_eval(preds, truths, edge_cord)
    logging.info('\tTest Precision:' + str(prec) +
                 '\tTest Recall:' + str(rec) +
                 '\tTest F1:' + str(f1) +
                 '\tTest Jaccard:' + str(jac) +
                 '\tTest Reach:' + str(reach))

    print("==> Starting Evaluation...")
    metrics = cal_regression_metric(truths_eta, preds_eta, unit='minute')
    metrics_sec = cal_regression_metric(truths_eta, preds_eta, unit='second')

    logging.info('Time: ' + str(epoch_secs) + 's')
    logging.info('\tTest Evaluation Metrics:' + str(metrics))
    logging.info('\tTest Evaluation Metrics (sec):' + str(metrics_sec))


if __name__ == '__main__':
    main()
