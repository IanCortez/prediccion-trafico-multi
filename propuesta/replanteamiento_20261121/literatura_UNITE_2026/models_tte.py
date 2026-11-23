import random

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset

from models.layers import sequence_mask, ODTEncoder, NextSegPred, SegTimeEstimator
from torch.nn import MultiheadAttention as MHA


def collate_fn(data):
    o_seg, d_seg, depart, seqs, trg_labels, trg_times, day, discrete_depart, seg_seqs, tt_label = zip(*data)

    o_seg = torch.tensor(o_seg)
    d_seg = torch.tensor(d_seg)
    discrete_depart = torch.vstack(discrete_depart)
    depart = torch.tensor(depart)
    day = torch.tensor(day)

    lengths = [len(item) for item in seqs]
    seqs = rnn_utils.pad_sequence(seqs, batch_first=True, padding_value=-1)
    trg_labels = rnn_utils.pad_sequence(trg_labels, batch_first=True, padding_value=0)
    trg_times = rnn_utils.pad_sequence(trg_times, batch_first=True, padding_value=0)

    return o_seg, d_seg, depart, lengths, seqs, trg_labels, trg_times, day, discrete_depart, seg_seqs, tt_label


class TTEData(Dataset):

    def __init__(self, trajs, rn, params):
        self.rn = rn
        self.trajs = trajs
        self.parameters = params
        out_degrees = [len(item) for item in rn.edgeNbrs]
        self.maxDeg = max(out_degrees)
        self.slot_size = params.slot_size

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj = self.trajs[index]
        seg_seq, departure_time, node_seq, real_time, tt_label, daytime_norm, offset, times, ls_pt, speed_seq = traj

        o_seg = seg_seq[0]
        d_seg = seg_seq[-1]

        ts = ls_pt[0].time_arr
        discrete_depart = [ts.month, ts.day, ts.weekday(), ts.hour, ts.minute, ts.second]
        discrete_depart = torch.tensor(discrete_depart, dtype=torch.float32)
        depart = [(ts.hour * 3600 + ts.minute * 60 + ts.second) // self.slot_size]
        depart = torch.tensor(depart, dtype=torch.long)
        day = [ts.weekday()]
        day = torch.tensor(day, dtype=torch.long)

        seq = torch.tensor(seg_seq)
        trg_idx = self.get_trg_idx(seg_seq[:-1], seg_seq[1:])
        trg_label = F.one_hot(torch.tensor(trg_idx), num_classes=self.maxDeg)
        trg_time = torch.tensor(tt_label, dtype=torch.float32)

        return o_seg, d_seg, depart, seq, trg_label, trg_time, day, discrete_depart, seg_seq, tt_label

    def get_trg_idx(self, src_seq, trg_seq):
        idx = []
        for src, trg in zip(src_seq, trg_seq):
            idx.append(self.rn.edgeNbrs[src].index(trg))
        return idx

    def get_time_idx(self, pt_list):
        slots = []
        for pt in pt_list:
            time_arr = pt.time_arr
            idx = (time_arr.hour * 60 * 60 + time_arr.minute * 60 + time_arr.second) // self.slot_size
            slots.append(idx)
        return slots


class UNITE(nn.Module):

    def __init__(self, parameters):
        super().__init__()

        self.rnn_dist_flag = parameters.rnn_dist_flag
        self.rnn_tim_flag = parameters.rnn_tim_flag
        self.decoder_des_flag = parameters.decoder_des_flag
        self.decoder_tim_flag = parameters.decoder_tim_flag
        self.decoder_observed_flag = parameters.decoder_observed_flag
        self.decoder_future_flag = parameters.decoder_future_flag

        self.seg_size = parameters.seg_size
        self.id_size = parameters.id_size
        self.freq_limit = 1

        self.odt_encoder = ODTEncoder(parameters)
        self.num_slots = 24 * 60 * 60 // parameters.slot_size
        self.emb_departure = nn.Embedding(self.num_slots * 7, parameters.id_emb_dim)
        self.window_size = parameters.window_size

        self.candi_encoder = NextSegPred(parameters)

        self.emb_id = nn.Embedding(parameters.seg_size + 1, parameters.id_emb_dim)

        seg_input_dim = parameters.id_emb_dim
        if self.rnn_tim_flag:
            seg_input_dim += parameters.id_emb_dim
        if self.rnn_dist_flag:
            tt_dists = torch.from_numpy(parameters.tt_distribution)
            tt_dist_dim = tt_dists.shape[-1]
            self.tt_dists = torch.cat([torch.zeros_like(tt_dists[:, :1]), tt_dists], dim=1)
            self.tt_distribution_linear = nn.Linear(tt_dist_dim, parameters.hid_dim // 2)
            seg_input_dim += parameters.hid_dim // 2
        rnn_input_dim = parameters.hid_dim
        if self.decoder_des_flag:
            rnn_input_dim += parameters.hid_dim
        if self.decoder_tim_flag:
            rnn_input_dim += parameters.hid_dim
        if self.decoder_observed_flag:
            self.attn_observed = MHA(parameters.hid_dim, parameters.heads, parameters.dropout, batch_first=False)
            rnn_input_dim += parameters.hid_dim
        if self.decoder_future_flag:
            self.emb_future = nn.Embedding(parameters.seg_size + 1, parameters.id_emb_dim * 4)
            nbrs3 = 1 + rnn_utils.pad_sequence(parameters.ls_nbr3, batch_first=True, padding_value=-1)
            mask_nbr3 = rnn_utils.pad_sequence(parameters.mask_nbr3, batch_first=True, padding_value=0)
            self.max_num3 = nbrs3.shape[1]
            self.nbrs3 = torch.cat([torch.zeros((1, nbrs3.shape[1]), dtype=torch.int64), nbrs3], dim=0)
            self.mask_nbr3 = torch.cat([torch.zeros((1, nbrs3.shape[1]), dtype=torch.float32), mask_nbr3], dim=0)
            self.attn_future = MHA(parameters.hid_dim, parameters.heads, parameters.dropout, kdim=parameters.id_emb_dim * 4, vdim=parameters.id_emb_dim * 4, batch_first=True)
            rnn_input_dim += parameters.hid_dim

        self.all_mlp = nn.Sequential(
            nn.Linear(seg_input_dim, parameters.hid_dim),
            nn.ReLU()
        )
        self.num_layers = parameters.num_layers
        self.hid_dim = parameters.hid_dim
        self.rnn = nn.GRU(rnn_input_dim, parameters.hid_dim, num_layers=self.num_layers, batch_first=False)

        self.eta_fuse_mlp = nn.Sequential(
            nn.Linear(parameters.hid_dim * 2, parameters.hid_dim),
            nn.ReLU()
        )

        fc_id_out_input_dim = parameters.hid_dim * 3
        self.fc_id_out = nn.Sequential(
            nn.Linear(fc_id_out_input_dim, parameters.hid_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=parameters.dropout),
            nn.Linear(parameters.hid_dim * 2, 1),
        )

        self.eta_predictor = SegTimeEstimator(parameters)

        self.init_weights()

    def to(self, device):
        super().to(device)
        self.candi_encoder.to(device)
        self.eta_predictor.to(device)
        if self.rnn_dist_flag:
            self.tt_dists = self.tt_dists.to(device)
        if self.decoder_future_flag:
            self.nbrs3 = self.nbrs3.to(device)
            self.mask_nbr3 = self.mask_nbr3.to(device)

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def get_dest_info(self, dest, slots_all):
        batch_size = dest.shape[0]

        idx = torch.arange(batch_size, dtype=torch.int, device=dest.device)
        ones = torch.ones(batch_size, dtype=torch.int, device=dest.device)
        des_sparse = torch.sparse_coo_tensor(indices=torch.vstack([idx, dest]), values=ones, size=(batch_size, self.id_size), dtype=torch.float32, device=dest.device).to_sparse_csr()
        des_cols = torch.matmul(des_sparse, self.candi_encoder.dam_col).to_dense()

        dest_geo = self.candi_encoder.cord_edge(dest)

        t_right = torch.arange(1, self.window_size + 1, device=dest.device)
        t_left = t_right - (self.window_size + 1)
        t_add = torch.cat([t_left, t_right], dim=-1).repeat(batch_size, 1)
        t_idx = slots_all.unsqueeze(-1).repeat(1, t_add.shape[-1]) + t_add
        t_idx = t_idx % (self.num_slots * 7)

        diff = torch.tensor([0, -self.num_slots, self.num_slots], device=dest.device).reshape(1, 3, 1).repeat(batch_size, 1, 1)
        t_graph = torch.cat([slots_all.unsqueeze(-1), t_idx], dim=-1).unsqueeze(1).repeat(1, 3, 1)
        t_graph = (t_graph + diff) % (self.num_slots * 7)

        return des_cols, idx, t_idx, dest_geo, t_graph

    def get_seg_emb(self, input_id, slots_all, t_emb):
        rnn_input = self.emb_id(input_id)
        if self.rnn_tim_flag:
            rnn_input = torch.cat([rnn_input, t_emb], dim=-1)
        if self.rnn_dist_flag:
            tt_dist = self.tt_dists[slots_all, input_id]
            dist_emb = self.tt_distribution_linear(tt_dist)
            rnn_input = torch.cat([rnn_input, dist_emb], dim=-1)
        recurrent_input = self.all_mlp(rnn_input)
        return recurrent_input

    def prog_state_encoding(self, input_id, curr_emb, hidden, seg_embs, t):
        rnn_input = curr_emb.clone()
        if self.decoder_observed_flag:
            prev_emb = seg_embs[:t].clone()

            observed_emb, _ = self.attn_observed(curr_emb.unsqueeze(0), prev_emb, prev_emb, key_padding_mask=None, need_weights=False)
            observed_emb = observed_emb.squeeze(0)
            rnn_input = torch.cat([rnn_input, observed_emb], dim=-1)
        if self.decoder_future_flag:
            mask_nbrs3 = self.mask_nbr3[input_id]
            nbrs3_lens = torch.clip(mask_nbrs3.sum(-1).long(), min=1)
            nbr_valid_len = nbrs3_lens.max()
            mask_nbrs3 = ~mask_nbrs3[:, :nbr_valid_len].bool()
            mask_nbrs3[:, 0] = False

            nbrs3 = self.nbrs3[input_id][:, :nbr_valid_len]
            nbrs3_emb = self.emb_future(nbrs3)

            future_emb, _ = self.attn_future(curr_emb.unsqueeze(1), nbrs3_emb, nbrs3_emb, key_padding_mask=mask_nbrs3, need_weights=False)
            future_emb = future_emb.squeeze(1)
            rnn_input = torch.cat([rnn_input, future_emb], dim=-1)

        rnn_input = rnn_input.unsqueeze(0)

        output, hidden = self.rnn(rnn_input, hidden)
        curr_out = output.squeeze(0)

        return curr_out, hidden

    def forward(self, o_seg, d_seg, depart, lengths, seqs, trg_times, day, discrete_depart, teacher_forcing_ratio):
        o_seg = 1 + o_seg
        d_seg = 1 + d_seg
        seqs = 1 + seqs
        max_trg_len, batch_size = seqs.shape
        device = seqs.device

        if teacher_forcing_ratio == -1:
            max_trg_len = 50

        outputs_prob = torch.zeros([max_trg_len, batch_size, self.candi_encoder.max_deg], device=device)
        outputs_id = torch.zeros([max_trg_len, batch_size], dtype=torch.int64, device=device)
        outputs_id[0] = o_seg

        seg_used = torch.zeros([batch_size, self.seg_size + 1], dtype=torch.int8, device=device)
        rows = torch.arange(batch_size, dtype=torch.int64, device=device)
        seg_used[rows, o_seg] = 1

        stop = torch.zeros([batch_size], dtype=torch.bool, device=device)
        actual_length = max_trg_len * torch.ones([batch_size], dtype=torch.int64, device=device)

        outputs_tte = torch.zeros([max_trg_len, batch_size], device=device)

        odt_emb = self.odt_encoder(o_seg, d_seg, depart, discrete_depart)
        t_emb = self.emb_departure(depart)
        slots_all = day * self.num_slots + depart
        des_emb = self.get_seg_emb(d_seg, slots_all, t_emb)

        seg_embs = torch.zeros([max_trg_len, batch_size, self.hid_dim], dtype=torch.float32, device=device)
        seg_embs[0] = des_emb

        des_cols, idx, t_idx, d_cords, t_graph = self.get_dest_info(d_seg, slots_all)

        hidden = odt_emb.unsqueeze(0).repeat(self.num_layers, 1, 1)

        input_id = o_seg
        for t in range(1, max_trg_len):
            curr_emb = self.get_seg_emb(input_id, slots_all, t_emb)
            seg_embs[t] = curr_emb

            prediction_emb, hidden = self.prog_state_encoding(input_id, curr_emb, hidden, seg_embs, t)

            pred_eta, emb_y = self.eta_predictor(odt_emb, prediction_emb, input_id, slots_all, t_idx, t_graph)

            outputs_tte[t - 1] = pred_eta

            if teacher_forcing_ratio <= 1:
                candi_emb, nbrs, nbrs_mask, da_values, cos_values = self.candi_encoder(input_id, des_cols, idx, d_cords, t_idx, slots_all)

                prediction_emb = self.eta_fuse_mlp(torch.cat([prediction_emb, emb_y], dim=-1))

                inputs = torch.cat([candi_emb, prediction_emb.unsqueeze(1).repeat(1, candi_emb.shape[1], 1), des_emb.unsqueeze(1).repeat(1, candi_emb.shape[1], 1)], dim=-1)
                scores = self.fc_id_out(inputs).squeeze(-1)

                scores = scores.masked_fill(nbrs_mask == 0, -1e6)
                probs = torch.sigmoid(scores)

                if teacher_forcing_ratio > 0:
                    pred_id = (F.one_hot(probs.argmax(dim=1), nbrs.shape[1]) * nbrs).sum(-1)
                else:
                    _, idxs = probs.topk(self.candi_encoder.max_deg, dim=1)
                    idx1 = idxs[:, 0]
                    pred_id = nbrs[rows, idx1]
                    for rank in range(1, self.candi_encoder.max_deg):
                        idx2 = idxs[:, rank]
                        next2 = nbrs[rows, idx2]
                        condition = seg_used[rows, pred_id] < self.freq_limit
                        if condition.all():
                            break
                        pred_id = torch.where(condition, pred_id, next2)

                    des_condi = (d_seg.unsqueeze(-1) == nbrs).sum(-1) > 0
                    pred_id = torch.where(des_condi, d_seg, pred_id)
                    tmp_length = torch.ones([batch_size], dtype=torch.int64, device=device) * (t + 1)
                    actual_length = torch.where(des_condi & (~stop), tmp_length, actual_length)
                    nbr_condi = nbrs_mask.sum(-1) == 0
                    tmp_length = torch.ones([batch_size], dtype=torch.int64, device=hidden.device) * t
                    actual_length = torch.where(nbr_condi & (~stop), tmp_length, actual_length)
                    stop = des_condi | nbr_condi | stop
                    seg_used[rows, pred_id] += 1

                outputs_prob[t] = probs
                outputs_id[t] = pred_id

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if teacher_force:
                input_id = seqs[t]
            else:
                input_id = pred_id

            if stop.all():
                break

        t = max_trg_len

        curr_emb = self.get_seg_emb(input_id, slots_all, t_emb)

        prediction_emb, hidden = self.prog_state_encoding(input_id, curr_emb, hidden, seg_embs, t)

        pred_eta, emb_y = self.eta_predictor(odt_emb, prediction_emb, input_id, slots_all, t_idx, t_graph)
        outputs_tte[t - 1] = pred_eta

        outputs_id = outputs_id - 1
        if teacher_forcing_ratio == -1:
            lengths_ttl = actual_length
        else:
            lengths_ttl = torch.tensor(lengths, device=hidden.device)
            mask_trg = torch.ones([batch_size, max_trg_len], device=hidden.device)
            mask_trg = sequence_mask(mask_trg, lengths_ttl)
            outputs_prob = outputs_prob.permute(1, 0, 2)  # batch size, seq len, 1
            outputs_prob = outputs_prob.masked_fill(mask_trg.unsqueeze(-1) == 0, 0)
            outputs_prob = outputs_prob.permute(1, 0, 2)
            outputs_tte = outputs_tte.permute(1, 0)  # batch size, seq len, 1
            outputs_tte = outputs_tte.masked_fill(mask_trg == 0, 0)
            outputs_tte = outputs_tte.permute(1, 0)
            actual_length = lengths_ttl
        return outputs_tte, outputs_prob, outputs_id, actual_length
