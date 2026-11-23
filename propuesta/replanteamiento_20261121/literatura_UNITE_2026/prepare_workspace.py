import argparse
import csv
import json
import os
import pickle
from ast import literal_eval
from collections import Counter

import numpy as np
from numpy.linalg import norm
import geopandas as gpd
import datetime as dt

import pandas as pd
from tqdm import tqdm
import sys

from dam import txt2npy, gen_dam, get_connection_strength_matrix_sparse
from seg_info import get_road_graph, rt_dict, get_road_type, get_road_type2, get_speed
from util import find_k_hop_neighbors
from utils.trajectory_func import STPoint

csv.field_size_limit(sys.maxsize)


def reformat_roadnet(map_dir, zone_range):
    min_lat, min_long, max_lat, max_long = zone_range
    nodes = gpd.read_file(os.path.join(map_dir, "nodes.shp"))
    edges = gpd.read_file(os.path.join(map_dir, "edges.shp"))

    node_res = []
    output_edge_dict = {}
    input_edge_dict = {}
    node2id = {}

    for i in tqdm(range(nodes.shape[0]), desc='node num'):
        node = nodes.iloc[i]
        k = node['osmid']
        input_list = edges.query('v == {}'.format(k))['fid'].tolist()
        output_list = edges.query('u == {}'.format(k))['fid'].tolist()
        input_edge_dict[i] = input_list
        output_edge_dict[i] = output_list

        long, lat = node['x'], node['y']
        assert long >= min_long and long <= max_long
        assert lat >= min_lat and lat <= max_lat
        features = [(long - min_long) / (max_long - min_long),
                    (lat - min_lat) / (max_lat - min_lat),
                    len(input_list),
                    len(output_list)]
        node_res.append(features)
        node2id[k] = i
    assert nodes.shape[0] == len(node_res) == len(node2id)

    idx = 0
    edge2node_dict = []
    edge_res = []
    turning_dict = {}
    for i in tqdm(range(edges.shape[0]), desc='edge num'):
        edge = edges.iloc[i]
        k = edge['fid']
        src = edge['u']
        trg = edge['v']
        nbrs = edges.query('u == {}'.format(trg))['fid'].tolist()
        for item in nbrs:
            turning_dict[(k, item)] = idx
            idx += 1
        edge2node_dict.append([node2id[src], node2id[trg]])

        features = node_res[node2id[src]][:2] + node_res[node2id[trg]][:2]
        rt = get_road_type(edge['highway'])
        rt_onehot = [0 for _ in range(len(set(rt_dict.values())))]
        rt_onehot[rt] = 1

        oneway = [0, 0]
        if edge['oneway']:
            oneway[1] = 1
        else:
            oneway[0] = 1
        features += rt_onehot + oneway

        length = edge['length'] / 1000
        maxspeed = get_speed(rt) * 3600 / 1000 / 100  # km/h/100
        features.append(length)
        features.append(maxspeed)
        edge_res.append(features)
    assert edges.shape[0] == len(edge_res) == len(edge2node_dict)

    return node_res, node2id, edge_res, edge2node_dict


def get_time_seq(ts):
    ts = np.array(ts)
    daytime = ts % (24 * 60 * 60)
    daytime_norm = daytime / 60 / 60 / 24
    offset = (ts - ts[0]) / (ts[-1] - ts[0])
    return list(daytime_norm), list(offset)


def check_src_dest(src, src_offset, dest, dest_offset, edge_lengths, confidence=0.1):
    src_ratio = src_offset / edge_lengths[src]
    dest_ratio = dest_offset / edge_lengths[dest]
    src_flag = src_ratio < confidence
    dest_flag = dest_ratio >= (1 - confidence)
    return src_flag, dest_flag


def reformat_trajs(file, tz, edge2node_dict, node_gps, rm_ht, edge_lengths):
    trajs = []
    data_rows = []
    total = 0
    reader = csv.reader(open(file, "r"))
    for line in tqdm(reader, desc='line num'):
        total += 1
        path = literal_eval(line[3])
        if rm_ht:
            offsets = literal_eval(line[2])
            src_flag, dest_flag = check_src_dest(path[0][0], offsets[0][0], path[-1][0], offsets[1][0], edge_lengths)
            if not src_flag:
                path = path[1:]
            if not dest_flag:
                path = path[:-1]
        trajs.append(path)
        seg_seq, ts_seq, speed_seq, travel_time = zip(*path)
        tt_label = [item / 60 for item in travel_time]
        real_time = sum(tt_label)

        node_seq = [edge2node_dict[seg_seq[0]][0]]
        for seg in seg_seq:
            srctrg = edge2node_dict[seg]
            assert srctrg[0] == node_seq[-1]
            node_seq.append(srctrg[1])

        times = list(ts_seq) + [ts_seq[-1] + travel_time[-1]]
        daytime_norm, offset_seq = get_time_seq(times)

        if daytime_norm[-1] > daytime_norm[0]:
            pred_time = (daytime_norm[-1] - daytime_norm[0]) * 24 * 60
        else:
            pred_time = (daytime_norm[-1] + 1 - daytime_norm[0]) * 24 * 60
        assert abs(pred_time - real_time) < 1e-1

        pt_list = []
        for idx, (nid, ts) in enumerate(zip(node_seq, times)):
            pt = STPoint(node_gps[nid][1], node_gps[nid][0], ts)
            pt.time_arr = dt.datetime.fromtimestamp(ts, tz)
            pt.cpath_idx = min(idx, len(path) - 1)
            pt_list.append(pt)
        assert abs((pt_list[-1].time - pt_list[0].time) / 60 - real_time) < 1e-1

        tmp = [list(seg_seq), ts_seq[0], node_seq, real_time, tt_label, daytime_norm, offset_seq, times, pt_list, speed_seq]
        data_rows.append(tmp)
    print(len(data_rows), total)
    return data_rows, trajs


def gen_map(map_dir, workspace):
    out_dir = os.path.join(workspace, "roadnet/")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    nodes = gpd.read_file(os.path.join(map_dir, "nodes.shp"))
    index = [i for i in range(nodes.shape[0])]
    nodes["fid"] = np.array(index, dtype=int)
    data = []
    nid_dict = {}
    for i in tqdm(range(nodes.shape[0]), desc="node num"):
        tmp = nodes.iloc[i]
        osmid = int(tmp['osmid'])
        fid = int(tmp['fid'])
        x = float(tmp['x'])
        y = float(tmp['y'])

        nid_dict[osmid] = fid
        data.append([fid, y, x])
    with open(os.path.join(out_dir, "nodeOSM.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(data)

    edges = gpd.read_file(os.path.join(map_dir, "edges.shp"))
    zone = [180, -180, 90, -90]
    rn_dict = {}
    data = []
    wayType = []
    for i in tqdm(range(edges.shape[0]), desc='edge num'):
        tmp = edges.iloc[i]
        eid = int(tmp['fid'])
        u = int(tmp['u'])
        v = int(tmp['v'])
        points = tmp['geometry'].coords
        zone[0] = min(zone[0], np.min(points.xy[0]))
        zone[1] = max(zone[1], np.max(points.xy[0]))
        zone[2] = min(zone[2], np.min(points.xy[1]))
        zone[3] = max(zone[3], np.max(points.xy[1]))

        desc, code = get_road_type2(tmp['highway'])
        wayType.append([eid, desc, code])

        row = [eid, nid_dict[u], nid_dict[v]]
        row.append(len(points))
        pts = []
        for lon, lat in points:
            row += [float(lat), float(lon)]
            pts.append([float(lat), float(lon)])
        data.append(row)

        tmp_dict = {"coords": pts, "length": float(tmp['length']), "level": code}
        rn_dict[eid] = tmp_dict
    print(zone)
    with open(os.path.join(out_dir, "rn_dict.json"), 'w') as fp:
        json.dump(rn_dict, fp)
    with open(os.path.join(out_dir, "edgeOSM.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(data)
    with open(os.path.join(out_dir, "wayTypeOSM.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(wayType)


def get_time_idx(time_arr, slot_size):
    num_slots = 24 * 60 * 60 // slot_size
    idx = time_arr.weekday() * num_slots + (time_arr.hour * 60 * 60 + time_arr.minute * 60 + time_arr.second) // slot_size
    return int(idx)


def get_speed_profile(road_speeds, slot_size):
    num_slots = 24 * 60 * 60 // slot_size
    num_segs = len(road_speeds)
    speed_profile = np.zeros((num_segs, num_slots * 7), dtype=np.float32)
    for seg, speeds in enumerate(road_speeds):
        cates = [[] for _ in range(num_slots * 7)]
        for v, t in speeds:
            idx = get_time_idx(t, slot_size)
            cates[idx].append(v)
        for idx, item in enumerate(cates):
            if len(item) > 0:
                speed_profile[seg, idx] = np.mean(item)
    return speed_profile


def fill_missing_speed(speed_profile, G):

    def cosine_filling(seg):
        succ_segs = find_k_hop_neighbors(G, seg, k=3)
        predec_segs = find_k_hop_neighbors(G, seg, k=3, reverse=True)
        seg_mask = masks[seg]
        seg_speed = speed_profile[seg]
        nbrs = []
        for item in (succ_segs | predec_segs):
            tmp_mask = (masks[item] & seg_mask).sum()
            if tmp_mask > num_slots * 0.25:
                tmp_speed = speed_profile[item]
                similarity = np.dot(seg_speed, tmp_speed) / (norm(seg_speed) * norm(tmp_speed))
                nbrs.append([item, similarity])
        nbrs.sort(key=lambda elem: elem[1], reverse=True)
        indexs = np.where(seg_mask == 0)[0]
        for empty in indexs:
            value = 0
            for nbr, _ in nbrs:
                value = speed_profile[nbr, empty]
                if value > 0:
                    break
            seg_speed[empty] = value

    def stnbr_filling(seg):
        seg_mask = masks[seg]
        seg_speed = speed_profile[seg]
        indexs = np.where(seg_mask == 0)[0]
        nbr_dict = {}
        for empty in indexs:
            values = []
            k = 1
            while len(values) == 0 and k < 10:
                if k not in nbr_dict:
                    succ_segs = find_k_hop_neighbors(G, seg, k=k)
                    predec_segs = find_k_hop_neighbors(G, seg, k=k, reverse=True)
                    nbr_dict[k] = succ_segs | predec_segs | {seg}
                nbrs = nbr_dict[k]
                slots = []
                for i in range(empty - k, empty + k + 1):
                    if i < num_slots:
                        slots.append(i)
                    else:
                        slots.append(i % num_slots)

                for item in nbrs:
                    tmp_mask = (masks[item] & seg_mask).sum()
                    if tmp_mask > 0:
                        for slot in slots:
                            v = speed_profile[item, slot]
                            if v > 0:
                                values.append(v)
                k += 2
            if len(values) > 0:
                seg_speed[empty] = np.mean(values)

    num_slots = speed_profile.shape[1]

    masks = speed_profile > 0
    valid_num = masks.sum(axis=1)
    num_fill = []
    for seg, item in enumerate(valid_num):
        if 0 < item < num_slots:
            num_fill.append([seg, item])
    num_fill.sort(key=lambda elem: elem[1], reverse=True)

    for seg, num in tqdm(num_fill, desc='missing num'):
        if num > num_slots * 0.25:
            cosine_filling(seg)
            if (speed_profile[seg] > 0).sum() < num_slots:
                stnbr_filling(seg)
        else:
            stnbr_filling(seg)
    return speed_profile


def get_speed_distribution(road_speeds, edge_lengths, slot_size, k=5):
    segment_travel_time_dict = {}
    cnt = 0
    for seg, speeds in tqdm(enumerate(road_speeds), desc='seg num'):
        length = edge_lengths[seg]
        buckets = {}
        for v, departure in speeds:
            start_day = departure.weekday()
            slot_id = (departure.hour * 3600 + departure.minute * 60 + departure.second) // slot_size

            if (start_day, slot_id) not in buckets:
                buckets[(start_day, slot_id)] = []

            travel_time = length / v
            buckets[(start_day, slot_id)].append(round(travel_time))
        cnt += len(buckets)
        for (day, slot_id), value in buckets.items():
            num_samples = len(value)
            mean = sum(value) / num_samples
            counter = Counter(value)
            res = []
            if len(counter) <= k:
                topk = counter.most_common(k)
                for sec, freq in topk:
                    prob = freq / num_samples
                    res.extend([sec, prob])
                res.extend([0, 0] * (k - len(topk)))
            else:
                tmp = [list(item) for item in counter.items()]
                tmp.sort(key=lambda elem: elem[1], reverse=True)
                max_freq = tmp[0][1]
                heads = []
                for item in tmp:
                    if item[1] == max_freq:
                        heads.extend([item[0]] * max_freq)
                anchor = np.median(heads).item()
                for item in tmp:
                    dis = abs(anchor - item[0])
                    item.append(dis)
                tmp.sort(key=lambda elem: elem[2], reverse=False)
                tmp.sort(key=lambda elem: elem[1], reverse=True)
                for sec, freq, _ in tmp[:k]:
                    prob = freq / num_samples
                    res.extend([sec, prob])
            res.append(mean)
            assert len(res) == (k * 2 + 1)
            segment_travel_time_dict[day, slot_id, seg] = res
    assert cnt == len(segment_travel_time_dict)
    return segment_travel_time_dict


def get_speed_tensor(segment_travel_time_dict, speed_profile, G, slot_size):
    num_slots = 24 * 60 * 60 // slot_size
    num_segs = speed_profile.shape[0]
    dim = len(list(segment_travel_time_dict.values())[0])

    speed_distribution = np.zeros((7 * num_slots, num_segs, dim), dtype=np.float32)
    for w in tqdm(range(7), desc='day num'):
        for slot in range(num_slots):
            key = w * num_slots + slot
            for seg in range(num_segs):
                if (w, slot, seg) in segment_travel_time_dict:
                    speed_distribution[key, seg] = segment_travel_time_dict[(w, slot, seg)]
                else:
                    speed = speed_profile[seg, key]
                    if speed > 0:
                        length = G.nodes[seg]['length']
                        travel_time = length / speed
                    else:
                        travel_time = G.nodes[seg]['time']
                    value = np.zeros(dim, dtype=np.float32)
                    value[0] = travel_time
                    value[1] = 1
                    value[-1] = travel_time
                    speed_distribution[key, seg] = value
    return speed_distribution


def gen_speed(data_dir, edge_num, tz, rm_ht, edge_lengths):
    files = [os.path.join(data_dir, "traj_train_100")]

    data_rows = [[] for _ in range(edge_num)]
    for file in files:
        reader = csv.reader(open(file, "r"))
        cnt = 0
        for line in tqdm(reader, desc='traj num'):
            cnt += 1
            path = literal_eval(line[3])
            offsets = literal_eval(line[2])
            src_flag, dest_flag = check_src_dest(path[0][0], offsets[0][0], path[-1][0], offsets[1][0], edge_lengths)
            if rm_ht:
                if not src_flag:
                    path = path[1:]
                if not dest_flag:
                    path = path[:-1]
            for seg in path:
                eid = seg[0]
                ts = seg[1]
                tm = dt.datetime.fromtimestamp(ts, tz)
                speed = seg[2]
                data_rows[eid].append([speed, tm])
        print("# Input Trajectories: {}, {}".format(cnt, len(data_rows)))
    return data_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/dataset/dataset_eta/hrb_chrono")
    parser.add_argument('--workspace', type=str, default="data/hrb1")
    parser.add_argument('--map_dir', type=str, default="/dataset/RoadNetwork/Harbin")
    parser.add_argument('--city', type=str, default='harbin')
    parser.add_argument('--scale', type=int, default=100)
    parser.add_argument('--slot_size', type=int, default=900)
    parser.add_argument('--mode', type=int, default=1)
    parser.add_argument('--rmht', type=int, default=1)
    args = parser.parse_args()
    print(args)

    city = args.city
    if city in ["chengdu"]:
        zone_range = [30.6443, 104.0288, 30.7416, 104.1375]
        utc = 8
    elif city in ["xian"]:
        zone_range = [34.2060, 108.9058, 34.2825, 109.0049]
        utc = 8
    elif city in ["harbin"]:
        zone_range = [45.7104, 126.5769, 45.7992, 126.7084]
        ts = 60
        utc = 0
    else:
        raise NotImplementedError
    tz = dt.timezone(dt.timedelta(hours=utc))

    if args.mode == 1:
        node_res, node2id, edge_res, edge2node_dict = reformat_roadnet(args.map_dir, zone_range)
        pickle.dump((node_res, node2id, edge_res, edge2node_dict), open(os.path.join(args.workspace, "roadnet.pkl"), "wb"))

        nodes = gpd.read_file(os.path.join(args.map_dir, "nodes.shp"))
        node_gps = nodes[['x', 'y']].to_numpy().tolist()
        edges = gpd.read_file(os.path.join(args.map_dir, "edges.shp"))
        edge_lengths = edges['length'].to_numpy().tolist()

        train_file = os.path.join(args.data_dir, 'traj_train_{}'.format(args.scale))
        train_data, train_trajs = reformat_trajs(train_file, tz, edge2node_dict, node_gps, args.rmht, edge_lengths)
        pickle.dump(train_data, open(os.path.join(args.workspace, "train_trajs.pkl"), "wb"))
        _, _, node_seqs, *_ = zip(*train_data)
        node_seqs = list(node_seqs) + edge2node_dict
        del train_data

        unigram, mat_a = get_connection_strength_matrix_sparse(node_seqs, nodes.shape[0])
        with open(os.path.join(args.workspace, "csm_node.txt"), 'w') as fp:
            fields_output_file = csv.writer(fp, delimiter=' ')
            fields_output_file.writerows(mat_a)
        print("Saved csm_node.txt")

        gen_dam(train_trajs, args.map_dir, args.workspace, utc)
        txt2npy(args.workspace)
        get_road_graph(args.workspace)
        gen_map(args.map_dir, args.workspace)

        valid_file = os.path.join(args.data_dir, 'traj_valid')
        valid_data, _ = reformat_trajs(valid_file, tz, edge2node_dict, node_gps, args.rmht, edge_lengths)
        pickle.dump(valid_data, open(os.path.join(args.workspace, "valid_trajs.pkl"), "wb"))
        del valid_data

        test_file = os.path.join(args.data_dir, 'traj_test')
        test_data, _ = reformat_trajs(test_file, tz, edge2node_dict, node_gps, args.rmht, edge_lengths)
        pickle.dump(test_data, open(os.path.join(args.workspace, "test_trajs.pkl"), "wb"))
        del test_data

        edges = gpd.read_file(os.path.join(args.map_dir, "edges.shp"))
        edge_lengths = edges['length'].to_numpy().tolist()
        edge_num = edges.shape[0]
        tz = dt.timezone(dt.timedelta(hours=utc))
        traffic = gen_speed(args.data_dir, edge_num, tz, rm_ht=args.rmht, edge_lengths=edge_lengths)
        pickle.dump(traffic, open(os.path.join(args.data_dir, "road_speeds{}.pkl".format('_rmht' if args.rmht else '')), "wb"))
    elif args.mode == 2:
        G = pickle.load(open(os.path.join(args.workspace, "road_graph_wtime"), "rb"))

        road_speeds = pickle.load(open(os.path.join(args.data_dir, "road_speeds{}.pkl".format('_rmht' if args.rmht else '')), "rb"))
        speed_profile = get_speed_profile(road_speeds, args.slot_size)

        speed_profile = fill_missing_speed(speed_profile, G)
        pickle.dump(speed_profile, open(os.path.join(args.workspace, "speed_profile.pkl"), "wb"))

        edges = gpd.read_file(os.path.join(args.map_dir, "edges.shp"))
        edge_lengths = edges['length'].to_numpy().tolist()
        segment_travel_time_dict = get_speed_distribution(road_speeds, edge_lengths, args.slot_size)
        pickle.dump(segment_travel_time_dict, open(os.path.join(args.workspace, "segment_travel_time_distribution_dict.pkl"), "wb"))

        speed_distribution = get_speed_tensor(segment_travel_time_dict, speed_profile, G, args.slot_size)
        pickle.dump(speed_distribution, open(os.path.join(args.workspace, "tt_distribution.pkl"), "wb"))
    elif args.mode == 3:
        train_data = pickle.load(open(os.path.join(args.workspace, "train_trajs.pkl"), "rb"))
        train_ods = set()
        for traj in train_data:
            train_ods.add((traj[0][0], traj[0][-1]))
        pickle.dump(train_ods, open(os.path.join(args.workspace, "train_ods.pkl"), "wb"))

    print('done')
