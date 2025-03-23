import os
import csv
import math
import random
import datetime
from typing import Optional, Dict

import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader


def load_from_raw_txt(txt_path, split: float = 0.8):
    with open(txt_path, 'r') as fin:
        lines = fin.readlines()
        fin.close()

    v_lines = []
    v_size = -1
    for ln in lines:
        if ln.endswith('\n'):
            ln = ln[:-1]
        v_line = [float(vs) for vs in ln.split(sep=',')]
        if v_size < 0:
            v_size = len(v_line)
        else:
            assert v_size == len(v_line), "size dis-match!!!"
        v_lines.append(v_line)

    v_lines = np.asarray(v_lines).astype(np.float32)
    n_frs, n_dim = v_lines.shape[0], v_lines.shape[1]
    n_trn_sz = int(split * n_frs)

    min_v = np.min(v_lines, axis=0)
    max_v = np.max(v_lines, axis=0)
    mean_v = np.mean(v_lines, axis=0)
    std_v = np.std(v_lines, axis=0)
    assert n_dim == min_v.shape[0] == max_v.shape[0] == mean_v.shape[0] == std_v.shape[0]

    data = v_lines
    state = {'min': min_v, 'max': max_v, "mean": mean_v, 'std': std_v}
    print("data shape :", data.shape)

    return {"train": data[:n_trn_sz, :], "valid": data[n_trn_sz:, :]}, state


def load_from_raw_csv(csv_path, split: float = 0.8):
    t_list, v_list = [], []
    with open(csv_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        headers = next(csv_reader)
        assert 'date' == headers[0]

        for row in csv_reader:
            v_dt = datetime.datetime.strptime(row[0], '%Y/%m/%d %H:%M')
            v_vl = [float(v) for v in row[1:]]
            t_list.append(v_dt)
            v_list.append(v_vl)
        csvfile.close()
    v_lines = np.asarray(v_list).astype(np.float32)
    v_delta = [(t_list[i + 1] - t_list[i]).days for i in range(len(t_list) - 1)]
    assert all(x == v_delta[0] for x in v_delta), "time interval not even"

    n_frs, n_dim = v_lines.shape[0], v_lines.shape[1]
    n_trn_sz = int(split * n_frs)

    min_v = np.min(v_lines, axis=0)
    max_v = np.max(v_lines, axis=0)
    mean_v = np.mean(v_lines, axis=0)
    std_v = np.std(v_lines, axis=0)
    assert n_dim == min_v.shape[0] == max_v.shape[0] == mean_v.shape[0] == std_v.shape[0]

    data = v_lines
    state = {'min': min_v, 'max': max_v, "mean": mean_v, 'std': std_v}
    print("data shape :", data.shape)

    return {"train": data[:n_trn_sz, :], "valid": data[n_trn_sz:, :]}, state


class MultichannelTS(Dataset):
    def __init__(self, data_source, data_mode, win_size=300, hop_size=270, t_max: float = math.pi * 6, p_size=30, r_ratio=0.8):
        assert hop_size < win_size
        self.win_size = win_size
        self.hop_size = hop_size
        self.t_max = t_max
        self.p_size = p_size  # prediction size
        self.r_ratio = r_ratio  # irregular time seires
        self.b_norm = True

        self.data: Optional[np.ndarray] = None
        self.state: Optional[Dict[str, np.ndarray]] = None
        self.is_train = (data_mode == 'train')

        self.n_frame = 0
        self.n_dim = 0
        assert self.load(data_source)
        self.n_block = int(self.n_frame // self.hop_size) + (0 if 0 == self.n_frame % self.hop_size else 1)

    def load(self, data_source: str) -> bool:
        if not os.path.exists(data_source):
            print("data path-{}-empty, load fail".format(data_source))
            return False

        try:
            if isinstance(data_source, str) and data_source.lower().endswith('.txt'):
                data, state = load_from_raw_txt(data_source)
            elif isinstance(data_source, str) and data_source.lower().endswith('.csv'):
                data, state = load_from_raw_csv(data_source)
            else:
                data, state = data_source

        except Exception as e:
            print("load data from {} fail, cause {}".format(e))
            return False
        self.data = data['train'] if self.is_train else data['valid']
        self.state = state
        if self.b_norm:
            self.data = (self.data - self.state['min']) / (self.state['max'] - self.state['min'] + 1e-6)

        self.n_frame, self.n_dim = self.data.shape
        if (self.n_frame == 0) or (self.n_dim == 0):
            return False
        return True

    def __len__(self):
        return self.n_block

    def __getitem__(self, idx):
        rngA = self.hop_size * idx
        rngB = rngA + self.win_size
        if rngB > self.n_frame:
            rngB = self.n_frame
            rngA = rngB - self.win_size

        xs = th.from_numpy(self.data[rngA:rngB, :]).to(th.float32)
        ts = th.linspace(start=0, end=self.t_max, steps=xs.shape[0])

        # # pred shape:
        # xs_org = xs[:-self.p_size]
        # ts_org = ts[:-self.p_size]
        # xs_idx = [i for i in range(xs_org.shape[0])]
        # xs_keep = int(len(xs_idx) * self.r_ratio)
        #
        # # irregular time seires
        # random.shuffle(xs_idx)
        # ix_keep = sorted(xs_idx[:xs_keep])
        # xs_ir = xs_org[ix_keep, :]
        # ts_ir = ts_org[ix_keep]
        # return xs_ir, ts_ir,
        return xs, ts


def statistic():
    txt_path = [
        # r'D:\workspace\project_frqmlp\data\electricity\electricity.txt',
        # r'D:\workspace\project_frqmlp\data\exchange_rate\exchange_rate.txt',
        # r'D:\workspace\project_frqmlp\data\solar-energy\solar_AL.txt',
        # r'D:\workspace\project_frqmlp\data\traffic\traffic.txt',
        r'D:\workspace\project_frqmlp\dataset\covid.csv',
        r'D:\workspace\project_frqmlp\dataset\ECG_data.csv',
    ]
    for tp in txt_path:
        print("tp :", tp)
        ds = MultichannelTS(tp, 'train')


def main():
    txt_path = r'D:\workspace\project_frqmlp\data\electricity\electricity.txt'

    ds = MultichannelTS(txt_path, 'train')
    xs_ir, ts_ir, xs, ts = ds[0]
    print("xs_ir : {}, ts_ir : {}, xs : {}, ts : {}".format(
        xs_ir.shape, ts_ir.shape, xs.shape, ts.shape
    ))


if __name__ == '__main__':
    # main()
    statistic()
