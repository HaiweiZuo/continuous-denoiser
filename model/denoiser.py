import math
from typing import Optional

import torch
import torch as th
import torch.nn as nn
import torchcde as tcde
from model.layer import FrEncoderLayer, FrDecoderLayer, StepMLP


class ContinueDenoiser(nn.Module):
    def __init__(self, n_feat, n_hidden, n_head: int = 4, n_layer: int = 6, d_embeds=32, d_feedforward=1024, dropout=0.1, t_max=6 * math.pi):
        super().__init__()

        self.xt_in = nn.Linear(n_feat, n_hidden)
        self.xt_out = nn.Linear(n_hidden, n_feat)
        self.st_mlp = StepMLP(n_hidden, scale=4)

        self.ts_max = nn.Parameter(th.FloatTensor([t_max]), requires_grad=False)
        self.ts_pos = nn.Parameter(th.FloatTensor([math.pow(10000.0, 2.0 * (i // 2) / n_hidden) for i in range(n_hidden)]), requires_grad=False)

        self.ts_layers = nn.ModuleList([
            FrEncoderLayer(n_head=n_head, d_model=n_hidden, d_feedforward=d_feedforward, d_embeds=d_embeds, dropout=dropout)
            for _ in range(n_layer)]
        )

    def encode_ts(self, ts):
        result = ts.unsqueeze(-1) / self.ts_pos
        result[:, 0::2] = th.sin(result[:, 0::2])
        result[:, 1::2] = th.cos(result[:, 1::2])
        return result

    def step_ts(self, step_t):
        step_emb = self.st_mlp(step_t)
        return step_emb

    def prepare_head_tail(self, xs, t_ini):
        xs_last = xs[:, -1:, :]
        xs_pad = torch.cat(tensors=[xs, xs_last], dim=1)
        ts_pad = torch.cat(tensors=[t_ini, self.ts_max], dim=-1).to(t_ini.device)
        return xs_pad, ts_pad

    def forward(self, x, t, xs_ir, ts_ir, ts_org=None):
        """
        tgt_mode: Temporal or Frequency
        """
        n_batch, n_len = xs_ir.shape[0], xs_ir.shape[1]

        #####################################
        # 1. latent position embedding
        x_noise = self.xt_in(x)

        x_emb = self.xt_in(xs_ir)
        t_emb = self.encode_ts(ts_ir).unsqueeze(0).repeat(n_batch, 1, 1)
        s_emb = self.step_ts(t).unsqueeze(1).repeat(1, n_len, 1)

        latent_x = x_emb + t_emb + s_emb  # it can be viewed as a kind of position embedding for irregular time series

        #####################################
        # 2. latent continue representation

        latent_irx, latent_irt = self.prepare_head_tail(latent_x, ts_ir)
        if ts_org is None:
            t_est_interval = self.ts_max - ts_ir[:1]
            ts_org = torch.linspace(ts_ir[0], t_est_interval[0], ts_ir.shape[0])
        ix = tcde.LinearInterpolation(latent_irx, t=latent_irt)
        if False:  # ablation here
            ix = cdeint(X=ix, backend='torchdiffeq', method='dopri5', options=dict(jump_t=X.grid_points)
        latent_icx = ix.evaluate(ts_org).float()  # continue latent x representation

        #####################################
        # 3. frequency denoise operation
        x = x_noise + latent_icx
        for layer in self.ts_layers:
            x = layer(x)

        #####################################
        # 4. output
        x = self.xt_out(x)

        return x


class ConditionContinueDenoiser(ContinueDenoiser):
    def __init__(self, n_feat, n_cond, n_hidden, n_head: int = 4, n_layer: int = 6, d_embeds=32, d_feedforward=1024, dropout=0.1, t_max=6 * math.pi):
        super().__init__(n_feat, n_hidden, n_head=n_head, n_layer=n_layer, d_embeds=d_embeds, d_feedforward=d_feedforward, dropout=dropout, t_max=t_max)
        self.ct_in = nn.Linear(n_cond, n_hidden)
        self.ts_layers = nn.ModuleList([
            FrDecoderLayer(n_head=n_head, d_model=n_hidden, d_feedforward=d_feedforward, d_embeds=d_embeds, dropout=dropout)
            for _ in range(n_layer)]
        )

    def forward(self, xs, cond, ts, t_step, org_ts=None):
        """
        tgt_mode: Temporal or Frequency
        """
        n_batch, n_len = xs.shape[0], xs.shape[1]
        mem = self.ct_in(cond)

        #####################################
        # 1. latent position embedding
        x_emb = self.xt_in(xs)
        t_emb = self.encode_ts(ts).unsqueeze(0).repeat(n_batch, 1, 1)
        s_emb = self.step_ts(t_step).unsqueeze(1).repeat(1, n_len, 1)

        latent_x = x_emb + t_emb + s_emb  # it can be viewed as a kind of position embedding for irregular time series

        #####################################
        # 2. latent continue representation

        latent_irx, latent_irt = self.prepare_head_tail(latent_x, ts)
        if org_ts is None:
            t_est_interval = self.ts_max - ts[:1]
            org_ts = torch.linspace(ts[0], t_est_interval[0], ts.shape[0])

        ix = tcde.LinearInterpolation(latent_irx, t=latent_irt)
        latent_icx = ix.evaluate(org_ts).float()  # continue latent x representation

        #####################################
        # 3. frequency denoise operation
        x = latent_icx
        for layer in self.ts_layers:
            x = layer(x, mem)

        #####################################
        # 4. output
        return x


def main():
    net_uncd = ContinueDenoiser(n_feat=20, n_hidden=256)
    net_cond = ConditionContinueDenoiser(n_feat=20, n_hidden=256, n_cond=40)
    _ = net_uncd.train()
    # xs, ts, org_ts = None, tgt_mode: str = "Temporal"
    n_batch, n_seq = 4, 150

    xs = th.randn(size=(n_batch, n_seq, 20))
    cd = th.randn(size=(n_batch, 30, 40))
    ts1 = th.sort(th.randn(size=(n_seq,))).values  # each time only one sampler
    ts2 = th.sort(th.randn(size=(n_seq + 20,))).values
    t_step = th.randint(low=0, high=1000, size=(n_batch, ))

    ys_uncond = net_uncd(xs, t_step, ts1 , org_ts=ts2)
    ys_cond = net_cond(xs, cd, ts1, t_step, org_ts=ts2)
    print("pred ys -> un-condition : {}, condition : {}".format(ys_uncond.shape, ys_cond.shape))


if __name__ == '__main__':
    main()
