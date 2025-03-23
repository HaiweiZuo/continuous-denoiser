import os
import json
import math
import time
import random
from typing import Optional

import numpy as np
import torch
import torch as th
import tqdm

from data_process.dataset import MultichannelTS, DataLoader
from model.layer import print_net_size
from model.denoiser import ContinueDenoiser
from diffusion import create_diffusion, create_named_schedule_sampler
from utils.trainer import ITrainer, Accelerator


class TrainerUnCond(ITrainer):
    def __init__(self, conf, accl: Accelerator):
        super().__init__(conf, accl)
        self.log(fn='__init__', msg="[TrainerUnCond] Start trainer-{}, run name : {}, run folder : {}".format(self.device, self.run_name, self.run_folder))
        self.train_batch_size = int(self.config.Train.batch_size)  #
        self.valid_batch_size = int(self.config.Train.valid_size)
        self.num_worker = self.config.Data.num_worker

        ###################################
        # 1. build dataset
        self.data_conf = self.config.Data
        self.data_root = self.config.Data.data_root
        result = self._build_data()
        self.train_ds = result["train_dataset"]
        self.valid_ds = result["valid_dataset"]
        self.train_dl = self.accelerator.prepare_data_loader(result["train_loader"])
        self.valid_dl = self.accelerator.prepare_data_loader(result["valid_loader"])
        self.log(fn='__init__', msg="[Train] Data size : {}, Data batch : {} | [Valid] Data size : {}, Data batch : {}. Batch size {}".format(
            len(self.train_ds), len(self.train_dl), len(self.valid_ds), len(self.valid_dl), self.train_batch_size))

        ###################################
        # 2. build model
        self.net_conf = self.config.Net
        self.diff_conf = self.config.Diffusion

        model, diffusion = self._build_model()
        param_size, model_size = print_net_size(model)
        self.model = self.accelerator.prepare_model(model)
        self.diffusion = diffusion
        self.sampler = create_named_schedule_sampler('uniform', self.diffusion)
        self.log(fn='__init__', msg="Param size : {:.2f}Mb, Model Size : {:.2f}Mb".format(param_size / (1024 ** 2), model_size / (1024 ** 2)))

        ######################################################################
        # 3. build optim
        self.lr = self.config.Train.lr
        self.weight_decay = self.config.Train.weight_decay
        self.enable_schedule = self.config.Train.enable_schedule

        optim, scheduler = self._build_optimize(use_schedule=self.enable_schedule)
        self.optim = optim
        self.scheduler = scheduler
        self.progress = self.config.Train.progress

        ######################################################################
        # resume setting
        if self.resume:
            max_ep, max_it = self._find_lastest()
            self.cur_epoch = max_ep
            self.cur_iters = max_it
        self.log(fn='__init__', msg="end construction.")

    def _build_data(self):
        trn_ds = MultichannelTS(data_source=self.data_root, data_mode='train', win_size=self.data_conf.win_size, hop_size=self.data_conf.hop_size,
                                t_max=math.pi * self.data_conf.t_max, p_size=self.data_conf.p_size, r_ratio=self.data_conf.r_ratio)
        val_ds = MultichannelTS(data_source=self.data_root, data_mode='valid', win_size=self.data_conf.win_size, hop_size=self.data_conf.hop_size,
                                t_max=math.pi * self.data_conf.t_max, p_size=self.data_conf.p_size, r_ratio=self.data_conf.r_ratio)
        trn_dl = DataLoader(trn_ds, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_worker)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
        return {"train_dataset": trn_ds, "train_loader": trn_dl, "valid_dataset": val_ds, "valid_loader": val_dl}

    def _build_model(self):
        net = ContinueDenoiser(**self.net_conf)
        diff = create_diffusion(**self.diff_conf)
        return net, diff

    def _build_optimize(self, use_schedule=False):
        optim = th.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = None
        if use_schedule:
            scheduler = None  # Noam_Scheduler(optimizer=optim, warmup_steps=self.config.Train.warmup_steps)
        optim = self.accelerator.prepare_optimizer(optim)
        if use_schedule:
            scheduler = None  # self.accelerator.prepare_scheduler(scheduler)
        return optim, scheduler

    def main(self):
        self.log(fn='main', msg="start train main loop ...")
        if self.cur_epoch < 0:
            self.cur_epoch = 0
            self.cur_iters = 0

        while (self.cur_epoch < self.max_epoch) and (self.cur_iters < self.max_iters):
            bar = tqdm.tqdm(self.train_dl)
            vlr = self.lr if (self.scheduler is None) else self.scheduler.get_last_lr()[0]

            _ = self.model.train()
            for idx, batch in enumerate(bar):
                loss, desc_loss = self.train_batch(batch, b_detail_loss=True)
                if loss is None:
                    continue

                dlos = ["{}: {:.4f}".format(ky, vl) for ky, vl in desc_loss.items()]
                desc = 'Train Epoch [{}]/[{}]-[iteration: {}]. loss : {:.6f}. lr : {:.6f}'.format(
                    self.cur_epoch + 1, self.max_epoch + 1, self.cur_iters + 1, loss, vlr
                )
                desc += "[detail loss :"
                for dls in dlos:
                    desc += dls + " "
                desc += "]"
                bar.set_description(desc=desc)
                self.cur_iters += 1

                if self.cur_iters > 0 and (self.cur_iters % self.print_every_iter == 0):
                    self.log(fn="main", msg=desc)

                if self.cur_iters > 0 and (self.cur_iters % self.save_every_iter == 0):
                    self.save_state_checkpoint(eid=self.cur_epoch, iid=self.cur_iters, safe_serialization=False)
                    self.log(fn="main", msg="save checkpoint.{}.{}".format(self.cur_epoch, self.cur_iters))

            self.cur_epoch += 1
            if self.scheduler is not None:
                self.scheduler.step()

            if (self.eval_every_iter > 1) and (self.cur_epoch > 0 and (self.cur_epoch % self.eval_every_iter != 0)):
                continue

            _ = self.model.eval()
            with th.no_grad():
                bar = tqdm.tqdm(self.valid_dl)
                avg_succ = True
                avg_loss = {}
                avg_time = []
                eval_pth = os.path.join(self.run_folder, "eval_{}".format(self.cur_epoch))
                os.makedirs(eval_pth, exist_ok=True)

                try:
                    for idx, batch in enumerate(bar):
                        val_loss, sample, val_time = self.valid_batch(batch)

                        for ky in val_loss.keys():
                            if ky not in avg_loss.keys():
                                avg_loss[ky] = []
                            avg_loss[ky].append(val_loss[ky])
                        avg_time.append(val_time)
                        desc = "val [{}/{}] time : {:.4f}".format(self.cur_epoch, self.max_epoch, val_time)
                        desc_loss = "".join(["{} :{:.4f}| ".format(ky, vl) for ky, vl in val_loss.items()])
                        desc += ". " + desc_loss
                        bar.set_description(desc=desc)

                except Exception as e:
                    avg_succ = False
                    self.log(fn="main", msg="validating in {}, error cause {}".format(self.cur_epoch, e))
                if avg_succ:
                    desc_loss = "".join(["{} :{:.4f}| ".format(ky, np.mean(vl)) for ky, vl in val_loss.items()])
                    self.log(fn="main",
                             msg="validating in {}, average time : {}. average loss : {}".format(self.cur_epoch, np.mean(avg_time), desc_loss))

        net = self.accelerator.unwrap_model(self.model)
        _ = net.eval()
        _ = net.cpu()
        torch.save(net.state_dict(), os.path.join(self.run_folder, "checkpoint.pt"))
        self.log(fn='main', msg="training succ!")
        return True

    def train_batch(self, batch, b_detail_loss=True):
        xs, ts = batch

        # pred shape:
        xs_org = xs[:, :-self.data_conf.p_size, :]
        ts_org = ts[:, :-self.data_conf.p_size]
        xs_idx = [i for i in range(xs_org.shape[1])]
        xs_keep = int(len(xs_idx) * self.data_conf.r_ratio)

        # irregular time seires
        random.shuffle(xs_idx)
        xs_idx_keep = xs_idx[:xs_keep]
        xs_idx_keep = sorted(xs_idx_keep)
        xs_ir = xs_org[:, xs_idx_keep, :]
        ts_ir = ts_org[:, xs_idx_keep]
        # return xs_ir, ts_ir,
        bat = xs_ir.shape[0]

        self.optim.zero_grad()
        t, weights = self.sampler.sample(bat, self.accelerator.device)
        model_kwargs = {"xs_ir": xs_ir, "ts_ir": ts_ir[0], "ts_org": ts[0]}  # ts, org_ts

        loss_all = self.diffusion.training_losses(model=self.model, x_start=xs, t=t, model_kwargs=model_kwargs)
        loss = torch.mean(loss_all['loss'] * weights)

        if torch.isnan(loss).any():
            self.log(fn='train_batch', msg='torch nan found, return None here in 1', level='warning')
            return None, None

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.accelerator.backward(loss)
        self.optim.step()
        desc = {
            "loss": loss.cpu().item(),
        }
        if b_detail_loss:
            for ky, vl in loss_all.items():
                if (ky == 'loss') or (vl is None):
                    continue
                desc[ky] = torch.mean(loss_all[ky] * weights).detach().cpu().item()
        return loss.cpu().item(), desc

    def valid_batch(self, batch):
        xs, ts = batch

        # pred shape:
        xs_org = xs[:, :-self.data_conf.p_size, :]
        ts_org = ts[:, :-self.data_conf.p_size]
        xs_idx = [i for i in range(xs_org.shape[1])]
        xs_keep = int(len(xs_idx) * self.data_conf.r_ratio)

        # irregular time seires
        random.shuffle(xs_idx)
        xs_idx_keep = xs_idx[:xs_keep]
        xs_idx_keep = sorted(xs_idx_keep)
        xs_ir = xs_org[:, xs_idx_keep, :]
        ts_ir = ts_org[:, xs_idx_keep]

        bat = xs_ir.shape[0]
        assert 1 == bat

        t1 = time.time()
        model_kwargs = {"xs_ir": xs_ir, "ts_ir": ts_ir[0], "ts_org": ts[0]}  # ts, org_ts
        model_output = self.diffusion.ddim_sample_loop(
            self.model, shape=xs.shape, progress=False, model_kwargs=model_kwargs
        )
        t2 = time.time()

        xs_future_gt = xs[:, -self.data_conf.p_size:, :]
        xs_future_pred = model_output[:, -self.data_conf.p_size:, :]
        xs_continue_pred = torch.cat([xs[:, :-self.data_conf.p_size, :], model_output[:, -self.data_conf.p_size:, :]], dim=1)

        val_loss = {}
        val_loss["val_pred"] = torch.mean(torch.abs(xs_future_gt - xs_future_pred)).detach().cpu().item()
        val_time = t2 - t1
        sample = {
            # 1. loss prediction related
            "gt_future": xs_future_gt,
            "pred_future": xs_future_pred,
            "pred_continue": xs_continue_pred,

            # 2. org data
            "irg_x": xs_ir,
            "irg_t": ts_ir,
            "org_x": xs,
            "org_t": ts,
        }
        return val_loss, sample, val_time
