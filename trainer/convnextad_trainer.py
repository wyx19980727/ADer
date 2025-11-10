import os
import copy
import glob
import shutil
import datetime
import time

import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp
from functools import partial

def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x

def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss



@TRAINER.register_module
class ConvnextRDTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(ConvnextRDTrainer, self).__init__(cfg)
        
    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        if len(self.imgs.shape)==5:
            self.imgs = self.imgs.flatten(0, 1)
            self.imgs_mask = self.imgs_mask.flatten(0, 1)
        self.cls_name = inputs['cls_name']
        self.cls_name = np.array(self.cls_name)
        self.cls_name = np.transpose(self.cls_name, (1, 0)) 
        
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']

        self.img_path = np.array(self.img_path)
        self.img_path = np.transpose(self.img_path, (1, 0))
  
        self.sample_anomaly = inputs['sample_anomaly']
        self.bs = self.imgs.shape[0]
 
    def forward(self):
        self.en, self.de = self.net(self.imgs, self.img_path)

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            # p_final = 0.9
            # p = min(p_final * self.iter / 1000, p_final)
            # loss_cos = global_cosine_hm_percent(self.en, self.de, p=p, factor=0.1)
            loss_cos = self.loss_terms['cos'](self.en, self.de)
        self.backward_term(loss_cos, self.optim)
        update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 1, self.master)

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)
        # imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
        test_img_length = self.cfg.data.test_length*5
        img_size = self.cfg.size
        results=dict(
            imgs_masks=np.empty((test_img_length, 1, img_size, img_size), dtype='int64'),
            anomaly_maps=np.empty((test_img_length, 1, img_size, img_size), dtype='float32'),
            cls_names=np.empty((test_img_length,), dtype='object'),
            anomalys=np.empty((test_img_length,), dtype='int64'),
            sample_anomaly=np.empty((self.cfg.data.test_length,), dtype='int64'),
            # img_path=np.empty((test_img_length,), dtype='object'),
            # sample_anomaly=np.empty((test_img_length,), dtype='int64')
            # img_path = self.img_path,
            # sample_anomaly = self.sample_anomaly
        )
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            # if batch_idx == 10:
            # 	break
            t1 = get_timepc()
            # batch_idx += 1
            #import ipdb; ipdb.set_trace()
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()
            loss_cos = self.loss_terms['cos'](self.en, self.de)
            # p_final = 0.9
            # p = min(p_final * self.iter / 1000, p_final)
            # loss_cos = global_cosine_hm_percent(self.en, self.de, p=p, factor=0.1)
            update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 1, self.master)
            # get anomaly maps
            anomaly_map, _ = self.evaluator.cal_anomaly_map(self.en, self.de, [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False, amap_mode='add', gaussian_sigma=4)
            anomaly_map = np.expand_dims(anomaly_map, axis=1)
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            
            if len(test_data['img'].shape) == 5:
                #import ipdb; ipdb.set_trace()
                # 1. not last batch
                if len(test_data['img']) == self.cfg.batch_test_per:
                    results['imgs_masks'][5*batch_idx*self.cfg.batch_test_per:5*(batch_idx+1)*self.cfg.batch_test_per] = self.imgs_mask.cpu().numpy().astype(int)
                    results['anomaly_maps'][5*batch_idx*self.cfg.batch_test_per:5*(batch_idx+1)*self.cfg.batch_test_per] = anomaly_map
                    results['cls_names'][5*batch_idx*self.cfg.batch_test_per:5*(batch_idx+1)*self.cfg.batch_test_per] = self.cls_name.flatten()
                    results['anomalys'][5*batch_idx*self.cfg.batch_test_per:5*(batch_idx+1)*self.cfg.batch_test_per] = self.anomaly.flatten().cpu().numpy().astype(int)
                    results['sample_anomaly'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = self.sample_anomaly
                # 2. last batch
                else:
                    #import ipdb; ipdb.set_trace()
                    results['imgs_masks'][5*batch_idx*self.cfg.batch_test_per:] = self.imgs_mask.cpu().numpy().astype(int)
                    results['anomaly_maps'][5*batch_idx*self.cfg.batch_test_per:] = anomaly_map
                    results['cls_names'][5*batch_idx*self.cfg.batch_test_per:] = self.cls_name.flatten()
                    results['anomalys'][5*batch_idx*self.cfg.batch_test_per:] = self.anomaly.flatten().cpu().numpy().astype(int)
                    results['sample_anomaly'][batch_idx*self.cfg.batch_test_per:] = self.sample_anomaly
            
            else:
                # 1. not last batch
                if len(test_data['img']) == self.cfg.batch_test_per:
                    results['imgs_masks'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = self.imgs_mask.cpu().numpy().astype(int)
                    results['anomaly_maps'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = anomaly_map
                    results['cls_names'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = self.cls_name.flatten()
                    results['anomalys'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = self.anomaly.flatten().cpu().numpy().astype(int)
                    results['sample_anomaly'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = self.sample_anomaly
                # 2. last batch
                else:
                    # import ipdb; ipdb.set_trace()
                    results['imgs_masks'][batch_idx*self.cfg.batch_test_per:] = self.imgs_mask.cpu().numpy().astype(int)
                    results['anomaly_maps'][batch_idx*self.cfg.batch_test_per:] = anomaly_map
                    results['cls_names'][batch_idx*self.cfg.batch_test_per:] = self.cls_name.flatten()
                    results['anomalys'][batch_idx*self.cfg.batch_test_per:] = self.anomaly.flatten().cpu().numpy().astype(int)
                    results['sample_anomaly'][batch_idx*self.cfg.batch_test_per:] = self.sample_anomaly
            
            batch_idx += 1
   
            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
            # ---------- log ----------
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
                    log_msg(self.logger, msg)
        # # merge results
        # if self.cfg.dist:
        # 	results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
        # 	torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
        # 	if self.master:
        # 		results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
        # 		valid_results = False
        # 		while not valid_results:
        # 			results_files = glob.glob(f'{self.tmp_dir}/*.pth')
        # 			if len(results_files) != self.cfg.world_size:
        # 				time.sleep(1)
        # 			else:
        # 				idx_result = 0
        # 				while idx_result < self.cfg.world_size:
        # 					results_file = results_files[idx_result]
        # 					try:
        # 						result = torch.load(results_file)
        # 						for k, v in result.items():
        # 							results[k].extend(v)
        # 						idx_result += 1
        # 					except:
        # 						time.sleep(1)
        # 				valid_results = True
        # else:
        # 	results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
        if self.master:
            # results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None
                # msg += f'\n{cls_name:<10}'
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
            msg_csv = copy.deepcopy(msg)
            for key in list(msg_csv.keys()):
                if "(Max)" in key:
                    msg_csv.pop(key)
            msg_csv = tabulate.tabulate(msg_csv, headers='keys', tablefmt=tabulate.simple_separated_format(","), floatfmt='.3f', numalign="center", stralign="center", )
            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center", )
            log_msg(self.logger, f'\n{msg}')
            #import ipdb; ipdb.set_trace()
            # df = pd.read_html(msg, index_col=0)[0]
            # df.to_csv(f'{self.cfg.logdir}/result.csv')
            with open(f'{self.cfg.logdir}/result.csv', 'w', newline='') as f:
                f.write(msg_csv)
            
