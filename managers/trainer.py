import numpy as np
import torch
from tqdm import tqdm
from gnnfree.managers.learner import Learner

from gnnfree.utils.utils import SmartTimer

class Trainer():
    def __init__(self, evaluator, loss_fn, num_workers=4, train_sample_size=None, eval_sample_size=None):
        self.timer = SmartTimer(False)
        self.evaluator = evaluator
        self.loss_fn = loss_fn
        self.num_workers = num_workers
        self.train_sample_size = train_sample_size
        self.eval_sample_size = eval_sample_size

    def full_epoch(self, learners, device=None):
        train_metric = self.train_scheduled(learners[0], device)
        eval_metric = self.eval_scheduled(learners[1], device)
        return eval_metric

    def train_scheduled(self, learner, optimizer, device=None):
        train_metric = self.train_epoch(learner, optimizer, device=device)
        print(train_metric)
        return train_metric
    
    def eval_scheduled(self, learner, device=None):
        eval_metric = self.eval_epoch(learner, device=device)
        print(eval_metric)
        return eval_metric

    def train_epoch(self, learner:Learner, optimizer, device=None):
        dataloader = learner.create_dataloader(learner.batch_size, num_workers=self.num_workers, sample_size=self.train_sample_size)
        pbar = tqdm(dataloader)
        learner.train()
        learner.preprocess(device=device)
        self.timer.record()
        # with torch.autograd.detect_anomaly():
        t_loss = []
        for batch in pbar:
            self.timer.cal_and_update('data')
            data = learner.load(batch, device)
            self.timer.cal_and_update('move')
            res = learner.forward_func(data)
            self.timer.cal_and_update('forward')
            loss_arg = learner.data_to_loss_arg(res, data)
            loss = self.loss_fn(*loss_arg)
            self.timer.cal_and_update('loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.timer.cal_and_update('back')
            with torch.no_grad():
                eval_arg = learner.data_to_eval_arg(res, data)
                self.evaluator.collect_res(*eval_arg)
                t_loss.append(loss.item())
            self.timer.cal_and_update('score')
        metrics = self.evaluator.summarize_res()
        metrics['loss'] = np.array(t_loss).mean()
        learner.postprocess()
        self.evaluator.reset()
        return metrics
    
    def eval_epoch(self, learner, device=None):
        print('Eval ' + learner.name+ ':')
        dataloader = learner.create_dataloader(learner.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False, sample_size=self.eval_sample_size)
        pbar = tqdm(dataloader)
        with torch.no_grad():
            learner.eval()
            learner.preprocess(device=device)
            self.timer.record()
            for batch in pbar:
                self.timer.cal_and_update('data')
                data = learner.load(batch, device)
                self.timer.cal_and_update('move')
                res = learner.forward_func(data)
                # self.timer.cal_and_update('forward')
                # loss = learner.loss_func(res, data)
                eval_arg = learner.data_to_eval_arg(res, data)
                self.evaluator.collect_res(*eval_arg)
                self.timer.cal_and_update('loss')
        metrics = self.evaluator.summarize_res()
        learner.postprocess()
        self.evaluator.reset()
        return metrics

    def eval_metric(self, new_res, pre_res):
        return self.evaluator.better_results(new_res, pre_res)

    def init_metric(self):
        return self.evaluator.init_result()

class FilteredTrainer(Trainer):
    def eval_scheduled(self, learners, device=None):
        eval_metric1 = self.eval_epoch(learners[0], device=device)
        print(eval_metric1)
        eval_metric2 = self.eval_epoch(learners[1], device=device)
        print(eval_metric2)
        eval_metric = {}
        for k in eval_metric1:
            eval_metric[k] = (eval_metric1[k] + eval_metric2[k])/2
        print(eval_metric)
        return eval_metric
