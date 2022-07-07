import numpy as np
import torch
from tqdm import tqdm

from gnnfree.utils.utils import SmartTimer

class Trainer():
    def __init__(self, params):
        self.timer = SmartTimer(False)
        self.params = params

    def full_epoch(self, learners, evaluator, device=None):
        train_metric = self.train_scheduled(learners[0], evaluator, device)
        eval_metric = self.eval_scheduled(learners[1], evaluator, device)
        return eval_metric

    def train_scheduled(self, learner, evaluator, device=None):
        train_metric = self.train_epoch(learner, learner.optimizer, evaluator, device=device)
        print(train_metric)
        return train_metric
    
    def eval_scheduled(self, learner, evaluator, device=None):
        eval_metric = self.eval_epoch(learner, evaluator, device=device)
        print(eval_metric)
        return eval_metric

    def train_epoch(self, learner, optimizer, evaluator, device=None):
        dataloader = learner.create_dataloader(learner.batch_size, num_workers=self.params.num_workers)
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
            loss = learner.loss_fn(res, data)
            self.timer.cal_and_update('loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.timer.cal_and_update('back')
            with torch.no_grad():
                evaluator.collect_res(res, data)
                t_loss.append(loss.item())
            self.timer.cal_and_update('score')
        metrics = evaluator.summarize_res()
        metrics['loss'] = np.array(t_loss).mean()
        learner.postprocess()
        evaluator.reset()
        return metrics
    
    def eval_epoch(self, learner, evaluator, device=None):
        print('Eval ' + learner.name+ ':')
        dataloader = learner.create_dataloader(learner.batch_size, num_workers=self.params.num_workers, shuffle=False)
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
                evaluator.collect_res(res, data)
                self.timer.cal_and_update('loss')
        metrics = evaluator.summarize_res()
        learner.postprocess()
        evaluator.reset()
        return metrics

    def eval_metric(self, metrics, metric_name, presult):
        return metrics[metric_name]<=presult, metrics[metric_name]

    def init_metric(self):
        return 1000000

class MaxTrainer(Trainer):
    def __init__(self, params):
        super().__init__(params)
    
    def init_metric(self):
        return 0

    def eval_metric(self, metrics, metric_name, presult):
        return metrics[metric_name]>=presult, metrics[metric_name]

class FilteredTrainer(Trainer):
    def __init__(self, params):
        super().__init__( params)

    def eval_scheduled(self, learners, evaluator, device=None):
        eval_metric1 = self.eval_epoch(learners[0], evaluator, device=device)
        print(eval_metric1)
        eval_metric2 = self.eval_epoch(learners[1], evaluator, device=device)
        print(eval_metric2)
        eval_metric = {}
        for k in eval_metric1:
            eval_metric[k] = (eval_metric1[k] + eval_metric2[k])/2
        print(eval_metric)
        return eval_metric

class FilteredMaxTrainer(FilteredTrainer, MaxTrainer):
    def __init__(self, params):
        super().__init__(params)