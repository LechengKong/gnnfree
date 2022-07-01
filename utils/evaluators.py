import numpy as np
import torch
from sklearn import metrics as met

class Evaluator():
    def __init__(self, name) -> None:
        self.name = name

    def collect_res(self, res, batch):
        pass

    def summarize_res(self):
        pass

    def reset(self):
        pass

class HeadPosRankingEvaluator(Evaluator):
    def __init__(self, name, neg_sample_size) -> None:
        super().__init__(name)
        self.rankings = []
        self.neg_sample_size = neg_sample_size

    def collect_res(self, res, batch):
        score_mat = res.view(-1, self.neg_sample_size+1).cpu().numpy()
        score_sort = np.argsort(score_mat, axis=1)
        rankings = len(score_mat[0])-np.where(score_sort==0)[1]
        self.rankings+=rankings.tolist()

    def summarize_res(self):
        metrics = {}
        rankings = np.array(self.rankings)
        metrics['h10'] = np.mean(rankings<=10)
        metrics['h3'] = np.mean(rankings<=3)
        metrics['h1'] = np.mean(rankings==1)
        metrics['mrr'] = np.mean(1/rankings)
        return metrics

    def reset(self):
        self.rankings = []

class BinaryHNEvaluator(Evaluator):
    def __init__(self, name, hn=None) -> None:
        super().__init__(name)
        self.targets = []
        self.scores = []
        self.hn = hn

    def collect_res(self, res, batch):
        labels = batch.labels
        self.scores.append(res.cpu().numpy().flatten())
        self.targets.append(labels.cpu().numpy())
        
    def summarize_res(self):
        metrics = {}
        all_scores = np.concatenate(self.scores)
        all_targets = np.concatenate(self.targets)
        metrics['auc'] = met.roc_auc_score(all_targets, all_scores)
        metrics['apr'] = met.average_precision_score(all_targets, all_scores)
        if self.hn is not None:
            sort_ind = np.argsort(all_scores)
            ranked_targets = all_targets[sort_ind[::-1]]
            ranked_targets = np.logical_not(ranked_targets)
            sumed_arr = np.cumsum(ranked_targets)
            break_ind = np.where(sumed_arr==self.hn)[0][0]
            hncount = break_ind-(self.hn-1)
            metrics['h'+str(self.hn)] = hncount/np.sum(all_targets)
        return metrics

    def reset(self):
        self.targets = []
        self.scores = []

class BinaryAccEvaluator(Evaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.targets = []
        self.scores = []

    def collect_res(self, res, batch):
        labels = batch.labels
        self.scores.append(res.cpu().numpy())
        self.targets.append(labels.cpu().numpy())
        
    def summarize_res(self):
        metrics = {}
        all_scores = np.concatenate(self.scores)
        all_targets = np.concatenate(self.targets)
        pos = np.argmax(all_scores, axis=-1)
        metrics['acc'] = met.accuracy_score(all_targets, pos)
        return metrics

    def reset(self):
        self.targets = []
        self.scores = []

class InfoNEEvaluator(Evaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.loss = []

    def collect_res(self, res, batch):
        n = len(res)
        e_neg_mat = res.view(-1)[1:].view(n-1,n+1)[:,:-1].reshape(n,n-1)
        e_pos = torch.diagonal(res)
        loss = -torch.mean(torch.log(torch.exp(e_pos)/torch.exp(e_neg_mat).sum(dim=-1)))
        self.loss.append(loss.item())
        
    def summarize_res(self):
        metrics = {}
        metrics['mi_loss'] = np.array(self.loss).mean()
        return metrics

    def reset(self):
        self.loss = []

class LossEvaluator(Evaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.loss = []
    
    def collect_res(self, res, batch):
        self.loss.append(res.item())

    def summarize_res(self):
        metrics = {}
        total_loss = np.array(self.loss).mean()
        metrics['loss'] = total_loss
        return metrics

    def reset(self):
        self.loss = []

class CollectionEvaluator(Evaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.out_val = []
    
    def collect_res(self, res, batch):
        self.out_val.append(res)

    def summarize_res(self):
        metrics = {}
        res = torch.cat(self.out_val)
        metrics['res_col'] = res
        return metrics

    def reset(self):
        self.out_val = []

class MSEEvaluator(Evaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.pred = []
        self.labels = []
    
    def collect_res(self, res, batch):
        self.pred.append(res.cpu().numpy())
        self.labels.append(batch.labels.cpu().numpy())
    
    def summarize_res(self):
        metrics = {}
        pred = np.concatenate(self.pred)
        labels = np.concatenate(self.labels)
        metrics['mse'] = met.mean_squared_error(labels, pred)
        return metrics
    
    def reset(self):
        self.pred = []
        self.labels = []
