import torch

from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam

from gnnfree.utils.datasets import DatasetWithCollate

class Learner():
    def __init__(self, name, data, model, batch_size):
        self.name = name
        if isinstance(data, DatasetWithCollate):
            self.collate_func = data.get_collate_fn()
        else:
            self.collate_func = None
        self.data = data
        self.model = model
        self.batch_size = batch_size

        self.current_dataloader = None

    def create_dataloader(self, batch_size, num_workers=4, sample_size=None, shuffle=True, drop_last=True):
        if sample_size is None:
            self.current_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_func, shuffle=shuffle, pin_memory=True, drop_last=drop_last)
        else:
            self.current_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_func, sampler=RandomSampler(self.data, num_samples=sample_size, replacement=True), drop_last=drop_last)
        return self.current_dataloader

    def preprocess(self, device=None):
        pass

    def postprocess(self):
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def load(self, batch, device):
        batch.to(device)
        batch.to_name()
        return batch

    def forward_func(self, batch):
        pass

    def data_to_loss_arg(self, res, batch):
        pass

    def data_to_eval_arg(self, res, batch):
        return self.data_to_loss_arg(res, batch)

    def num_model_params(self):
        pass

    def update_data(self, data):
        self.data = data
        self.collate_func = data.get_collate_fn()

class SingleModelLearner(Learner):
    def num_model_params(self):
        model_params = list(self.model.parameters())
        return sum(map(lambda x: x.numel(), model_params))

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

