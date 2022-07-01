import torch

from torch.utils.data import DataLoader, RandomSampler

class Learner():
    def __init__(self, name, data, model, loss, optimizer, batch_size):
        self.name = name
        self.optimizer_type = optimizer
        self.collate_func = data.get_collate_fn()
        self.data = data
        self.model = model
        self.loss = loss
        self.batch_size = batch_size

        self.current_dataloader = None
        self.optimizer = None

    def create_dataloader(self, batch_size, num_workers=4, sample_size=None, shuffle=True):
        if sample_size is None:
            self.current_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_func, shuffle=shuffle, pin_memory=True)
        else:
            self.current_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_func, sampler=RandomSampler(self.data, num_samples=sample_size, replacement=True))
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
        pass

    def forward_func(self, batch):
        pass

    def loss_fn(self, res, batch):
        pass

    def setup_optimizer(self, optimizer_groups):
        pass

    def save_model(self, path, epoch):
        pass

    def load_model(self, path, device=None):
        pass

    def num_model_params(self):
        pass

class SingleModelLearner(Learner):
    def setup_optimizer(self, optimizer_groups):
        parameters = [p for p in self.model.parameters()]
        optimizer_groups[0]['params'] = parameters
        self.optimizer = self.optimizer_type(optimizer_groups)

    def save_model(self, path, save_dict):
        save_dict['state_dict'] = self.model.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path, device=None):
        state_d = torch.load(path, device)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_d['optimizer'])
        if self.model is not None:
            self.model.load_state_dict(state_d['state_dict'])
        return state_d

    def num_model_params(self):
        model_params = list(self.model.parameters())
        return sum(map(lambda x: x.numel(), model_params))

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

