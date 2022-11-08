from abc import ABCMeta, abstractmethod

from torch.utils.data import DataLoader, RandomSampler


class BaseLearner(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def create_dataloader(self):
        pass

    @abstractmethod
    def preprocess(self, device=None):
        pass

    @abstractmethod
    def postprocess(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load(self, batch, device):
        pass

    @abstractmethod
    def forward_func(self, batch):
        pass

    @abstractmethod
    def data_to_loss_arg(self, res, batch):
        pass

    @abstractmethod
    def data_to_eval_arg(self, res, batch):
        pass

    @abstractmethod
    def num_model_params(self):
        pass


class Learner(BaseLearner):
    def __init__(self, name, data, model, batch_size):
        super().__init__(name)
        self.collate_func = data.get_collate_fn()
        self.data = data
        self.model = model
        self.batch_size = batch_size

    def create_dataloader(
        self,
        batch_size,
        num_workers=4,
        sample_size=None,
        shuffle=True,
        drop_last=True,
    ):
        if sample_size is None:
            return DataLoader(
                self.data,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=self.collate_func,
                shuffle=shuffle,
                pin_memory=True,
                drop_last=drop_last,
            )
        else:
            return DataLoader(
                self.data,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=self.collate_func,
                sampler=RandomSampler(
                    self.data, num_samples=sample_size, replacement=True
                ),
                drop_last=drop_last,
            )

    def preprocess(self, device=None):
        pass

    def postprocess(self):
        pass

    def load(self, batch, device):
        batch.to(device)
        batch.to_name()
        return batch

    def data_to_eval_arg(self, res, batch):
        return self.data_to_loss_arg(res, batch)


class SingleModelLearner(Learner):
    def num_model_params(self):
        model_params = list(self.model.parameters())
        return sum(map(lambda x: x.numel(), model_params))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
