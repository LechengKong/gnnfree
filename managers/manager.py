import torch

class Manager():
    def __init__(self, save_path='model'):
        self.save_path = save_path
        self.starting_epoch = 0
        self.current_epoch = self.starting_epoch
            
    def train(self, train_learner, val_learner, trainer, optimizer, metric_name= 'mrr', save_epoch=1, device=None, eval_every=1, num_epochs=10):
        print('Train: Optimize w.r.t', metric_name)
        best_res = trainer.init_metric()
        for epoch in range(self.starting_epoch+1, self.starting_epoch+num_epochs+1):
            print('Epoch', epoch)
            self.current_epoch = epoch
            # metrics = trainer.full_epoch(self.learner, device=device)
            train_metric = trainer.train_scheduled(train_learner, optimizer, device)
            if epoch%eval_every==0 or epoch==self.starting_epoch+num_epochs:
                metrics = trainer.eval_scheduled(val_learner, device)
                update, res = trainer.eval_metric(metrics[metric_name], best_res)
                if update:
                    print('Found better model')
                    best_res = res
                    self.save_model(train_learner)
            if epoch%save_epoch==0:
                self.save_checkpoint(train_learner)
            self.optimizer_update(optimizer)

    def eval(self, learner, trainer, device=None):
        metrics = trainer.eval_scheduled(learner, device=device)
        return metrics

    def load_model(self, learner):
        learner.load_model(self.save_path+'.pth')

    def optimizer_update(self, optimizer):
        pass

    def save_model(self, learner):
        save_dict = {'epoch':self.current_epoch}
        learner.save_model(self.save_path+'.pth', save_dict)

    def save_checkpoint(self, learner):
        save_dict = {'epoch':self.current_epoch}
        learner.save_model(self.save_path+'check.pth', save_dict)

