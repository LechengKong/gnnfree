import torch

class Manager():
    def __init__(self, save_path='model'):
        self.save_path = save_path
        self.starting_epoch = 0
        self.current_epoch = self.starting_epoch
            
    def train(self, train_learner, val_learner, trainer, optimizer, metric_name= 'mrr', save_epoch=1, device=None, eval_every=1, num_epochs=10, scheduler=None):
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
                    self.save_model(train_learner, optimizer)
            if epoch%save_epoch==0:
                self.save_checkpoint(train_learner, optimizer)
            self.optimizer_update(optimizer, scheduler)

    def eval(self, learner, trainer, device=None):
        metrics = trainer.eval_scheduled(learner, device=device)
        return metrics

    def load_model(self, learner, optimizer=None, device=None):
        state_d = torch.load(self.save_path+'.pth', device)
        if optimizer is not None:
            optimizer.load_state_dict(state_d['optimizer'])
        learner.model.load_state_dict(state_d['state_dict'])
        self.current_epoch = state_d['epoch']+1

    def optimizer_update(self, optimizer, scheduler):
        if scheduler is not None:
            scheduler.step()

    def save_model(self, learner, optimizer):
        save_dict = {'epoch':self.current_epoch}
        save_dict['state_dict'] = learner.model.state_dict()
        save_dict['optimizer'] = optimizer.state_dict()
        torch.save(save_dict, self.save_path+'.pth')

    def save_checkpoint(self, learner, optimizer):
        save_dict = {'epoch':self.current_epoch}
        save_dict['state_dict'] = learner.model.state_dict()
        save_dict['optimizer'] = optimizer.state_dict()
        torch.save(save_dict, self.save_path+'_checkpoint.pth')

