import time
import numpy as np
from sklearn.model_selection import StratifiedKFold

class SmartTimer():
    def __init__(self, verb = True) -> None:
        self.last = time.time()
        self.verb = verb

    def record(self):
        self.last = time.time()
    
    def cal_and_update(self, name):
        now = time.time()
        if self.verb:
            print(name,now-self.last)
        self.record()


def get_rank(b_score):
    order = np.argsort(b_score)
    return len(order)-np.where(order==0)[0][0]

def save_params(filename, params):
    with open(filename, 'a') as f:
        f.write("\n\n")
        d = vars(params)
        string = "python train_graph.py "
        for k in d:
            string+="--"+k+" "+str(d[k])+" "
        f.write(string)


def sample_variable_length_data(rown, coln, row_ind, col_ind):
    # print(row_ind)
    keep_arr = np.zeros((rown, coln+2))
    row_size_count = np.bincount(row_ind, minlength=rown)
    row_cum = np.cumsum(row_size_count)
    fill_size_count = np.clip(row_size_count, 0, coln)
    keep_arr[:,0]=1
    keep_arr[np.arange(len(row_size_count)), fill_size_count+1] = -1
    keep_arr = np.cumsum(keep_arr,axis=-1)[:,1:-1]
    keep_row, keep_col = keep_arr.nonzero()
    shuffle_ind = row_size_count>coln
    ind_arr = np.zeros_like(keep_arr, dtype=int)
    ind_arr[:] = np.arange(coln)
    if shuffle_ind.sum()>0:
        select_ind = np.random.choice(1500,size=(np.sum(shuffle_ind),coln), replace=False)
        select_ind = select_ind%row_size_count[shuffle_ind][:, None]
        ind_arr[shuffle_ind] = select_ind
    if rown>1:
        ind_arr[1:] += row_cum[:-1, None]
    sampled_ind = ind_arr[keep_row, keep_col]
    sample_val = col_ind[sampled_ind]
    sampled_arr = np.zeros_like(keep_arr)
    sampled_arr[keep_row,keep_col] = sample_val
    return sampled_arr, keep_arr


def cv_with_valid(data, labels, data_cons_func, model, fold, epochs, train_learner, test_learner, val_learner, trainer, manager, evaluator, eval_metric, optimizer_parameter, device):
    val_res_col = []
    test_res_col = []
    folds = k_fold_ind(labels, fold)
    for i in range(fold):
        test_arr = np.zeros(len(labels), dtype=bool)
        test_arr[folds[i]]=1
        val_arr = np.zeros(len(labels), dtype=bool)
        val_arr[folds[int((i+1)%fold)]]=1
        train_arr = np.logical_not(np.logical_or(test_arr, val_arr))
        train_ind = train_arr.nonzero()[0]
        test_ind = test_arr.nonzero()[0]
        val_ind = val_arr.nonzero()[0]
        train = data_cons_func(data, labels, train_ind)
        test = data_cons_func(data, labels, test_ind)
        val = data_cons_func(data, labels, val_ind)

        print(f'Train: {len(train)} graphs; Test: {len(test)} graphs; Test: {len(val)} graphs')

        model.reset_parameters()
        cur_model = model.to(device)
        
        train_learner.update_data(train)
        train_learner.model = cur_model
        train_learner.setup_optimizer(optimizer_parameter)
        test_learner.update_data(test)
        test_learner.model = cur_model
        val_learner.update_data(val)
        val_learner.model = cur_model
        
        manager.train([train_learner, val_learner], trainer, evaluator, eval_metric, device=device, num_epochs=epochs)
        manager.load_model(train_learner)

        val_res = manager.eval(val_learner, trainer, evaluator, device=device)
        test_res = manager.eval(test_learner, trainer, evaluator, device=device)

        val_res_col.append(val_res)
        test_res_col.append(test_res)

    val_res_dict = dict_res_summary(val_res_col)
    test_res_dict = dict_res_summary(test_res_col)

    return val_res_dict, test_res_dict

def k_fold_ind(labels, fold):
    ksfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=10)
    folds = []
    for _, t_index in ksfold.split(np.zeros_like(np.array(labels)), np.array(labels)):
        folds.append(t_index)
    return folds

def dict_res_summary(res_col):
    res_dict = {}
    for res in res_col:
        for k in res:
            if k not in res_dict:
                res_dict[k] = []
            res_dict[k].append(res[k])
    return res_dict