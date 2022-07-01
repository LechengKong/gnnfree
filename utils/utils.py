import time
import numpy as np

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
    keep_col, keep_row = keep_arr.nonzero()
    shuffle_ind = row_size_count>10
    ind_arr = np.zeros_like(keep_arr, dtype=int)
    ind_arr[:] = np.arange(coln)
    if shuffle_ind.sum()>0:
        select_ind = np.random.randint(1500, size=(np.sum(shuffle_ind),coln))
        select_ind = select_ind%row_size_count[shuffle_ind][:, None]
        ind_arr[shuffle_ind] = select_ind
    if rown>1:
        ind_arr[1:] += row_cum[:-1, None]
    sampled_ind = ind_arr[keep_col, keep_row]
    sample_val = col_ind[sampled_ind]
    sampled_arr = np.zeros_like(keep_arr)
    sampled_arr[keep_col,keep_row] = sample_val
    return sampled_arr, keep_arr