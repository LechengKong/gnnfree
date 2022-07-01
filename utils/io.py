import time
import numpy as np
from scipy.sparse import csr_matrix, tril
import scipy.io as io
import os.path as osp
import os
import torch
from tqdm import tqdm

def read_knowledge_graph(files, relation2id=None):
    entity2id = {}
    if relation2id is None:
        relation2id = {}

    converted_triplets = {}
    rel_list = [[] for i in range(len(relation2id))]

    ent = 0
    rel = len(relation2id)

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1
                rel_list.append([])

            data.append([entity2id[triplet[0]], relation2id[triplet[1]], entity2id[triplet[2]]])
        
        for trip in data:
            rel_list[trip[1]].append([trip[0], trip[2]])

        converted_triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    adj_list = []
    for rel_mat in rel_list:
        rel_array = np.array(rel_mat)
        if len(rel_array)==0:
            adj_list.append(csr_matrix((len(entity2id),len(entity2id))))
        else:
            adj_list.append(csr_matrix((np.ones(len(rel_mat)),(rel_array[:,0],rel_array[:,1])), shape=(len(entity2id),len(entity2id))))

    return adj_list, converted_triplets, entity2id, relation2id, id2entity, id2relation

def read_homogeneous_graph(path, is_text=False):
    if is_text:
        data = np.genfromtxt(path, delimiter=',')
        train_num_entities = np.max(data)+1
        ind_num_entities = train_num_entities
        head, tail = data[:,0], data[:,1]
    else:
        Amat = io.loadmat(path)['net']
        train_num_entities = Amat.shape[0]
        ind_num_entities = train_num_entities
        edge_mat = tril(Amat)
        head, tail = edge_mat.nonzero()
    train_num_rel = 1
    ind_num_rel = 1
    relation2id = None
    k = np.ones((train_num_entities,train_num_entities))
    k[head,tail]=0
    k[tail,head]=0
    nh,nt = k.nonzero()
    neg_perm = np.random.permutation(len(nt))
    perm = np.random.permutation(len(head))
    train_ind = int(len(perm)*0.85)
    test_ind = int(len(perm)*0.95)
    new_mat = np.zeros((len(head),3),dtype=int)
    new_mat[:,0] = head
    new_mat[:,2] = tail
    neg_mat = np.zeros((len(head),3), dtype=int)
    neg_mat[:,0] = nh[neg_perm[:len(head)]]
    neg_mat[:,2] = nt[neg_perm[:len(head)]]
    converted_triplets = {"train":new_mat[perm[:train_ind]], "train_neg":neg_mat[perm[:train_ind]], "test":new_mat[perm[train_ind:test_ind]],"test_neg":neg_mat[perm[train_ind:test_ind]], "valid":new_mat[perm[test_ind:]], "valid_neg":neg_mat[perm[test_ind:]]}
    converted_triplets_ind = converted_triplets
    rel_mat = converted_triplets['train']
    adj_list = [csr_matrix((np.ones(len(rel_mat)),(rel_mat[:,0],rel_mat[:,1])), shape=(train_num_entities,train_num_entities))]
    adj_list_ind = adj_list
    return adj_list, converted_triplets, relation2id


    # for row in data:
    #     if row[0] not in entity2id:
    #         entity2id[row[0]] = ent
    #         ent += 1
    #     if row[1] not in entity2id:
    #         entity2id[row[1]] = ent
    #         ent += 1
    #     edges.append([entity2id[row[0]], entity2id[row[1]]])

def save_load_torch_data(folder_path, data, num_output=1, data_fold=5, data_name='saved_gd_data'):
    saved_data_path = osp.join(folder_path, data_name)
    if not osp.exists(saved_data_path):
        os.mkdir(saved_data_path)
        print('gt?')
        dt = torch.utils.data.DataLoader(data, batch_size=256, num_workers=96,)
        pbar = tqdm(dt)
        fold_len = int(len(dt)/(data_fold-1))
        count = 0
        fold_count = 0
        for i, t in enumerate(pbar):
            if count == 0:
                data_col = []
                for j in range(num_output):
                    data_col.append([])
            if num_output == 1:
                data_col[0].append(t)
            else:
                for j, v in enumerate(t):
                    data_col[j].append(v)
            count+=1
            if count == fold_len:
                for i, it in enumerate(data_col):
                    cdata = torch.cat(it, dim=0).numpy()
                    np.save(osp.join(saved_data_path, str(i)+'_'+str(fold_count)), cdata)
                    for itm in it:
                        del itm
                    del cdata
                fold_count += 1
                count = 0
        if count >0 :
            for i, it in enumerate(data_col):
                cdata = torch.cat(it, dim=0).numpy()
                np.save(osp.join(saved_data_path, str(i)+'_'+str(fold_count)), cdata)
    saved_data = []
    if num_output == 1:
        for j in range(data_fold):
            ipath = osp.join(saved_data_path, str(0)+'_'+str(j)+'.npy')
            if osp.exists(ipath):
                saved_data.append(np.load(ipath))
    else:
        for i in range(num_output):
            saved_data.append([])
            for j in range(data_fold):
                ipath = osp.join(saved_data_path, str(i)+'_'+str(j)+'.npy')
                if osp.exists(ipath):
                    saved_data[i].append(np.load(ipath))
    return saved_data, data_fold