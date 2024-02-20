import numpy as np
import torch
from torchmeta.utils.data import Dataset

from maml.NEW_dataset import MoNuSeg as Mo
from maml.NEW_dataset import TNBC as TN
from maml.NEW_dataset import CoNSep as Co
from maml.NEW_dataset import CPM as cpm

import time


class Nuclei_dataset(Dataset):

    def __init__(self, meta_learn_iteration, unseen_dataset=None, meta_train_sample=2, meta_test_sample=2):
        dataset_dict = {'MoNuSeg': Mo.MoNuSeg, 'TNBC': TN.TNBC, 'CPM': cpm.CPM, 'CoNSep': Co.CoNSep}
        # print('dataset_dict:', dataset_dict)
        self.seen_dataset_dict = {dataset_name: dataset_dict[dataset_name](unseen_dataset) for dataset_name in dataset_dict if
                                  dataset_name != unseen_dataset}
        self.meta_train_sample = meta_train_sample
        self.meta_test_sample = meta_test_sample
        self.total_sample = self.meta_train_sample + self.meta_test_sample
        self.iteration = meta_learn_iteration

        super(Nuclei_dataset, self).__init__(index=None)

        # print('seen dataset name: ', self.seen_dataset_dict.keys())
        # print()

    def __len__(self):
        return self.iteration

    def sample_index(self, dataset):

        # print('idx dataset:', dataset) #MoNuSeg, TNBC, CPM, CoNSep
        index = self.seen_dataset_dict[dataset]
        sampels = range(len(index))
        index = np.random.choice(sampels, self.total_sample, replace=False)
        # print('index:', index)
        meta_train = index[:self.meta_train_sample]
        meta_test = index[self.meta_train_sample:]
        # print('meta_train:', meta_train)
        # print('meta_test:', meta_test)

        return meta_train, meta_test

    def __getitem__(self, inx):
        # self.seen_dataset_dict.keys()
        seen_dataset_dict_keys = list(self.seen_dataset_dict.keys())  # Mo, TNBC, CPM, CoNSep
        array_seen_dataset_dict_keys = np.array(seen_dataset_dict_keys)  # Mo, TNBC, CPM, CoNSep

        shuffled_seen_dataset_dict = np.random.permutation(
            array_seen_dataset_dict_keys)  # shuffle the dataset: MoNuSeg, TNBC, CPM, CoNSep
        meta_train_samples_inputs = []
        meta_train_samples_masks = []
        meta_train_hs_masks = []
        meta_test_samples_inputs = []
        meta_test_samples_masks = []
        meta_test_hs_masks = []
        
        # domain_name = []

        for e in shuffled_seen_dataset_dict:
#             print('dataname for e:', e)
            SINCE = time.time()
            train_domain_inputs = []
            train_domain_masks = []
            train_hs_masks = []

            test_domain_inputs = []
            test_domain_masks = []
            test_hs_masks = []

            meta_train_index, meta_test_index = self.sample_index(e)  # sample index from each dataset: MoNuSeg, TNBC, CPM, CoNSep
            # meta_train_index: 8, meta_test_index: 2
            # meta_train_index: 8 different iamges and 8 different masks
            # meta_test_index: 2 different iamges and 2 different masks
            index_since_time = time.time()
            # print('index_since:', index_since_time - SINCE)

            # len(meta_train_index) = 4

            #
            # print('----------------------------------')
            # print('meta_train_index:', meta_train_index)
            # print('meta_test_index:', meta_test_index)
            # print('----------------------------------')
            for train_idx in meta_train_index:
                # print('train_idx:', train_idx)
                temp_meta_train_inputs, temp_meta_train_masks, _, _, temp_sp, temp_hs1, temp_hs2, temp_hs3,label_his_b, label_his_f = \
                    self.seen_dataset_dict[e][train_idx]
                train_domain_inputs.append(temp_meta_train_inputs.unsqueeze(0))
                train_domain_masks.append(temp_meta_train_masks.unsqueeze(0))
                train_hs_masks.append(torch.cat([temp_sp, temp_hs1, temp_hs2, temp_hs3], dim=0).unsqueeze(0))

            train_idx_time = time.time()
            # print('train time:', e, train_idx_time - index_since_time)

            for test_idx in meta_test_index:
                # print('test_idx:', test_idx)
                temp_meta_test_inputs, temp_meta_test_masks, _, _, temp_test_sp, temp_test_hs1, temp_test_hs2, temp_test_hs3,label_his_b, label_his_f = self.seen_dataset_dict[e][test_idx]
                test_domain_inputs.append(temp_meta_test_inputs.unsqueeze(0))
                test_domain_masks.append(temp_meta_test_masks.unsqueeze(0))
                test_hs_masks.append(torch.cat([temp_test_sp, temp_test_hs1, temp_test_hs2, temp_test_hs3], dim=0).unsqueeze(0))

            test_idx_time = time.time()
            # print('test time:', e, test_idx_time - train_idx_time)

            train_domain_inputs = torch.cat(train_domain_inputs, dim=0)
            train_domain_masks = torch.cat(train_domain_masks, dim=0)
            train_hs_masks = torch.cat(train_hs_masks, dim=0)
            test_domain_inputs = torch.cat(test_domain_inputs, dim=0)
            test_domain_masks = torch.cat(test_domain_masks, dim=0)
            test_hs_masks = torch.cat(test_hs_masks, dim=0)

            # temp_meta_train_inputs, temp_meta_train_masks = self.seen_dataset_dict[e][meta_train_index]
            # temp_meta_test_inputs, temp_meta_test_masks = self.seen_dataset_dict[e][meta_test_index]

            meta_train_samples_inputs.append(train_domain_inputs.unsqueeze(0))
            meta_train_samples_masks.append(train_domain_masks.unsqueeze(0))
            meta_train_hs_masks.append(train_hs_masks.unsqueeze(0))
            
            meta_test_samples_inputs.append(test_domain_inputs.unsqueeze(0))
            meta_test_samples_masks.append(test_domain_masks.unsqueeze(0))
            meta_test_hs_masks.append(test_hs_masks.unsqueeze(0))

        # print('meta_train_samples_inputs:', meta_train_samples_inputs)
        # print('meta_train_samples_masks:', meta_train_samples_masks)
        # print('meta_test_samples_inputs:', meta_test_samples_inputs)
        # print('meta_test_samples_masks:', meta_test_samples_masks)

        meta_train_samples_inputs = torch.cat(meta_train_samples_inputs, dim=0)
        meta_train_samples_masks = torch.cat(meta_train_samples_masks, dim=0)
        meta_train_hs_masks = torch.cat(meta_test_hs_masks, dim=0)
        
        meta_test_samples_inputs = torch.cat(meta_test_samples_inputs, dim=0)
        meta_test_samples_masks = torch.cat(meta_test_samples_masks, dim=0)
        meta_test_hs_masks = torch.cat(meta_test_hs_masks, dim=0)

        # print('after_meta_train_samples_inputs:', after_meta_train_samples_inputs)
        # print('after_meta_train_samples_masks:', after_meta_train_samples_masks)
        # print('after_meta_test_samples_inputs:', after_meta_test_samples_inputs)
        # print('after_meta_test_samples_masks:', after_meta_test_samples_masks)

        '''
        train_inputs: Meta_bs * task_N * C * H * W

        {'train': [train_inputs, train_masks],
        'test' : [test_inputs, test_masks],}
        '''
#         domain_name = list(shuffled_seen_dataset_dict.keys())  # Mo, TNBC, CPM, CoNSep  
        domain_name = list(seen_dataset_dict_keys)
        # print('domain_name:', domain_name) #meta_train_hs_masks,
        return {'train': [meta_train_samples_inputs, meta_train_samples_masks, meta_test_hs_masks, label_his_b, label_his_f], 
                'test': [meta_test_samples_inputs, meta_test_samples_masks, meta_test_hs_masks],
                'domain_name': domain_name
                }
 
        # after_meta_train_samples_inputs, after_meta_train_samples_masks, after_meta_test_samples_inputs, after_meta_test_samples_masks


"""
    Notes
    -----
    #define init
    #Training dataset:   3, Seen
    #Test     dataset:   1, Unseen
    #input, label path for each 3
    #iteration: maximum iteration, it_max
    #for each iteration: define our batch size B
    #for each batch size: define our tasks (sample N datasets, , for each dataset, sample K_train+K_test images)
    # it_max * B * N
    # it_max B * N (1,2,3,4), (4,4,3,1)
    # for e in (1,2,3,4)
    # sample K image from each dataset
    # define index (i, i+k) % length of each dataset or random sample
    # load

    #define our getitem function:
    # return  NK * 3 * H * W, NK * 1 * H * W
"""


