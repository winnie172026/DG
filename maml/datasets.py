import torch.nn.functional as F

from collections import namedtuple

from maml.model import testmodel, Embedding_Net #ModelConvUnet, 
from maml.nuclei_dataset import Nuclei_dataset
from maml.NEW_dataset import MoNuSeg as Mo
from maml.NEW_dataset import TNBC as TN
from maml.NEW_dataset import CoNSep as Co
from maml.NEW_dataset import CPM as cpm

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_test_dataset '
                                    'model loss_function')

def get_benchmark_by_name(name,
                          meta_train_bs,
                          meta_test_bs,
                          num_iterations):
    dataset_dict = {'MoNuSeg': Mo.MoNuSeg, 'TNBC': TN.TNBC, 'CPM': cpm.CPM, 'CoNSep': Co.CoNSep}

    unseen_dataset_dict = {dataset_name: dataset_dict[dataset_name] for dataset_name in dataset_dict if
                              dataset_name == name}

    seen_dataset_dict = {dataset_name: dataset_dict[dataset_name](name) for dataset_name in dataset_dict if
                                  dataset_name != name}

    # print('unseen_dataset_dict:', unseen_dataset_dict)

    meta_train_dataset = Nuclei_dataset(meta_learn_iteration=num_iterations,
                                        unseen_dataset = name,
                                        meta_train_sample = meta_train_bs,
                                        meta_test_sample = meta_test_bs)
    unseen_dataset = unseen_dataset_dict[name](None)

    #model = ModelConvUnet(in_channel=3, out_channel=1, feature_size=32)'
    model = [testmodel(), Embedding_Net()]
    loss_function = F.cross_entropy

    return meta_train_dataset, unseen_dataset, model, loss_function, seen_dataset_dict