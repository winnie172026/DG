import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import cv2
import sys
import skimage
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import torch
from collections import OrderedDict
from gradient import gradient_update_parameters
from maml.utils import tensors_to_device, compute_accuracy, compute_IoU, compute_loss, dice, _get_compactness_cost, TripletSemiHardLoss#, _get_coutour_sample
# from maml.metalearners.C2R import _get_coutour_sample #, _get_coutour_sample
from einops import rearrange
from maml.metric_net import get_metric_net
from maml.metalearners.C2R import C2R

import torchvision.transforms as T
from PIL import Image

__all__ = ['ModelAgnosticMetaLearning', 'MAML', 'FOMAML']


class ModelAgnosticMetaLearning(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.

    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).

    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.

    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].

    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.

    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.

    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].

    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.

    device : `torch.device` instance, optional
        The device on which the model is defined.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)

    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
    def __init__(self, model, optimizer=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=torch.nn.BCELoss, device=None, unseen_name=None):
        self.model = model[0].to(device=device)
        self.model_embedding = model[1].to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.get_compactness_cost = _get_compactness_cost
        self.triplet_loss = TripletSemiHardLoss
        self.margin = 10.0
        self.compactness_loss_weight = 1.0
        self.smoothness_loss_weight = 0.005
        self.metric_net_model = get_metric_net().to(device=device)
        self.contrastive_loss= C2R().to(device=device)
        self.bce_loss_weight = 1.0
        self.contrastive_loss_weight = 1.0 #.3 #0.2


        self.unseen_name = unseen_name
        dataset_list = ['MoNuSeg', 'TNBC', 'CoNSep', 'CPM']

        self.max_mIoU = {name: 0.0 for name in dataset_list if name != unseen_name}

        self.cl_hs_weights = self.curriculum_weight(self.unseen_name, self.max_mIoU)

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                dtype=param.dtype, device=self.device,
                requires_grad=learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                    for group in self.optimizer.param_groups])

    def sim_cl(self, x, y):

        return self.sim(x, y) / 0.5
    # def get_min(img):
    #     h,w = img.shape
    #     min_value = 255
    #     for i in range(h):
    #         for j in range(w):
    #             if img[i,j] < min_value:
    #                 min_value = img[i,j]
    #     return min_value
    #
    # def get_max(img):
    #     h,w = img.shape
    #     max_value = 0
    #     for i in range(h):
    #         for j in range(w):
    #             if img[i,j] > max_value:
    #                 max_value = img[i,j]
    #     return max_value

    def bce_loss(self, pred, target):

        target = target.to(pred.device)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        return -torch.mean(torch.log(pred) * target + torch.log(1 - pred) * (1 - target))

    def curriculum_weight(self, unseen, max_mIoU):
        '''
        Parameters
        ----------
        unseen:    string, unseen dataset name
        max_mIoU:  dictionary+string, dictionary of max mIoU of seen datasets

        Returns
        -------

        {'MoNuSeg': [0.71, 0.79, 0.84],
         'TNBC': [0.65, 0.76, 0.80],
        'CoNSep': [0.64, 0.76, 0.78],
         },

        '''

        cl_lambda = 0.3#0.3, lr=0.001
        selection = {'MoNuSeg': {'TNBC': [0.64, 0.76, 0.81],
                                 'CoNSep': [0.62, 0.76, 0.77],
                                 'CPM': [0.72, 0.81, 0.85], },
                     'TNBC': {'MoNuSeg': [0.69, 0.78, 0.84],
                              'CoNSep': [0.62, 0.75, 0.77],
                              'CPM': [0.73, 0.79, 0.85], },
                     'CoNSep': {'MoNuSeg': [0.70, 0.81, 0.84],
                                'TNBC': [0.66, 0.79, 0.80],
                                'CPM': [0.76, 0.79, 0.80], },
                     'CPM': {'MoNuSeg': [0.71, 0.79, 0.84],
                             'TNBC': [0.65, 0.76, 0.80],
                             'CoNSep': [0.64, 0.76, 0.78],
                             },
                     }
#         selection = {'MoNuSeg': {'TNBC': [0.5, 0.6, 0.75],
#                                  'CoNSep': [0.4, 0.55, 0.65],
#                                  'CPM': [0.6, 0.7, 0.75], },
#                      'TNBC': {'MoNuSeg': [0.63, 0.75, 0.8],
#                               'CoNSep': [0.45, 0.58, 0.67],
#                               'CPM': [0.63, 0.75, 0.8], },
#                      'CoNSep': {'MoNuSeg': [0.6, 0.7, 0.75],
#                                 'TNBC': [0.6, 0.7, 0.8],
#                                 'CPM': [0.67, 0.75, 0.80], },
#                      'CPM': {'MoNuSeg': [0.68, 0.75, 0.8],
#                              'TNBC': [0.57, 0.7, 0.75],
#                              'CoNSep': [0.56, 0.67, 0.75],
#                              },
#                      }
#         selection = {'MoNuSeg': {'TNBC': [0.4, 0.6, 0.75],
#                                  'CoNSep': [0.4, 0.55, 0.65],
#                                  'CPM': [0.5, 0.7, 0.75], },
#                      'TNBC': {'MoNuSeg': [0.5, 0.75, 0.8],
#                               'CoNSep': [0.4, 0.58, 0.67],
#                               'CPM': [0.5, 0.65, 0.7], },
#                      'CoNSep': {'MoNuSeg': [0.5, 0.65, 0.75],
#                                 'TNBC': [0.5, 0.6, 0.7],
#                                 'CPM': [0.6, 0.7, 0.8], },
#                      'CPM': {'MoNuSeg': [0.6, 0.7, 0.8],
#                              'TNBC': [0.5, 0.65, 0.75],
#                              'CoNSep': [0.5, 0.63, 0.75],
#                              },
#                      }

        #print('unseen: ', unseen)
        difficulty_dict = selection[unseen]
        weights_dict = {name: [] for name in difficulty_dict}
        #seen_dt = {name: [0., 0., 0.] for name in selection if name != unseen}
        #print('seen_dt: ', seen_dt)


        for name in difficulty_dict:
            #print('key: ', name)

            weights = [(1 + cl_lambda) if max_mIoU[name] > difficulty_level else (1 - cl_lambda) for difficulty_level in difficulty_dict[name]]

            weights.append(len(difficulty_dict[name]))
            new_weights = [i / sum(weights) for i in weights]

            weights_dict[name] = new_weights
            print('weights dict: ', weights_dict)

        return weights_dict

    def cal_max_mIoU(self, unseen, trainingdataloader_dict):

        #do evaluatino on each dataset
        #calulate mIoU
        #get mIoU = {'dataset1':..., 'dataset2'...}
        #
        self.model.eval()
        mIoU = {name: 0.0 for name in trainingdataloader_dict.keys()}
        with torch.no_grad():
            for name in trainingdataloader_dict.keys():
                cur_iou = 0
                for i, batch in enumerate(trainingdataloader_dict[name]):
                    # print('batch: ', len(batch))
                    data, target, _, _, _, _, _, _, _, _ = batch
                    data, target = data.to(self.device), target.to(self.device)

                    output,_,_ = self.model(data)
                    cur_iou += compute_IoU(output, target)

                mIoU[name] = cur_iou / len(trainingdataloader_dict[name])
                self.max_mIoU[name] = max(self.max_mIoU[name], mIoU[name])

        # max_mIoU = {'TNBC': 0.50, 'MoNuSeg': 0.6, 'CPM': 0.7}
        self.cl_hs_weights = self.curriculum_weight(unseen, self.max_mIoU)
        self.model.train()
        print('max_mIoU: ', self.max_mIoU)

    def extract_coutour_embedding(self, contour, embeddings):
        # print('contour: ', contour.shape)
        # print('embedding: ', embeddings.shape)
        # contour: 4x1x256x256
        # embeddings: 4x96x256x256
        # contour_embeddings: 4x96x256x256
        # sum_embeddings: 4x96
        # sum_contour: 4x1
        # average_embeddings: 4x96
        # print('contour shape: ', contour.shape)
        # print('embeddings shape: ', embeddings.shape)
        # print('contour type: ', contour.dtype)
        # print('embeddings type: ', embeddings.dtype)
        contour_embeddings = contour * embeddings
        sum_embeddings = torch.sum(contour_embeddings, dim=(2, 3))
        sum_contour = torch.sum(contour, dim=(2, 3))
        #sum_contour = torch.ones((4, 1)).float().cuda()
        #sum_contour[0, 0] = 0.0
        average_embeddings = sum_embeddings / (sum_contour+1)

        # print('sum_embeddings: ', torch.sum(sum_embeddings))
        # print()
        # print('sum_contour: ', sum_contour)
        # print()
        # print('average_embeddings: ', torch.sum(average_embeddings))
        # print()

        return average_embeddings


    def get_outer_loss(self, batch, save_pth=None):

        # if 'test' not in batch:
        #     raise RuntimeError('The batch does not contain any test dataset.')
        # print('outer loss batch:', len(batch))

        # print(batch)
        # print(batch.keys())

        train_inputs_perdomin, train_targets_perdomin, train_hs_perdomain, label_his_b, label_his_f = batch['train']
        test_inputs_perdomin, test_targets_perdomin, test_hs_perdomain = batch['test']

        print('train_inputs_perdomain:', train_inputs_perdomin.size())
        print('train_targets_perdomain:', train_targets_perdomin.size())
        print('test_inputs_perdomain:', test_inputs_perdomin.size())
        print('test_targets_perdomain:', test_targets_perdomin.size())
        print('----------------------------------------------------')

        num_tasks = test_targets_perdomin.size(1)

        print('num_tasks:', num_tasks)
        print("test_targets_perdomin:", test_targets_perdomin.dtype)
        is_classification_task = (not test_targets_perdomin.dtype.is_floating_point)

        # print('is classification task:', is_classification_task)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses_bce': np.zeros((num_tasks,), dtype=np.float32),
            'outer_losses_cl': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }

        results.update({
            'mIoU_before': np.zeros((num_tasks,), dtype=np.float32),
            'mIoU_after': np.zeros((num_tasks,), dtype=np.float32)
        })

        mean_outer_loss = torch.tensor(0., device=self.device)

        train_inputs_perdomin = rearrange(train_inputs_perdomin, 'b d db c h w -> d (b db) c h w')
        train_targets_perdomin = rearrange(train_targets_perdomin, 'b d db c h w -> d (b db) c h w')
        train_hs_perdomain = rearrange(train_hs_perdomain, 'b d db c h w -> d (b db) c h w')
        test_inputs_perdomin = rearrange(test_inputs_perdomin, 'b d db c h w -> d (b db) c h w')
        test_targets_perdomin = rearrange(test_targets_perdomin, 'b d db c h w -> d (b db) c h w')
        test_hs_perdomain = rearrange(test_hs_perdomain, 'b d db c h w -> d (b db) c h w')

        print('train inputs predomain:', train_inputs_perdomin.shape)
        print('test inputs perdomain:', test_inputs_perdomin.shape)
        print('################################################')

        for task_id, (train_inputs, train_targets, test_inputs, test_targets, train_hs, test_hs, domain_name) \
                in enumerate(zip(train_inputs_perdomin, train_targets_perdomin, train_hs_perdomain, \
                                 test_inputs_perdomin, test_targets_perdomin, test_hs_perdomain, batch['domain_name'])):
            #
            #print('train_inputs:', train_inputs.shape)
            #print('train_targets:', train_targets.shape)
#             print('!!!!!!!!!!!!!!!!!!!!!!!!domain name', domain_name)
            # print('task id:', task_id)
            # print()
            
            train_inputs = train_inputs.to(device=self.device)
            train_targets = train_targets.to(device=self.device)



            params, adaptation_results = self.adapt(train_inputs, train_targets, label_his_b, label_his_f, domain_name,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            results['mIoU_before'][task_id] = adaptation_results['mIoU_before']

            with torch.set_grad_enabled(self.model.training):
                # print('test_inputs shape:', test_inputs.shape)


                test_inputs = test_inputs.to(device=self.device)
                print('test_inputs shape:', test_inputs.shape)

                test_logits, layer_features, hist_features = self.model(test_inputs, params=params) # for ours


                embeddings = self.model_embedding(layer_features) # for ours
                test_targets = test_targets.to(device=self.device)
                test_hs = test_hs.to(device=self.device)
                # print('test_logits:', test_logits.shape)
                # print('test_targets:', test_targets.shape)
                # contour_group, metric_label_group = _get_coutour_sample(test_targets)


                test_ps = (test_hs[0]).reshape(test_hs[0].shape[0], 1, test_hs[0].shape[1], test_hs[0].shape[2])
                test_hs_n1 = (test_hs[1]).reshape(test_hs[1].shape[0], 1, test_hs[1].shape[1], test_hs[1].shape[2])
                test_hs_n2 = (test_hs[2]).reshape(test_hs[2].shape[0], 1, test_hs[2].shape[1], test_hs[2].shape[2])
                test_hs_n3 = (test_hs[3]).reshape(test_hs[3].shape[0], 1, test_hs[3].shape[1], test_hs[3].shape[2])


                # contour_group, metric_label_group = self.extract_coutour_embedding(test_targets, test_logits, test_ps, test_hs_n1, test_hs_n2, test_hs_n3)
                # print('contour_group::::::::::', contour_group)



                # embedding, y_pred, y_true, ps, hs1, hs2, hs3, cl_weights
#                 print('C2R weight:', self.cl_hs_weights[domain_name[0]])

                contrastive_loss = self.contrastive_loss(embeddings, test_logits, test_targets, test_ps, \
                                                         test_hs_n1, test_hs_n2, test_hs_n3, self.cl_hs_weights[domain_name[0]])
                #

                #
#                 print('C2R LOSS:', contrastive_loss)
                # print('------------------------------------')
                # coutour_embeddings = self.extract_coutour_embedding(contour_group, embeddings)
                # metric_embeddings = self.metric_net_model(coutour_embeddings)  # 4 * 24
                # # print('!!!!!embeddings:', embeddings)
                # # print('contour_group:', contour_group)
                # #
                # smoothness_loss_b = self.triplet_loss(metric_label_group[..., 0], metric_embeddings, self.device)
                # # smoothness_loss_b = self.contrastive_loss(metric_label_group[..., 0], metric_embeddings, self.device)
                # smoothness_loss_b = self.smoothness_loss_weight * smoothness_loss_b
                # # print(smoothness_loss_b)

                # print('test_logits:', test_logits.shape)
                # print('test_targets:', test_targets.shape)
                # contour_group, metric_label_group = _get_coutour_sample(test_targets)

                # print('contour_group::::::::::', contour_group.shape)

                # # compute compactness loss
                # compactness_loss_b, length, area, boundary_b = self.get_compactness_cost(test_logits, test_targets)
                # compactness_loss_b = self.compactness_loss_weight * compactness_loss_b
                #
                # #
                # coutour_embeddings = self.extract_coutour_embedding(contour_group, embeddings)
                # metric_embeddings = self.metric_net_model(coutour_embeddings)  # 4 * 24
                # #
                # smoothness_loss_b = self.triplet_loss(metric_label_group[..., 0], metric_embeddings, self.device)
                # smoothness_loss_b = self.smoothness_loss_weight * smoothness_loss_b
                # print(smoothness_loss_b)
                


                bce_loss = self.bce_loss_weight * self.bce_loss(test_logits, test_targets)
                contrastive_loss = self.contrastive_loss_weight * contrastive_loss
                outer_iou = compute_IoU(test_logits, test_targets)
                outer_dice = dice(test_logits, test_targets)

                #outer_loss = self.bce_loss(test_logits, test_targets) + self.contrastive_loss_weight * contrastive_loss #+ compactness_loss_b + smoothness_loss_b #
                outer_loss = bce_loss + contrastive_loss #compactness_loss_b + smoothness_loss_b#+ contrastive_loss # + compactness_loss
                results['outer_losses_bce'][task_id] = bce_loss.item()
                results['outer_losses_cl'][task_id] = contrastive_loss.item()
                # results['outer_losses_compactness'][task_id] = compactness_loss.item()
                mean_outer_loss += outer_loss
                

            if is_classification_task:
                results['mIoU_after'][task_id] = compute_IoU(
                    test_logits, test_targets)


        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()
        outer_iou = np.mean(outer_iou)
        outer_dice = np.mean(outer_dice)

        return mean_outer_loss, results, outer_iou, outer_dice


    def adapt(self, inputs, targets, targets_his_b, targets_his_f, domain_name, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False):

        params = None
        # print('num adaptation steps:', num_adaptation_steps)
        # print('step size:', step_size)
        # print('is_classification_task:', is_classification_task)
#         print('inputs shape:',inputs.shape)
#         print('targets shape:', targets.shape)
#         print('targets_his_b shape:', targets_his_b.shape)
#         print('targets_his_f shape:', targets_his_f.shape)

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            logits, _, hist_features= self.model(inputs, params=params)

            ########### add histogram feature loss funciton here ##############
            ## hist1=bacakground hist2= foreground
            # 1/(3*256) * |color_hist_b - color_hist_b_gt|
            #                 np.clip(test_targets.cpu(), 0, 1)
            # hist_gram = np.zeros((3, 256))
            # hist_gram[:, 0] = 0

#             color_hist_b_gt = inputs * (1 - targets)
#             color_hist_f_gt = inputs * targets

#             color_hist_b_gt = color_hist_b_gt.cpu().numpy()
#             color_hist_f_gt = color_hist_f_gt.cpu().numpy()
            #
#             print('color_hist_b_gt:', targets_hist_b.shape)
#             print('color_hist_f_gt:', targets_hist_f.shape)
            #
            targets_his_b = np.array(targets_his_b)
            targets_his_f = np.array(targets_his_f)
            
#             print('targets_his_b type:', targets_his_b.dtype)
#             print('targets_his_f type:', targets_his_f.dtype)
            loss_hist_b_t = 0.0
            for i in range(len(targets_his_b)):
                # print('i:', i) 
                  
                # image_red, image_green, image_blue = color_hist_b_gt[i][:, :, 0], color_hist_b_gt[i][:, :, 1], \
                #                                         color_hist_b_gt[i][:, :, 2]
                # print('red, green, blue:', image_red.shape, image_green.shape, image_blue.shape)
                

                hist_red_b = cv2.calcHist([targets_his_b[i]], [0], None, [256], [0, 256])
                hist_green_b = cv2.calcHist([targets_his_b[i]], [1], None, [256], [0, 256])
                hist_blue_b = cv2.calcHist([targets_his_b[i]], [2], None, [256], [0, 256])

#                 print('hist bin RGB:', hist_red_b.shape, hist_green_b.shape, hist_blue_b.shape)
                hist_red_b = np.squeeze(hist_red_b)
                hist_green_b = np.squeeze(hist_green_b)
                hist_blue_b = np.squeeze(hist_blue_b)
                
                hist_red_b[0] = 0
                hist_green_b[0] = 0
                hist_blue_b[0] = 0
                # print('hist bin RGB:', hist_red_b.shape, hist_green_b.shape, hist_blue_b.shape)
                hist_b = np.array([hist_red_b, hist_green_b, hist_blue_b])
#                 print('histograms_array:', hist_b.shape) # (3, 256)
#                 print('histo bg shape:', hist_features[0][i].shape) # (3, 256)
#                 print('hist loss:', np.sum(np.abs(hist_b - hist_features[0][i].cpu().detach().numpy())))

                # loss_hist_b = np.abs(hist_b - hist_features[0][i].cpu().detach().numpy())
                loss_hist_b = cosine_similarity(hist_b, hist_features[0][i].cpu().detach().numpy())
                
                loss_hist_b = loss_hist_b / (3 * 256)
                # print('loss_hist_b each iter:', loss_hist_b)
                loss_hist_b_t += loss_hist_b 
                # print('loss_hist_b_t:', loss_hist_b_t)
            # print(len(color_hist_b_gt))
            # print('loss_hist_b_t:', loss_hist_b_t)
            loss_hist_b_mean = np.mean(loss_hist_b_t / len(targets_his_b))
            # print('loss_hist bg mean:', loss_hist_b_mean.item())
            # np.clip(gt, 0, 1)
            # hist_gram = np.zeros((3, 256))
            # hist_gram[:,0] = 0

            # skimage color_hist
            # time.time()
            # image1, image2, color_hist(image1), save color_hist image1_hist_gram.npy
            # dataloader load image1hist_gra.py

            loss_hist_f_t = 0.0
            for i in range(len(targets_his_f)):

                hist_red_f = cv2.calcHist([targets_his_f[i]], [0], None, [256], [0, 256])
                hist_green_f = cv2.calcHist([targets_his_f[i]], [1], None, [256], [0, 256])
                hist_blue_f = cv2.calcHist([targets_his_f[i]], [2], None, [256], [0, 256])

                hist_red_f = np.squeeze(hist_red_f)
                hist_green_f = np.squeeze(hist_green_f)
                hist_blue_f = np.squeeze(hist_blue_f)
                
                hist_red_b[0] = 0
                hist_green_b[0] = 0
                hist_blue_b[0] = 0
                # print('hist bin RGB:', hist_red.shape, hist_green.shape, hist_blue.shape)
                hist_f = np.array([hist_red_f, hist_green_f, hist_blue_f])
                # print('histograms_array:', hist) # (3, 256)
                # print('histo1, 2 shape:', hist_features[0][i]) # (3, 256)
                
                # print('hist_f :', hist_f.shape)
                # print('hist_features[1][i]:', hist_features[1][i].shape)

                # loss_hist_f = np.abs(hist_f - hist_features[1][i].cpu().detach().numpy())
#                 print('loss_hist_f:', loss_hist_f)
                loss_hist_f = cosine_similarity(hist_f, hist_features[1][i].cpu().detach().numpy())
                loss_hist_f = loss_hist_f / (3 * 256)
#                 print('loss_hist_b each iter:', loss_hist_f)
                loss_hist_f_t += loss_hist_f
                # print('single loss_hist_f:', loss_hist_f)
                # print('loss_hist_f_t:', loss_hist_f_t)
            loss_hist_f_mean = np.mean(loss_hist_f_t / len(targets_his_f))
            # print('loss_hist fg mean:', loss_hist_f_mean.item())

            # ###################################################################
            hist_loss = loss_hist_b_mean + loss_hist_f_mean
            bce_loss = self.bce_loss(logits, targets)
            print('histo loss:', hist_loss.item())
            # print('cell and bg hist loss:', hist_loss.item())


            ########################### contrastive loss #######################
            #train_embeddings = logits
            #train_hs = train_hs.to(device=self.device)
            # print('test_logits:', test_logits.shape)
            # print('test_targets:', test_targets.shape)
            # contour_group, metric_label_group = _get_coutour_sample(test_targets)

            #train_ps = (train_hs[0]).reshape(train_hs[0].shape[0], 1, train_hs[0].shape[1], train_hs[0].shape[2])
            #train_hs_n1 = (train_hs[1]).reshape(train_hs[1].shape[0], 1, train_hs[1].shape[1], train_hs[1].shape[2])
            #train_hs_n2 = (train_hs[2]).reshape(train_hs[2].shape[0], 1, train_hs[2].shape[1], train_hs[2].shape[2])
            #train_hs_n3 = (train_hs[3]).reshape(train_hs[3].shape[0], 1, train_hs[3].shape[1], train_hs[3].shape[2])

            # contour_group, metric_label_group = self.extract_coutour_embedding(test_targets, test_logits, test_ps, test_hs_n1, test_hs_n2, test_hs_n3)
            # print('contour_group::::::::::', contour_group)

            # embedding, y_pred, y_true, ps, hs1, hs2, hs3, cl_weights
            #                 print('C2R weight:', self.cl_hs_weights[domain_name[0]])

            #contrastive_loss = self.contrastive_loss(train_embeddings, logits, targets, train_ps, \
            #                                         train_hs_n1, train_hs_n2, train_hs_n3,
            #                                         self.cl_hs_weights[domain_name[0]])

            inner_loss = bce_loss + hist_loss #+ 0.3 * contrastive_loss

            print('inner loss:', inner_loss.item())
            print('-----------------------------------------')
            results['inner_losses'][step] = inner_loss.item()

            if (step == 0): #and is_classification_task:
                # results['accuracy_before'] = compute_accuracy(logits, targets)
                results['mIoU_before'] = compute_IoU(logits.detach(), targets)

            self.model.zero_grad()
            #inner_loss.backward()
            params = gradient_update_parameters(self.model, inner_loss,
                step_size=step_size, params=params,
                first_order=(not self.model.training) or first_order)

        return params, results

    def train(self, dataloader, max_batches=500, verbose=True, save_pth=None, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches, save_pth=save_pth):
                pbar.update(1)
                # print('save path:', save_pth)
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(
                        np.mean(results['accuracies_after']))
                pbar.set_postfix(**postfix)

    def train_iter(self, dataloader, max_batches=500, save_pth=None):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        dice_total =0.0
        iou_total =0.0
        
        mean_dice = 0.0
        mean_iou =0.0
        self.model.train()
        while num_batches < max_batches:
#             print('num batches:', num_batches)
            for i, batch in enumerate(dataloader):
#                 print('batch:', len(batch))
                # print(i)
                # train_inputs_perdomin, train_targets_perdomin = batch['train']
                # test_inputs_perdomin, test_targets_perdomin = batch['test']

                # print()
                if num_batches >= max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()

                outer_loss, results, iou, dice = self.get_outer_loss(batch)
                print('outer loss:', outer_loss.item())
                print('results BCE loss: ', results['outer_losses_bce'])
                print('results CL loss: ', results['outer_losses_cl'])
                print('results IoU: ', iou)
                print('results Dice: ', dice)


                yield results
                #
                # if self.scheduler is not None:


#                 with open(save_pth + '/train_result.txt', 'a') as f:
#                     f.write('mean_outer_loss: {}, IoU: {}, Dice: {}'.format(outer_loss, iou, dice))
#                     f.write('\n')

                outer_loss.backward()
                self.optimizer.step()

                num_batches += 1
                dice_total += dice
                iou_total += iou
        mean_dice = dice_total / num_batches
        mean_iou = iou_total / num_batches

        print('Evaluation mean IoU:', mean_iou)
        print('Evaluation mean Dice:', mean_dice)
        with open(save_pth + '/train_results.txt', 'a') as f_t:
            f_t.write('mean_outer_loss: {}, IoU: {}, Dice: {}'.format(outer_loss, mean_iou, mean_dice))
            f_t.write('\n')
    # def get_outer_loss_test(self, batch_unseen):

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count, mIoU, mDice = 0., 0., 0., 0., 0.

        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches, save_pth=kwargs['save_pth'], epoch = kwargs['epoch']):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count

                # 'mean_outer_loss': mean_outer_loss, 'IoU': IoU, 'Dice': Dice

                mIoU += (results['IoU'] - mIoU) / count
                mDice += (results['Dice'] - mDice) / count
                print('mIoU:', mIoU)
                print('mDice:', mDice)
                print('mean_outer_loss:', mean_outer_loss)

                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'mIoU_after' in results:
                    mean_accuracy += (np.mean(results['mIoU_after'])
                        - mean_accuracy) / count
                    postfix['mIoU'] = '{0:.4f}'.format(mIoU)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss, 'IoU': mIoU, 'Dice': mDice}
        if 'accuracies_after' in results:
            mean_results['mIoU_after'] = mIoU

        print('Done evaluating!!!!!!!!!!!!!')
        print()

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500, save_pth=None, epoch=None):
        num_batches = 0
        self.model.eval()
        dice_total =0.0
        iou_total =0.0
        
        mean_dice = 0.0
        mean_iou =0.0
       
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break
                print('batch:', len(batch))
                #batch = tensors_to_device(batch, device=self.device)
                inputs, targets, _, _, _, _, _, _, _ = batch

                with torch.no_grad():
                    inputs = inputs.to(self.device)
                    seg_result,_,_ = self.model(inputs)

                IoU = compute_IoU(seg_result, targets)
                mean_outer_loss = self.bce_loss(seg_result, targets)   #compute_loss(seg_result, targets)
                Dice = dice(seg_result, targets)

                print('Evaluation Loss:', mean_outer_loss)
                print('Evaluation IoU:', IoU)
                print('Evaluation Dice:', Dice)

                seg_result = (seg_result.cpu().squeeze(0).squeeze(0) > 0.5).float()
                seg_result = seg_result.numpy()
                seg_result = np.clip(seg_result * 255, 0, 255).astype(np.uint8)
                # seg_result = Image.fromarray(seg_result).convert('RGB')
                # print('seg_result shape:', seg_result.shape)
                #
                # print('target shape:', targets.shape)
                targets = targets.cpu().squeeze(0).squeeze(0)
                targets = targets.numpy()
                targets = np.clip(targets * 255, 0, 255).astype(np.uint8)

                seg_targets = np.concatenate([seg_result, targets], axis=1)
                seg_targets = Image.fromarray(seg_targets)

                save_pth_img = save_pth + '/' + str(epoch)

                if not os.path.exists(save_pth_img):
                    os.makedirs(save_pth_img)

                seg_targets.save((save_pth_img+'/seg_targets_{}.png').format(num_batches))

                results = {'mean_outer_loss': mean_outer_loss, 'IoU': IoU, 'Dice': Dice}

                with open(save_pth+'/result.txt', 'a') as f:
                    f.write('mean_outer_loss: {}, IoU: {}, Dice: {}'.format(mean_outer_loss, IoU, Dice))
                    f.write('\n')

                yield results

                num_batches += 1
                
                
                dice_total += Dice
                iou_total += IoU
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                print('Total dice:', dice_total)
                print('Total IoU:', iou_total)
                print('len of unseen:', num_batches)

            mean_dice = dice_total / num_batches
            mean_iou = iou_total / num_batches

            print('Evaluation mean IoU:', mean_iou)
            print('Evaluation mean Dice:', mean_dice)
        
        with open(save_pth+'/result.txt', 'a') as f:
            f.write('\n')
            f.write('---------------------------------------------------------------')
            f.write('\n')
            f.write('Mean evaluation IoU: {} and Dice: {}'.format(mean_iou, mean_dice)) 
            f.write('\n')
            f.write('---------------------------------------------------------------')
            f.write('\n')
        


MAML = ModelAgnosticMetaLearning

class FOMAML(ModelAgnosticMetaLearning):
    def __init__(self, model, optimizer=None, step_size=0.1,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        super(FOMAML, self).__init__(model, optimizer=optimizer, first_order=True,
            step_size=step_size, learn_step_size=learn_step_size,
            per_param_step_size=per_param_step_size,
            num_adaptation_steps=num_adaptation_steps, scheduler=scheduler,
            loss_function=loss_function, device=device)

