import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

sys.path.append('/')
class C2R(nn.Module):
    def __init__(self, weights_each_embedding_layer = [1/32.0, 1/16.0, 1/2.0, 1.0]):
        super(C2R, self).__init__()
        self.weights = weights_each_embedding_layer
        self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.l1 = nn.L1Loss()

    def sim_cl(self, x,y):
        return self.sim(x,y)/0.5


    def reduce_mean(self, embed, label):
         
        embed = label * embed
        embed = torch.sum(embed, dim=[2,3])

        label_sum = torch.sum(label, dim=[2,3])

        # print()
        # print()
        # print()
        # print('Embed and label')
        # print(embed, label_sum)

        embed = (embed+1) / (label_sum + 1)
        return embed

    def forward(self, embedding, y_pred, y_true, ps, hs1, hs2, hs3, cl_weights):

        """Computes the triplet loss_functions with semi-hard negative mining.
           The loss_functions encourages the positive distances (between a pair of embeddings
           with the same labels) to be smaller than the minimum negative distance
           among which are at least greater than the positive distance plus the
           margin constant (called semi-hard negative) in the mini-batch.
           If no such negative exists, uses the largest negative distance instead.
           See: https://arxiv.org/abs/1503.03832.
           We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
           [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
           2-D float `Tensor` of l2 normalized embedding vectors.
           Args:
             margin: Float, margin term in the loss_functions definition. Default value is 1.0.
             name: Optional name for the op.
           """
        # print('embedding shape:', embedding[0].shape)
        #False positive -> <- True positive
        loss = 0
        hs1_weigths, hs2_weights, hs3_weights, ps_weights = cl_weights
        # print('cl weights:', hs1_weigths,hs2_weights,hs3_weights, ps_weights)
        positive_smaple = y_true
        negative_sample = 1-y_true
        y_pred_binary = (y_pred.detach() >0.5).float()
        tp = (positive_smaple * y_pred_binary)
        fp = positive_smaple - tp
        tn = negative_sample * (1-y_pred_binary)
        # fn = (1-positive_smaple) - tn

        # # Anchor -> <- Positive
        # ap = positive_smaple * y_pred_binary
        # an = positive_smaple * (1-y_pred_binary)
        # # Anchor <- -> Negative
        # nn = negative_sample * (1-y_pred_binary)
        # np = negative_sample * y_pred_binary

        # # False positive < > True positive
        # tp = positive_smaple * y_pred_binary
        # fp = positive_smaple * (1-y_pred_binary)
        # # False negative < > True negative
        # tn = negative_sample * (1-y_pred_binary)
        # fn = negative_sample * y_pred_binary
        
#         print('device:', ps.device,negative_sample.device)
        ps = ps.to(device)
        hs1 = hs1.to(device)
        hs2 = hs2.to(device)
        hs3 = hs3.to(device)
#         print('device changed:', ps.device)
        #False positive < > True negative
        tn_ps = negative_sample * (1-ps)  #true negative
        fn_ps = negative_sample - tn_ps  #false negative
        # fn_ps = negative_sample * ps  #false negative

        tn_hs1 = negative_sample * (1-hs1)  #true negative
        fn_hs1 = negative_sample - tn_hs1  #false negative
        # fn_hs1 = negative_sample * hs1  #false negative

        tn_hs2 = negative_sample * (1-hs2)  #true negative
        fn_hs2 = negative_sample - tn_hs2  #false negative
        # fn_hs2 = negative_sample * hs2  #false negative

        tn_hs3 = negative_sample * (1-hs3)  #true negative
        fn_hs3 = negative_sample - tn_hs3  #false negative
        # fn_hs3 = negative_sample * hs3  #false negative

        # print()
        # print()
        # print()
        # print()
        # print('fp, tp, fn_ps, fn_hs1, fn_hs2, fn_hs3')
        # print(torch.sum(fp, dim=[2, 3]), \
        #       torch.sum(tp, dim=[2, 3]), \
        #       torch.sum(fn_ps, dim=[2, 3]), \
        #       torch.sum(fn_hs1, dim=[2, 3]),\
        #       torch.sum(fn_hs2, dim=[2, 3]), \
        #       torch.sum(fn_hs3, dim=[2, 3]))

        #print('embedding:', embedding[0].shape, embedding[1].shape)
        for i in range(len(self.weights)): 
            
            # print("embedding[i].shape", embedding[i].shape)
            # print("fp.shape", fp.shape)
            # print("tp.shape", tp.shape)
            #minimise simislarity

            # need the embedding close to the positive sample
            # need the embedding far from the negative sample

            ######## writen by Ruoyu
            # print(fp.shape)
            # print('sum:',  torch.sum(fp, dim=[2,3]))
            positive_dis = self.sim_cl(self.reduce_mean(embedding[i], fp), self.reduce_mean(embedding[i], tp).detach())


            # #pos = fn*embedding[i] ...
            # #maximise dissimilarity


            negative_dis_ps = self.sim_cl(self.reduce_mean(embedding[i], fp), self.reduce_mean(embedding[i], fn_ps).detach())
            negative_dis_hs1 = self.sim_cl(self.reduce_mean(embedding[i], fp), self.reduce_mean(embedding[i], fn_hs1).detach())
            negative_dis_hs2 = self.sim_cl(self.reduce_mean(embedding[i], fp), self.reduce_mean(embedding[i], fn_hs2).detach())
            negative_dis_hs3 = self.sim_cl(self.reduce_mean(embedding[i], fp), self.reduce_mean(embedding[i], fn_hs3).detach())
            
#             print('fp:', np.unique(fp.cpu()))
#             print('fn ps:', np.unique(fn_ps.cpu()))
#             print('fn hs1:', np.unique(fn_hs1.cpu()))
#             print('fn hs2:', np.unique(fn_hs2.cpu()))
#             print('fn hs3:', np.unique(fn_hs3.cpu()))
#             print('embedding:', embedding[i])

#             print('C2R: positive_dis:', positive_dis)
#             print('C2R: negative_dis_ps:', negative_dis_ps)
#             print('C2R: negative_dis_hs1:', negative_dis_hs1)
#             print('C2R: negative_dis_hs2:', negative_dis_hs2)
#             print('C2R: negative_dis_hs3:', negative_dis_hs3)

            # negative_dis_ps = self.l1(fn_ps*embedding[i], tn*embedding[i].detach())
            # negative_dis_hs1 = self.l1(fn_hs1*embedding[i], tn*embedding[i].detach())
            # negative_dis_hs2 = self.l1(fn_hs2*embedding[i], tn*embedding[i].detach())
            # negative_dis_hs3 = self.l1(fn_hs3*embedding[i], tn*embedding[i].detach())
            #
            contrastive = -(torch.log (positive_dis / (negative_dis_ps * ps_weights \
                                          + negative_dis_hs1 * hs1_weigths \
                                          + negative_dis_hs2 * hs2_weights \
                                          + negative_dis_hs3 * hs3_weights \
                                          + 1e-7))) 

#             contrastive = -(torch.log (positive_dis / (negative_dis_ps * ps_weights+1e-7)))
    
            contrastive = torch.mean(contrastive)


            ######### end
#             print('contrastive:', contrastive)

            loss += self.weights[i] * contrastive

        return loss