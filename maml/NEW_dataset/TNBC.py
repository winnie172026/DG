import os
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
import numpy as np
from torchvision import transforms

class TNBC(Dataset):
    name = 'TNBC'
    out_channels = 1

    def __init__(self, unseen_data, transform = None):
        super(TNBC, self).__init__()

        TNBC_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/patches/TNBC/'
        TNBC_sp_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/sp/TN_sp/'

        # self.root = os.path.expanduser(root)

        self.imgdir = osp.join(TNBC_dir, 'images')
        self.maskdir = osp.join(TNBC_dir, 'labels')
        self.spdir = TNBC_sp_dir

        self.unseen_data = unseen_data
        if self.unseen_data is not None:
            self.sample1_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/new_hard_samples_825/D2E2Net_pred_unseen' \
                               + unseen_data + '/on_TN'
            self.sample2_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/new_hard_samples_825/UNet_pred_unseen' \
                               + unseen_data + '/on_TN'
            self.sample3_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/new_hard_samples_825/SegNet_pred_unseen' \
                               + unseen_data + '/on_TN'
            self.sample1_list = sorted(os.listdir(self.sample1_dir))
            self.sample2_list = sorted(os.listdir(self.sample2_dir))
            self.sample3_list = sorted(os.listdir(self.sample3_dir))

        self.imglist = sorted(os.listdir(self.imgdir))
        self.masklist = sorted([item.replace('.png', '_label.png') for item in self.imglist])
        self.splist = sorted(os.listdir(self.spdir))

        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.resize = transforms.Resize((256, 256))
        # self.randomcrop = RandomCrop.RandomCrop(256)

    def __getitem__(self, idx):  # return one class
        # print('idx:', idx)
        # len_idx = len(idx)

        image_name = osp.join(self.imgdir, self.imglist[idx])
        mask_name = osp.join(self.maskdir, self.imglist[idx].replace(".png", "_label.png"))
#         sp_name = osp.join(self.spdir, self.splist[idx]) 
        sp_name = osp.join(self.spdir, self.imglist[idx])

        image = Image.open(image_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        sp = Image.open(sp_name).convert('L')
        
        #image_np = np.array(image)
        #calculate histgoram of image_np
        image_his = np.array(image)
        mask_his = np.array(mask)
        mask_his = np.clip(mask_his, 0, 1)
#         mask_his = np.uint8(mask_his)

        label_his_f = np.array(image_his)
        label_his_f[:,:,0] = image_his[:,:,0] * mask_his
        label_his_f[:,:,1] = image_his[:,:,1] * mask_his
        label_his_f[:,:,2] = image_his[:,:,2] * mask_his
         
        label_his_b = np.array(image_his)
        label_his_b[:,:,0] = image_his[:,:,0] * (1-mask_his)
        label_his_b[:,:,1] = image_his[:,:,1] * (1-mask_his)
        label_his_b[:,:,2] = image_his[:,:,2] * (1-mask_his)
        

        
#         label_his_f = np.reshape(label_his_f, ((label_his_f.shape)[2], (label_his_f.shape)[0], (label_his_f.shape)[1]))
#         label_his_b = np.reshape(label_his_b, ((label_his_b.shape)[2], (label_his_b.shape)[0], (label_his_b.shape)[1]))

        image = self.totensor(image)
        mask = self.totensor(mask)
        sp = self.totensor(sp)
        
#         print('img name:', image_name)
#         print('mask name:', mask_name)
#         print('speudo name:', sp_name)

        if self.unseen_data is not None:
#             hard_sample_name1 = osp.join(self.sample1_dir, self.sample1_list[idx])
#             hard_sample_name2 = osp.join(self.sample2_dir, self.sample2_list[idx])
#             hard_sample_name3 = osp.join(self.sample3_dir, self.sample3_list[idx])
            hard_sample_name1 = osp.join(self.sample1_dir, 'seg_' + self.imglist[idx])
            hard_sample_name2 = osp.join(self.sample2_dir, 'seg_' + self.imglist[idx])
            hard_sample_name3 = osp.join(self.sample3_dir, 'seg_' + self.imglist[idx])

#             print('TN HARD SAMPLE:', hard_sample_name1)

            hard_sample1 = Image.open(hard_sample_name1).convert('L')
            hard_sample2 = Image.open(hard_sample_name2).convert('L')
            hard_sample3 = Image.open(hard_sample_name3).convert('L')

            hard_sample1 = self.totensor(hard_sample1)
            hard_sample2 = self.totensor(hard_sample2)
            hard_sample3 = self.totensor(hard_sample3)

            return image, mask, image_name, len(self.imglist), sp, hard_sample1, hard_sample2, hard_sample3, label_his_b, label_his_f
        else:
            return image, mask, image_name, len(self.imglist), 0, 0, 0, 0, 0
    #

    def __len__(self):
        return len(self.imglist)
