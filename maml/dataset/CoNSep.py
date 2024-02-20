import os
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
import maml.dataset.RandomCrop as RandomCrop

class CoNSep(Dataset):
    name = 'CoNSep'
    out_channels = 1

    def __init__(self, transform = None):
        super(CoNSep, self).__init__()

        # CoNSep_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/path/to/data/CoNSep/'
        CoNSep_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/patches/CoNSep/'

        self.imgdir = osp.join(CoNSep_dir, 'images')
        self.maskdir = osp.join(CoNSep_dir, 'labels')

        self.imglist = sorted(os.listdir(self.imgdir))
        self.masklist = sorted(os.listdir(self.maskdir))

        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.resize = transforms.Resize((256, 256))
        # self.randomcrop = RandomCrop.RandomCrop(256)

    def __getitem__(self, idx):  # return one class
        # print('idx:', idx)
        # len_idx = len(idx)


        image_name = osp.join(self.imgdir, self.imglist[idx])
        mask_name = osp.join(self.maskdir, self.masklist[idx])

        image = Image.open(image_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        image = self.totensor(image)
        mask = self.totensor(mask)
        # image, ask = self.randomcrop(image, mask)

        # print('Co image shape:', image.shape)
        # print('Co mask shape:', mask.shape)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        # mask.unsqueeze_(0)


        return image, mask, image_name, len(self.imglist)


    def __len__(self):
        return len(self.imglist)
