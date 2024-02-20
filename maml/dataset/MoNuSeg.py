import os
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
import maml.dataset.RandomCrop as RandomCrop

class MoNuSeg(Dataset):
    name = 'MoNuSeg'
    out_channels = 1

    def __init__(self, transform = None):
        super(MoNuSeg, self).__init__()

        # MoNuSeg_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/path/to/data/MoNuSeg/'
        MoNuSeg_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/patches/MoNuSeg/'
        self.imgdir = osp.join(MoNuSeg_dir, 'images')
        self.maskdir = osp.join(MoNuSeg_dir, 'labels')

        self.imglist = sorted(os.listdir(self.imgdir))
        self.masklist = sorted(os.listdir(self.maskdir))

        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.resize = transforms.Resize((256, 256))
        # self.radomcrop = RandomCrop.RandomCrop(256)

    def __getitem__(self, idx):  # return one class
        # print('idx:', idx)
        image_name = osp.join(self.imgdir, self.imglist[idx])
        mask_name = osp.join(self.maskdir, self.masklist[idx])

        image = Image.open(image_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        image = self.totensor(image)
        mask = self.totensor(mask)
        # image, mask = self.radomcrop(image, mask)

        # print('Mo image shape:', image.shape)
        # print('Mo mask shape:', mask.shape)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        # mask.unsqueeze_(0)

        # print('MO image shape:', image.shape)
        # print('MO mask shape:', mask.shape)     # 3 * H * W, 1 * H * W



        return image, mask, image_name, len(self.imglist)

    def __len__(self):
        return len(self.imglist)
