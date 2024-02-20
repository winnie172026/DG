import os
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
import maml.dataset.RandomCrop as RandomCrop

class TNBC(Dataset):
    name = 'TNBC'
    out_channels = 1

    def __init__(self, transform = None):
        super(TNBC, self).__init__()

        #TNBC_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/path/to/data/TNBC/'
        TNBC_dir = '/home/kunzixie/Medical_Image_Analysis/pytorch-maml/patches/TNBC/'
        # self.root = os.path.expanduser(root)

        self.imgdir = osp.join(TNBC_dir, 'images')
        self.maskdir = osp.join(TNBC_dir, 'labels')

        self.imglist = sorted(os.listdir(self.imgdir))
        self.masklist = sorted([item.replace('.png', '_label.png') for item in self.imglist])

        self.transform = transform
        self.totensor = transforms.ToTensor()
        # self.resize = transforms.Resize((256, 256))
        # self.randomcrop = RandomCrop.RandomCrop(256)

    def __getitem__(self, idx):  # return one class
        # print('idx:', idx)
        # len_idx = len(idx)

        image_name = osp.join(self.imgdir, self.imglist[idx])
        mask_name = osp.join(self.maskdir, self.imglist[idx].replace(".png", "_label.png"))

        image = Image.open(image_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        image = self.totensor(image)
        mask = self.totensor(mask)
        # image, mask = self.randomcrop(image, mask)

        # print('TNBC image shape:', image.shape)
        # print('TNBC mask shape:', mask.shape)     # 3 * H * W, 1 * H * W

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        # mask.unsqueeze_(0)

        return image, mask, image_name, len(self.imglist)


    def __len__(self):
        return len(self.imglist)
