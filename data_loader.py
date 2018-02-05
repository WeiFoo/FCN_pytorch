import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import pdb
from torch.utils.data import Dataset, DataLoader



def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the src directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

class ADE20KLoader(Dataset):
    def __init__(self, src, n_classes, types="training", is_transform=False,
                 img_size=224):
        self.src = src
        self.types = types
        self.is_transform = is_transform
        self.n_classes = n_classes
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892]) #imgenet mean
        self.data = collections.defaultdict(list)

        for types in ["small", "training", "validation"]:
            file_list = recursive_glob(rootdir=self.src + 'images/' + self.types + '/', suffix='.jpg')
            self.data[types] = file_list

    def __len__(self):
        return len(self.data[self.types])

    def __getitem__(self, index):
        img_path = self.data[self.types][index].rstrip()
        lbl_path = img_path[:-4] + '_seg.png'

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl


    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        lbl = self.encode_segmap(lbl)
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)
        assert(np.all(classes == np.unique(lbl)))

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def encode_segmap(self, mask):
        # Refer : http://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]))
        label_mask = ( mask[:,:,0] / 10.0 ) * 256 + mask[:,:,1]
        return np.array(label_mask, dtype=np.uint8)

    def decode_segmap(self, temp, plot=False):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = 10 * (l%10)
            g[temp == l] = l
            b[temp == l] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = (r/255.0)
        rgb[:, :, 1] = (g/255.0)
        rgb[:, :, 2] = (b/255.0)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

