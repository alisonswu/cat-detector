from __future__ import division
import numpy as np
from scipy import misc
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import utils

class CustomDataset(data.Dataset):
    def __init__(self, img_indexes, H, W, data_dir, gt_bboxes_dict, aug):
        self.img_indexes = img_indexes
        self.H = H
        self.W = W
        self.data_dir = data_dir
        self.gt_bboxes_dict = gt_bboxes_dict
        self.aug = aug
    def __getitem__(self, index):
        # 1. Read one image,bboxes from file 
        ind = self.img_indexes[index]
        image_name = self.data_dir + '/JPEGImages/' + ind + '.jpg'       
        ori_img = misc.imread(image_name)
        gt_bboxes = self.gt_bboxes_dict.get(ind, []) 
        
        # 2. Resize image,bboxes to H,W
        ori_H, ori_W, _ = ori_img.shape
        img = misc.imresize(ori_img, (self.H,self.W,3))
        xzoom = float(self.W)/ori_W
        yzoom = float(self.H)/ori_H
        bboxes = np.array([[int(bbox[0]*xzoom), int(bbox[1]*yzoom), 
                            int(bbox[2]*xzoom), int(bbox[3]*yzoom)]
                  for bbox in gt_bboxes])
        
        # 3. image augmentation and transformation 
        if self.aug:
            img, bboxes = utils.img_aug(img,bboxes)
        else:
            img = img/255.0
            img -= np.array([0.485, 0.456, 0.406])
            img /= np.array([0.229, 0.224, 0.225])
        
        # 4. convert bboxes to target tensor
        target = utils.bboxes2tensor(bboxes, self.H, self.W)
        
        return img, target
    def __len__(self):
        return len(self.img_indexes)


class CustomSampler(Sampler):
    def __init__(self, nb_pos, nb_neg, neginpos_ratio):
        self.nb_pos = nb_pos
        self.nb_neg = nb_neg
        self.neginpos_ratio = neginpos_ratio
        self.nb_neg_sampled = int(self.nb_pos*self.neginpos_ratio)


    def __iter__(self):
        pos = np.arange(self.nb_pos)
        neg = np.random.choice(np.arange(self.nb_pos, self.nb_pos + self.nb_neg),
                               self.nb_neg_sampled,
                               replace=False)
        return iter(np.random.permutation(np.concatenate([pos, neg], axis=0)))

    def __len__(self):
        return self.nb_pos + self.nb_neg_sampled
