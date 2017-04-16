from __future__ import division
import numpy as np
from scipy import misc
import torch.utils.data as data
import utils

class CustomDataset(data.Dataset):
    def __init__(self, img_indexes, H, W, data_dir, gt_bboxes_dict):
        self.img_indexes = img_indexes
        self.H = H
        self.W = W
        self.data_dir = data_dir
        self.gt_bboxes_dict = gt_bboxes_dict
    def __getitem__(self, index):
        # 1. Read one image,bboxes from file 
        ind = self.img_indexes[index]
        image_name = self.data_dir + '/JPEGImages/' + ind + '.jpg'       
        ori_img = misc.imread(image_name)
        gt_bboxes = self.gt_bboxes_dict[ind] 
        # 2. Resize image,bboxes to H,W
        ori_H, ori_W, _ = ori_img.shape
        img = misc.imresize(ori_img, (self.H,self.W,3))
        xzoom = float(self.W)/ori_W
        yzoom = float(self.H)/ori_H
        bboxes = np.array([[int(bbox[0]*xzoom), int(bbox[1]*yzoom), 
                            int(bbox[2]*xzoom), int(bbox[3]*yzoom)]
                  for bbox in gt_bboxes])
        # 3. convert bboxes to target tensor
        target = utils.bboxes2tensor(bboxes, self.H, self.W)
        # 4. Return image tensor, target tensor 
        return img, target
    def __len__(self):
        return len(self.img_indexes)

