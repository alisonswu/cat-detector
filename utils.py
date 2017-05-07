
from __future__ import division
import cv2
import numpy as np
import random
from scipy import misc
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt



def imgindex(data_dir, cls):
    txtpath = data_dir + 'ImageSets/Main/' + cls + '_trainval.txt'
    indexes = np.loadtxt(txtpath, dtype='string')
    return indexes[indexes[:,1]=='1',0]


def tensor2bboxes(tensor, H = 448, W = 448, cellsize = 64):
    bboxes_new = []
    # find all cells with confidence 1 
    cells = np.array(np.where(tensor[:,:,4] == 1))
    for col in range(cells.shape[1]):
        # cell position
        i,j =  cells[:,col]
        x,y,h,w = tensor[i,j,:4]
        # enter of bounding box 
        cx = (j + x)* cellsize
        cy = (i + y)* cellsize
        # width, height of bounding box 
        bb_w = w*W
        bb_h = h*H
        # bounding box corner location 
        bbox = [cx - bb_w/2, cy - bb_h/2, cx + bb_w/2, cy + bb_h/2 ]
        bboxes_new.append(bbox)
    bboxes_new = np.array(bboxes_new, dtype=np.uint16)
    return bboxes_new


def bboxes2tensor(bboxes, H=448, W=448, cellsize=64):
    target = np.zeros([int(H/cellsize),int(W/cellsize),5])
    for _, bbox in enumerate(bboxes):
        # find the center of object
        cx = (bbox[0]+bbox[2])/2
        cy = (bbox[1]+bbox[3])/2
        # the center falls in (i,j) cell
        i = int(cy/cellsize)
        j = int(cx/cellsize)
        # set confidence 
        target[i,j,4] = 1
        # set center location adjustment to the grid cell, scaled by cellsize
        target[i,j,0] = (cx - j*cellsize)/cellsize 
        target[i,j,1] = (cy - i*cellsize)/cellsize
        # set height,width scaled by H,W
        target[i,j,2] = (bbox[3] - bbox[1])/H
        target[i,j,3] = (bbox[2] - bbox[0])/W
    return target


def img_aug(img, bboxes, max_scale = 0.2, max_saturation =0.05, max_exposure = 0.05):
    h, w, _ = img.shape

    # 1. zoom and crop images 
    if max_scale>0:
        scale = np.random.uniform(1.0, (1 + max_scale)**0.5)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        img = misc.imresize(img, (h_new,w_new,-1))
        # random crop scaled image to original size 
        crop_x = random.choice(range((w_new-w+1)))
        crop_y = random.choice(range((h_new-h+1)))
        img = img[crop_y : (crop_y + h), crop_x : (crop_x + w)]
        # adjust bounding boxes 
        if bboxes.size:
            bboxes = bboxes*scale 
            bboxes[:,[0,2]] -= crop_x
            bboxes[:,[1,3]] -= crop_y
            bboxes = bboxes.astype(int)
            # threshold bounding boxes to be inside image 
            bboxes[:,][bboxes[:,]<0] = 0
            bboxes[:,0][bboxes[:,0]>w-1] = w-1
            bboxes[:,2][bboxes[:,2]>w-1] = w-1
            bboxes[:,1][bboxes[:,1]>h-1] = h-1 
            bboxes[:,3][bboxes[:,3]>h-1] = h-1 
            # delete bounding boxes outside of the cropped image
            bboxes = bboxes[bboxes[:,0] != bboxes[:,2],:]
            bboxes = bboxes[bboxes[:,1] != bboxes[:,3],:]

    # 2. random horizontal flip 
    flip = random.choice([0,1])
    if flip: 
        img = np.fliplr(img)
        if bboxes.size:
            bboxes[:,[2,0]] = w-1 - bboxes[:,[0,2]]  

    # 3. scale image value to [0,1]
    img = img/255.0

    # 4. adjust saturation
    #if max_saturation>0:
    #    scale1 = np.random.uniform(1-max_saturation,1+max_saturation,3)/(1+max_saturation)
    #    img = img * scale1

    # 5. adjust exposure 
    #if max_exposure>0:
    #    scale2 = np.random.uniform(1-max_exposure,1+max_exposure)
    #    img = np.power(img, scale2)

    # 6. normalizer from pre-trained model on ImageNet
    img -= np.array([0.485, 0.456, 0.406])
    img /= np.array([0.229, 0.224, 0.225])

    return img, bboxes


def unionbox(bboxes):
    x1 = np.amin(bboxes[:,0])
    x2 = np.amax(bboxes[:,2])
    y1 = np.amin(bboxes[:,1])
    y2 = np.amax(bboxes[:,3])
    return np.array([x1,y1,x2,y2])


def NMS(bboxes, overlap_threshold=0.5):
    if len(bboxes)==0:
        return []
    areas = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
    idxs = np.argsort(areas)
    selected_idx = []
    while len(idxs)>0:
        selected_idx.append(idxs[-1])
        x1max = np.maximum(bboxes[idxs[-1]][0], bboxes[idxs[0:-1]][:,0])
        x2min = np.minimum(bboxes[idxs[-1]][2], bboxes[idxs[0:-1]][:,2])
        y1max = np.maximum(bboxes[idxs[-1]][1], bboxes[idxs[0:-1]][:,1])
        y2min = np.minimum(bboxes[idxs[-1]][3], bboxes[idxs[0:-1]][:,3])  
        overlaps = np.maximum(x2min-x1max, 0)*np.maximum(y2min-y1max, 0)/areas[idxs[-1]]
        overlap_idx = np.where(overlaps > overlap_threshold)[0]
        selected_bboxes = bboxes[np.concatenate((np.array([idxs[-1]]),idxs[overlap_idx]))]
        bboxes[idxs[-1]] = unionbox(selected_bboxes)
        idxs = np.delete(idxs, np.concatenate(([len(idxs)-1],overlap_idx)))
        
    return bboxes[selected_idx].astype(np.uint16)


# plot resized img and bounding boxes, press any key to exit the image window
def view_cat(img, bboxes, H = 448, W = 448, cellsize = 64):
    cv2img = img[:,:,::-1].copy()
    col = (225,225,0)

    # 1. add grid lines 
    for i in range(0,W,cellsize):
        cv2.line(cv2img, (i,0),(i,H), col1)     
    for j in range(0,H,cellsize):
        cv2.line(cv2img, (0,j),(W,j), col1)

    for num, bbox in enumerate(bboxes): 
        # 2. add bounding boxes 
        cv2.rectangle(cv2img,(bbox[0], bbox[1]), (bbox[2], bbox[3]), col)
        # 3. add diagonal line to bounding boxes
        cv2.line(cv2img, (bbox[0],bbox[1]),(bbox[2], bbox[3]), col)
        cv2.line(cv2img, (bbox[0],bbox[3]),(bbox[2], bbox[1]), col)
        # 4. add anotation of bounding box index 
        cv2.putText(cv2img, "{}".format(num + 1), (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)     
          
    # display image 
    cv2.imshow('example',cv2img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def detect(image_name, net, c_threshold=0.2, overlap_threshold=0.5):
    ori_img = misc.imread(image_name)
    ori_H, ori_W, _ = ori_img.shape
    hzoom = float(ori_H)/448
    wzoom = float(ori_W)/448
    img = misc.imresize(ori_img, (448,448,3))
    img = img/255.0
    img -= np.array([0.485, 0.456, 0.406])
    img /= np.array([0.229, 0.224, 0.225])
    img = torch.FloatTensor(img).unsqueeze(0).permute(0,3,1,2)
    target = net(Variable(img)).data.numpy().squeeze()
    bboxes = tensor2bboxes(target, c_threshold, hzoom, wzoom)
    # print 'Num of bboxes: %i' % len(bboxes)
    nmsbboxes = NMS(bboxes, overlap_threshold)
    # print 'Num of bboxes after NMS: %i' % len(nmsbboxes)
    cv2img = ori_img.copy()
    for bbox in nmsbboxes:
        cv2.rectangle(cv2img,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (225,0,0), int(ori_H/200))
    plt.imshow(cv2img)
    return nmsbboxes



