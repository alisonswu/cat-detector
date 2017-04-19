
from future import division
import cv2
import numpy as np
import random


#-----------------------------------------------------
#     Given resized img and bounding boxes, plot it.
#     Press any key to exit the image window
# ----------------------------------------------------
def view_cat(img, bboxes, grid = True, H = 448, W = 448, cellsize = 64):
    cv2img = img[:,:,::-1].copy()
    col1 = (225,0,225)
    col2 = (225,225,0)
    # add bounding boxes 
    for _, bbox in enumerate(bboxes): 
        cv2.rectangle(cv2img,(bbox[0], bbox[1]), (bbox[2], bbox[3]), col2)
    # add aditional information if grid option is True 
    if grid:
        # add grid lines
        for i in range(0,W,cellsize):
            cv2.line(cv2img, (i,0),(i,H), col1)     
        for j in range(0,H,cellsize):
            cv2.line(cv2img, (0,j),(W,j), col1)
        for num, bbox in enumerate(bboxes):
            # print information used to create tensor 
            print "-----------bounding box", num+1, "----------------"
            print bbox
            cx = (bbox[0]+bbox[2])/2
            cy = (bbox[1]+bbox[3])/2
            i = int(cy/cellsize)
            j = int(cx/cellsize)
            x = (cx - j*cellsize)/cellsize 
            y = (cy - i*cellsize)/cellsize
            h = (bbox[3] - bbox[1])/H
            w = (bbox[2] - bbox[0])/W
            print "i =", i, " j = ", j
            print "x =", x, " y = ", y
            print "h =", h, " w = ", w
            # add diagonal line to bounding box
            cv2.line(cv2img, (bbox[0],bbox[1]),(bbox[2], bbox[3]), col2)
            cv2.line(cv2img, (bbox[0],bbox[3]),(bbox[2], bbox[1]), col2)
            # add annotation of bounding box 
            cv2.putText(cv2img, "{}".format(num + 1), (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col2, 2)
    # display image 
    cv2.imshow('example',cv2img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

#------------------------------------------------
#   convert tensor back to array of bounding boxes
#------------------------------------------------
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


def img_aug(img, bboxes, max_scale = 0.2, max_saturation =0.1, max_exposure = 0.5):
    h, w, _ = img.shape
    # scale up image 
    scale = np.random.uniform(1.0, (1 + max_scale)**0.5)
    h_new, w_new = int(round(h*scale)), int(round(w*scale))
    img = misc.imresize(img, (h_new,w_new,-1))
    # random crop scaled image to original size 
    crop_x = random.choice(range((w_new-w+1)))
    crop_y = random.choice(range((h_new-h+1)))
    img = img[crop_y : (crop_y + h), crop_x : (crop_x + w)]
    # adjust bounding boxes 
    bboxes = bboxes*scale 
    bboxes[:,[0,2]] -= crop_x
    bboxes[:,[1,3]] -= crop_y
    bboxes = bboxes.astype(int)
    # threshold bounding boxes to be inside image 
    bboxes[:,0][bboxes[:,0]<0] = 0
    bboxes[:,1][bboxes[:,1]<0] = 0
    bboxes[:,2][bboxes[:,2]>w-1] = w-1
    bboxes[:,3][bboxes[:,3]>h-1] = h-1    
    # random horizontal flip 
    flip = random.choice([0,1])
    if flip: 
        img = np.fliplr(img)
        bboxes[:,[0,2]] = w-1 - bboxes[:,[0,2]]  
    # scale image value to [0,1]
    img = img/255.0
    # adjust saturation
    scale1 = np.random.uniform(1-max_saturation,1+max_saturation,3)/(1+max_saturation )
    img = img * scale1
    # adjust exposure 
    scale2 = np.random.uniform(1-max_exposure,1+max_exposure)
    img = np.power(img, scale2)
    # adjust image value to [0,225]
    img = np.array(img * 255., np.uint8)
    return img, bboxes


