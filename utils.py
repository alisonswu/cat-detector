
from future import division
import cv2
import numpy as np


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







