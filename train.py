
from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models
from torch.autograd import Variable
import torch.optim as optim

import dataset 
import utils
from model import Net, YOLOCriterion


# data directory
data_dir = 'data/VOCdevkit/VOC2012/'
cat_trainval = np.loadtxt(data_dir + 'ImageSets/Main/cat_trainval.txt', dtype='string')

# index of all cat images
img_indexes = cat_trainval[cat_trainval[:,1]=='1',0]

# height, width of resized image 
H,W = 448, 448

# pre-extract ground truth bboxes 
# e.g. gt_bboxes_dict['2008_006294'] = bboxes 
gt_bboxes_dict = dict()
for _, ind in enumerate(img_indexes):
    annotation_name = data_dir + '/Annotations/' + ind + '.xml'
    tree = ET.parse(annotation_name)
    objs = tree.findall('object')
    bboxes = []
    for _, obj in enumerate(objs):
        cls = obj.find('name').text.lower().strip()
        if cls == 'cat':
            bbox = obj.find('bndbox')
            # make pixel indexes 0-based
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            bboxes.append([x1, y1, x2, y2])
    bboxes = np.array(bboxes, dtype=np.uint16)
    gt_bboxes_dict[ind] = bboxes
    
catdata = dataset.CustomDataset(img_indexes, H, W, data_dir, gt_bboxes_dict)
trainloader = data.DataLoader(dataset=catdata, batch_size=2, shuffle=True, num_workers=2)

# use pretrained resnet18 model and reset the last two layers
resnet18 = models.resnet18(pretrained=True)  
pretrained_model = nn.Sequential(*list(resnet18.children())[:-2])
net = Net(pretrained_model)

# set optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0005 )

# check gpu availability 
use_gpu = torch.cuda.is_available()

if use_gpu:
    net.cuda()

net.train(True)

# train the model 
for epoch in range(2):  
    
    running_loss_x = 0.0
    running_loss_y = 0.0
    running_loss_h = 0.0
    running_loss_w = 0.0
    running_loss_c = 0.0
    running_loss = 0.0
    since = time.time()

    for i, batch in enumerate(trainloader):
        
        # get the inputs
        imgs, targets = batch
        inputs = imgs.permute(0, 3, 1, 2)

        # wrap them in Variable
        if use_gpu:
            inputs, targets = Variable(inputs.float().cuda()), Variable(targets.float().cuda())
        else:
            inputs, targets = Variable(inputs.float()), Variable(targets.float())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss_x, loss_y, loss_h, loss_w, loss_c  = YOLOCriterion(outputs, targets, lambda_coord=5, lambda_noobj=0.05)
        loss = loss_x + loss_y + loss_h + loss_w + loss_c 
        loss.backward()
        optimizer.step()

        # update loss
        running_loss_x += loss_x.data[0]
        running_loss_y += loss_y.data[0]
        running_loss_h += loss_h.data[0]
        running_loss_w += loss_w.data[0]
        running_loss_c += loss_c.data[0]
        running_loss += loss.data[0]

    time_elapsed = time.time() - since
    n = len(catdata)
    print('Elapsed: %.2f seconds' % (time_elapsed))
    print('[epoch %3d] x: %.3f, y: %.3f, h: %.3f, w: %.3f, c: %.3f, sum: %.3f'%
                  (epoch + 1, running_loss_x/n, running_loss_y/n, running_loss_h/n, running_loss_w/n, running_loss_c/n, running_loss/n))
        
    running_loss_x = 0.0
    running_loss_y = 0.0
    running_loss_h = 0.0
    running_loss_w = 0.0
    running_loss_c = 0.0
    running_loss = 0.0
    since = time.time()    
    # save model parameters after each epoch
    #torch.save(net.state_dict(), 'parameters_epoch' + epoch + '.pkl')
    
print('Finished Training')














