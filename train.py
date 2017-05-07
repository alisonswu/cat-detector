from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models
from torch.autograd import Variable
import torch.optim as optim
import time
import dataset 
import utils
from model import Net, YOLOCriterion


# load traning data (from PASCAL 2012 train and validation data)
data_dir = 'data/VOCdevkit/VOC2012/' 
cat_trainval = np.loadtxt(data_dir + 'ImageSets/Main/cat_trainval.txt', dtype='string')

# load testing data (From PASCAL 2007 test data)
testdata_dir = 'data/test/VOCdevkit/VOC2007/'
testcat_trainval = np.loadtxt(testdata_dir + 'ImageSets/Main/cat_test.txt', dtype='string')
testimg_indexes = testcat_trainval[testcat_trainval[:,1]=='1',0]

# positive samples include cat class
pos_indexes = utils.imgindex(data_dir,'cat')
# take negative samples from person class
neg_indexes = utils.imgindex(data_dir, 'person')
# filter out cats in negative samples
neg_indexes = np.setdiff1d(neg_indexes, pos_indexes)


# height, width of resized image 
H,W = 448, 448

# pre-extract ground truth bboxes 
# e.g. gt_bboxes_dict['2008_006294'] = bboxes 
gt_bboxes_dict = dict()
for _, ind in enumerate(pos_indexes):
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
    gt_bboxes_dict[ind] = bboxes
    
testgt_bboxes_dict = dict()
for _, ind in enumerate(testimg_indexes):
    annotation_name = testdata_dir + '/Annotations/' + ind + '.xml'
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
    testgt_bboxes_dict[ind] = bboxes

# data loader for training data 
# use a custon sampler to load all positive samples and random sample negative samples
# so that number size of negative sample is 10% of positive sample 
sampler = dataset.CustomSampler(len(pos_indexes),len(neg_indexes),0.1)
train_indexes = np.concatenate([pos_indexes, neg_indexes], axis=0)
catdata = dataset.CustomDataset(train_indexes, H, W, data_dir, gt_bboxes_dict, aug=True)
trainloader = data.DataLoader(dataset=catdata, batch_size=32, sampler=sampler, num_workers=4)

# data loader for testing data 
testdata = dataset.CustomDataset(testimg_indexes, H, W, testdata_dir, testgt_bboxes_dict, aug=False)
testloader = data.DataLoader(dataset=testdata, batch_size=32, shuffle=True, num_workers=4)

# use pretrained resnet34 model and reset the last two fc layers 
resnet34 = models.resnet34(pretrained=True)  
pretrained_model = nn.Sequential(*list(resnet34.children())[:-2])
net = Net(pretrained_model)

# set optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0005 )

# check gpu availability 
use_gpu = torch.cuda.is_available()

if use_gpu:
    net.cuda()

for epoch in range(100):  
    # traning model with test data 
    running_loss_x = 0.0
    running_loss_y = 0.0
    running_loss_h = 0.0
    running_loss_w = 0.0
    running_loss_c_obj = 0.0
    running_loss_c_noobj = 0.0
    running_loss = 0.0

    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    since = time.time()
    net.train() 

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
        # forward 
        outputs = net(inputs)
        # backward 
        loss_x, loss_y, loss_h, loss_w, loss_c_obj, loss_c_noobj  = YOLOCriterion(outputs, targets, 
            lambda_coord=5, lambda_obj=2, lambda_noobj=0.1)
        loss = loss_x + loss_y + loss_h + loss_w + loss_c_obj + loss_c_noobj 
        loss.backward()
        # optimize 
        optimizer.step()
        # update loss
        running_loss_x += loss_x.data[0]
        running_loss_y += loss_y.data[0]
        running_loss_h += loss_h.data[0]
        running_loss_w += loss_w.data[0]
        running_loss_c_obj += loss_c_obj.data[0]
        running_loss_c_noobj += loss_c_noobj.data[0]
        running_loss += loss.data[0]

    time_elapsed = time.time() - since
    
    # print summary for training epoch
    n = len(sampler)
    print('Elapsed: %.2f seconds' % (time_elapsed))
    print('[train epoch %3d] x: %.3f, y: %.3f, h: %.3f, w: %.3f, c_obj: %.3f, c_noobj: %.3f, sum: %.3f'%
                  (epoch + 1, running_loss_x/n, running_loss_y/n, running_loss_h/n, running_loss_w/n, running_loss_c_obj/n, running_loss_c_noobj/n, running_loss/n))
     
    # save model parameters every 50 epoch
    if epoch%50 == 49:
        torch.save(net.state_dict(), 'epoch'+str(epoch+1)+'.pkl')

    # evaluate current model on test data 
    running_loss_x = 0.0
    running_loss_y = 0.0
    running_loss_h = 0.0
    running_loss_w = 0.0
    running_loss_c_obj = 0.0
    running_loss_c_noobj = 0.0
    running_loss = 0.0
   
    net.eval() 
    for i, batch in enumerate(testloader):
        imgs, targets = batch
        inputs = imgs.permute(0, 3, 1, 2)
        inputs, targets = Variable(inputs.float().cuda()), Variable(targets.float().cuda())
        outputs = net(inputs)
        loss_x, loss_y, loss_h, loss_w, loss_c_obj, loss_c_noobj  = YOLOCriterion(outputs, targets, 
           lambda_coord=5, lambda_obj=2, lambda_noobj=0.1)
        running_loss_x += loss_x.data[0]
        running_loss_y += loss_y.data[0]
        running_loss_h += loss_h.data[0]
        running_loss_w += loss_w.data[0]
        running_loss_c_obj += loss_c_obj.data[0]
        running_loss_c_noobj += loss_c_noobj.data[0]
    
    # print summary for testing epoch
    print('[test epoch %3d] x: %.3f, y: %.3f, h: %.3f, w: %.3f, c_obj: %.3f, c_noobj: %.3f'%
                  (epoch + 1, running_loss_x/n, running_loss_y/n, running_loss_h/n, running_loss_w/n, running_loss_c_obj/n, running_loss_c_noobj/n))

print('Finished Training')
