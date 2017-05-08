import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

class Net(nn.Module):
    def __init__(self, pretrained_model):
        super(Net, self).__init__()
        self.pretrained_model = pretrained_model
        self.conv1= nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1) 
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU()
        self.conv2= nn.Conv2d(1024, 5, kernel_size=5, stride=1, padding=2, bias=False)
        self.softsign = nn.Softsign()
        
    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.permute(0,2,3,1)
        x = (self.softsign(x) + 1)*0.5
        return x


def YOLOCriterion(output, target, lambda_coord=5, lambda_obj=1, lambda_noobj=0.5):
    loss_x = torch.mul( target[:,:,:,4], (target[:,:,:,0] - output[:,:,:,0]).pow(2)).sum()
    loss_y = torch.mul( target[:,:,:,4], (target[:,:,:,1] - output[:,:,:,1]).pow(2)).sum()
    loss_h = lambda_coord * torch.mul( target[:,:,:,4], (target[:,:,:,2].pow(0.5) - output[:,:,:,2].pow(0.5)).pow(2)).sum()
    loss_w = lambda_coord * torch.mul( target[:,:,:,4], (target[:,:,:,3].pow(0.5) - output[:,:,:,3].pow(0.5)).pow(2)).sum()
    loss_c = (target[:,:,:,4]- output[:,:,:,4]).pow(2)
    loss_c_obj = lambda_obj*(target[:,:,:,4]*loss_c).sum()
    loss_c_noobj = lambda_noobj*((1-target[:,:,:,4])*loss_c).sum()
    return loss_x, loss_y, loss_h, loss_w, loss_c_obj, loss_c_noobj


resnet34 = models.resnet34(pretrained=False)
pretrained_model = nn.Sequential(*list(resnet34.children())[:-2])
catdetector = Net(pretrained_model)
