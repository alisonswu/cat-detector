import torch
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, pretrained_model):
        super(Net, self).__init__()
        self.pretrained_model = pretrained_model
        self.conv1= nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1) 
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(7*7*1024, 1024)
        self.fc2 = nn.Linear(1024, 7*7*5)
        self.softsign = nn.Softsign()
        
    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1,7,7,5)
        x = (self.softsign(x) + 1)*0.5
        return x



#------------------------------------------------------
# loss function
# output and target must be variables
#---------------------------------------------------- 
def YOLOCriterion(output, target, lambda_coord=5, lambda_noobj=0.05):
    loss_x = lambda_coord * torch.mul( target[:,:,:,4], (target[:,:,:,0] - output[:,:,:,0]).pow(2)).sum()
    loss_y = lambda_coord * torch.mul( target[:,:,:,4], (target[:,:,:,1] - output[:,:,:,1]).pow(2)).sum()
    loss_h = lambda_coord * torch.mul( target[:,:,:,4], (target[:,:,:,2].pow(0.5) - output[:,:,:,2].pow(0.5)).pow(2)).sum()
    loss_w = lambda_coord * torch.mul( target[:,:,:,4], (target[:,:,:,3].pow(0.5) - output[:,:,:,3].pow(0.5)).pow(2)).sum()
    loss_c = ((lambda_noobj + target[:,:,:,4]*(1-lambda_noobj))*(target[:,:,:,4]- output[:,:,:,4]).pow(2)).sum()
    return loss_x, loss_y, loss_h, loss_w, loss_c 
