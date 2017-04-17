
import torch
from torch.autograd import Variable

#------------------------------------------------------
# loss function
# output and target must be variables
#---------------------------------------------------- 
def loss(output, target, lambda_coord, lambda_noobj):
    loss_x = lambda_coord * torch.mul( target[:,:,4], (target[:,:,0] - output[:,:,0]).pow(2)).sum()
    loss_y = lambda_coord * torch.mul( target[:,:,4], (target[:,:,1] - output[:,:,1]).pow(2)).sum()
    loss_h = lambda_coord * torch.mul( target[:,:,4], (target[:,:,2].pow(0.5) - output[:,:,2].pow(0.5)).pow(2)).sum()
    loss_w = lambda_coord * torch.mul( target[:,:,4], (target[:,:,3].pow(0.5) - output[:,:,3].pow(0.5)).pow(2)).sum()
    loss_c = lambda_noobj * (target[:,:,4]- output[:,:,4]).pow(2).sum() + (1-lambda_noobj) * torch.mul(target[:,:,4], (target[:,:,4] - output[:,:,4]).pow(2)).sum()
    return loss_x + loss_y + loss_h +loss_w + loss_c 

