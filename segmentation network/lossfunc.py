import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

class FocalWBCEloss(nn.Module):
    """
    Focal_Loss= -1*alpha*(1-pt)**gamma*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`

    alpha

    """

    def __init__(self, weight=1, gamma=2, reduction='mean', **kwargs):
        super(FocalWBCEloss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)

        #loss = - self.weight * target * torch.log(output) - (1.0 - target) * torch.log(1.0 - output)

        loss = - self.weight * (1 - output) ** self.gamma * target * torch.log(output) - \
               output ** self.gamma * (1 - target) * torch.log(1 - output)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss


class WBCE(nn.Module):
    """
    Weighted Binary Cross Entropy.
    `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1.
    To decrease the number of false positives, set β<1.
    Args:
            @param weight: 前景的权重
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, weight=1, ignore_index=None, reduction='mean'):
        super(WBCE, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        weight = float(weight)
        self.weight = weight
        self.reduction = reduction
        self.smooth = 0.01

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # avoid `nan` loss

        # soft label
        #target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

        # loss = self.bce(output, target)
        loss = - self.weight* target*torch.log(output) - (1.0-target)*torch.log(1.0- output)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss
