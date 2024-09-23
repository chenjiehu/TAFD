import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weights = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        nclass = predict.shape[1]
        target = torch.nn.functional.one_hot(target.long(), nclass)#[1, 4]->[1, 4, 5]
        target = torch.transpose(torch.transpose(target, 1, 3), 2, 3)
        # target = torch.transpose(target, 1, 2)

        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                view1 = predict[:, i]
                view2 = target[:, i]
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weights is not None:
                    assert self.weights.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1] if self.weights is None else total_loss/(torch.sum(self.weights))


class DiceLoss_gai(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss_gai, self).__init__()
        self.kwargs = kwargs
        self.weights = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        nclass = predict.shape[1]
        target_unique = torch.unique(target)
        target_clone = target.clone()
        index_ignore = torch.where(target == self.ignore_index)
        target_clone[index_ignore] = 0
        target_clone = torch.nn.functional.one_hot(target_clone.long(), nclass)#[1, 4]->[1, 4, 5]
        target_clone[index_ignore[0],index_ignore[1], index_ignore[2],0] = 0
        target_clone = torch.transpose(torch.transpose(target_clone, 1, 3), 2, 3)

        # target = torch.transpose(target, 1, 2)

        assert predict.shape == target_clone.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target_clone.shape[1]):
            if i != self.ignore_index:
                view1 = predict[:, i]
                view2 = target_clone[:, i]
                dice_loss = dice(predict[:, i], target_clone[:, i])
                if self.weights is not None:
                    assert self.weights.shape[0] == target_clone.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target_clone.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target_clone.shape[1] if self.weights is None else total_loss/(torch.sum(self.weights))


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        bs = targets.size(0)
        smooth = 0.01

        probs = F.sigmoid(logits)
        m1 = probs.view(bs, -1)
        m2 = targets.view(bs, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / bs
        return score
