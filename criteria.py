import torch
import torch.nn as nn
import kornia
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp

class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()   
        return self.loss

class MaskediRMSELoss(nn.Module):
    def __init__(self):
        super(MaskediRMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        mtarget = target[valid_mask]
        # print(mtarget.min())
        mpred = pred[valid_mask]
        mpred = mpred
        # print(mpred.min())
        diff = 1.0/mtarget - 1.0/mpred
        self.loss = torch.sqrt((diff**2).mean())
        return self.loss



class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = (target - pred).abs()
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class LogMaskedL1loss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, pred, gt):
        e = torch.abs(torch.tensor(torch.finfo(torch.float32).eps))
        mask = (gt > 0).detach()
        diff = torch.abs(torch.log(gt[mask]+e) - torch.log(pred[mask]+e))
        return diff.mean()


class MaskediMAELoss(nn.Module):
    def __init__(self):
        super(MaskediMAELoss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        mtarget = target[valid_mask]
        mpred = pred[valid_mask]
        mpred = mpred
        diff = 1.0/mtarget - 1.0/mpred
        self.loss = diff.abs().mean()
        return self.loss

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = kornia.filters.Sobel()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        pred_edge = self.sobel(pred)
        target_edge = self.sobel(target)
        diff = target_edge - pred_edge
        self.loss = diff.abs().mean()
        return self.loss

class MixedLoss(nn.Module):
    def __init__(self, lmbda=0, lmbda2=0):
        super().__init__()
        self.lmbda = lmbda
        self.lmbda2 = lmbda2
        # self.sobel = SobelEdgeLoss()
        self.mse = MaskedMSELoss()
        self.mae = MaskedL1Loss()

    def forward(self, pred, target):
        # sobel_loss = self.sobel(pred, target)
        rmse_loss = torch.sqrt(self.mse(pred, target))
        mae_loss = self.mae(pred, target)
        # self.loss = self.lmbda * sobel_loss + rmse_loss + self.lmbda2 * mae_loss
        self.loss =  rmse_loss + self.lmbda2 * mae_loss
        return self.loss

class Delta(nn.Module):
    def __init__(self):
        super(Delta, self).__init__()

    def forward(self, pred, target, threshold):
        valid_mask = (target>0).detach()
        mtarget = target[valid_mask]
        mpred = pred[valid_mask]
        delta1, delta2 = mtarget/mpred, mpred/mtarget
        # delta1[torch.isnan(delta1)] = 0
        # delta2[torch.isnan(delta2)] = 0
        delta_map = torch.max(delta1, delta2)
        self.hit_rate = torch.sum(delta_map < threshold).float()/delta_map.numel()
        return self.hit_rate

class Rel(nn.Module):
    def __init__(self):
        super(Rel, self).__init__()

    def forward(self, pred, target):
        valid_mask = (target>0).detach()
        mtarget = target[valid_mask]
        mpred = pred[valid_mask]       
        diff = torch.abs(mtarget - mpred)/mtarget
        self.rel = diff.mean()
        return self.rel


class Getdelta:
    def __init__(self, delta_list=[1.05,1.15,1.25,1.25**2,1.25**3]):
        self.delta_list = delta_list
        self.avg_dict = {}
        self.delta_dict = {}
        self.num_dict = {}
        for delta in self.delta_list:
            self.avg_dict[delta] = AverageMeter()
            self.delta_dict[delta] = Delta()
            self.num_dict[delta] = 0

    def calculate(self, pred, target):
        for delta in self.delta_list:
            self.num_dict[delta] = self.delta_dict[delta](pred, target, delta)

    def update(self):
        for delta in self.delta_list:
            self.avg_dict[delta].update(self.num_dict[delta])

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, eps=1e-5):
        super(SSIM, self).__init__()
        self.eps = eps
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred, gt):

        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        img2[img2 < self.eps] = 0
        img1[img2 < self.eps] = 0

        (_, channel, _, _) = img1.size()


        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)



