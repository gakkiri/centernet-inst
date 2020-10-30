import torch
from torch import nn
import torch.nn.functional as F
from .centernet_decode import gather_feature

__all__ = ['reg_l1_loss', 'modified_focal_loss', 'DiceLoss']


def reg_l1_loss(output, mask, index, target):
    pred = gather_feature(output, index, use_transform=True)
    mask = mask.unsqueeze(dim=2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss


def modified_focal_loss(pred, gt):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def dice_loss(input, target):
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / ((iflat * iflat).sum() + (tflat * tflat).sum() + smooth))


class DiceLoss(nn.Module):
    def __init__(self, feat_channel):
        super(DiceLoss, self).__init__()
        self.feat_channel = feat_channel

    def forward(self, seg_feat, conv_weight, mask, ind, target):
        mask_loss = 0.
        batch_size = seg_feat.size(0)
        weight = gather_feature(conv_weight, ind, use_transform=True)
        h, w = seg_feat.size(-2), seg_feat.size(-1)
        x, y = ind % w, ind / w
        x_range = torch.arange(w).float().to(device=seg_feat.device)
        y_range = torch.arange(h).float().to(device=seg_feat.device)
        y_grid, x_grid = torch.meshgrid([y_range, x_range])
        valid_batch_size = batch_size
        for i in range(batch_size):
            num_obj = target[i].size(0)
            if num_obj == 0:
                valid_batch_size -= 1
                continue
            conv1w, conv1b, conv2w, conv2b, conv3w, conv3b = \
                torch.split(weight[i, :num_obj], [(self.feat_channel + 2) * self.feat_channel, self.feat_channel,
                                                  self.feat_channel ** 2, self.feat_channel,
                                                  self.feat_channel, 1], dim=-1)
            y_rel_coord = (y_grid[None, None] - y[i, :num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()) / h # 128.
            x_rel_coord = (x_grid[None, None] - x[i, :num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()) / w # 128.
            feat = seg_feat[i][None].repeat([num_obj, 1, 1, 1])
            feat = torch.cat([feat, x_rel_coord, y_rel_coord], dim=1).view(1, -1, h, w)

            conv1w = conv1w.contiguous().view(-1, self.feat_channel + 2, 1, 1)
            conv1b = conv1b.contiguous().flatten()
            feat = F.conv2d(feat, conv1w, conv1b, groups=num_obj).relu()

            conv2w = conv2w.contiguous().view(-1, self.feat_channel, 1, 1)
            conv2b = conv2b.contiguous().flatten()
            feat = F.conv2d(feat, conv2w, conv2b, groups=num_obj).relu()

            conv3w = conv3w.contiguous().view(-1, self.feat_channel, 1, 1)
            conv3b = conv3b.contiguous().flatten()
            feat = F.conv2d(feat, conv3w, conv3b, groups=num_obj).sigmoid().squeeze()
            
            true_mask = mask[i, :num_obj, None, None].float()
            mask_loss += dice_loss(feat * true_mask, target[i] * true_mask)

        return mask_loss / valid_batch_size
