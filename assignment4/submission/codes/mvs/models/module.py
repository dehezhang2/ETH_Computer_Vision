import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO
        self.conv0 = nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(8)
        self.conv1 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(32)

        self.conv8 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        return x

class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO
        self.conv_0 = nn.Conv2d(G, 8, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.deconv_3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.deconv_4 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1) 
        self.conv_5 = nn.Conv2d(8, 1, 3, stride=1, padding=1) 
    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        B,G,D,H,W = x.size()
        x = x.view(B,G,D*H,W)
        conv0 = F.relu(self.conv_0(x))
        conv1 = F.relu(self.conv_1(conv0))
        conv2 = F.relu(self.conv_2(conv1))
        conv3 = self.deconv_3(conv2)
        conv4 = self.deconv_4(conv1 + conv3)
        res = self.conv_5(conv0 + conv4)
        return res.view(B,D,H,W)


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO P_ij = K
        xy1 = torch.stack((x, y, torch.ones_like(x))).repeat(B, 1, 1)   # [B, 3, H*W]
        # calculate R*[x,y,1]
        R_xy1 = torch.matmul(rot, xy1)                            # [B, 3, H*W]
        R_xy1 = R_xy1.unsqueeze(2).repeat(1, 1, D, 1)                   # [B, 3, D, H*W]
        depth = depth_values.repeat(H*W, 1, 1).permute(1, 2, 0)         # [B, D, H*W]
        depth = depth.view(B, 1, D, H*W)                                # [B, 1, D, H*W]
        # calculate R*[dx,dy,d] for each d
        d_R_xy1 = R_xy1 * depth                                         # [B, 3, D, H*W]
        # calculate [lambda * x', lambda * y', lambda] = R*[dx,dy,d] + t
        lambda_xy1 = d_R_xy1 + trans.view(B, 3, 1, 1)                   # [B, 3, D, H*W]

        # check negative depth: 
        negative_mask = lambda_xy1[:, 2:] <= 1e-3
        lambda_xy1[:, 0:1][negative_mask] = float(W)
        lambda_xy1[:, 1:2][negative_mask] = float(H)
        lambda_xy1[:, 2:3][negative_mask] = 1.0

        # divide by the last dim and normalize
        xy = lambda_xy1[:, :2, :, :] / lambda_xy1[:, 2:3, :, :]         # [B, 2, D, H*W]
        x_normalized = xy[:, 0, :, :] / ((W - 1) / 2) - 1               # [B, D, H*W]
        y_normalized = xy[:, 1, :, :] / ((H - 1) / 2) - 1
        
        
        proj_xy = torch.stack((x_normalized, y_normalized), dim=3)      # [B, D, H*W, 2]
        grid = proj_xy.view(B, D*H, W, 2)                               # [B, D*H, W, 2]

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    warped_src_fea = F.grid_sample(
        src_fea,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(B, C, D, H, W)                                               # [B, C, D*H, W]
    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    B,C,D,H,W = warped_src_fea.size()
    # [B, G, C/G, D, H, W]
    warped_src_fea_group = warped_src_fea.view(B, G, C//G, D, H, W)
    # [B, G, C/G, 1, H, W]
    ref_fea_group = ref_fea.view(B, G, C//G, 1, H, W)
    # [B, G, D, H, W]
    out = torch.mean(warped_src_fea_group * ref_fea_group, dim=2)
    return out

def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # out: [B,1,H,W]
    # TODO
    B,D,H,W = p.size()
    depth = depth_values[0].view(D, 1, 1)
    expectation = torch.sum(p * depth, dim=1)
    return expectation

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    valid_mask = mask > 0.5
    depth_est_valid = depth_est[valid_mask]
    depth_gt_valid = depth_gt[valid_mask]
    loss = F.l1_loss(depth_est_valid, depth_gt_valid)
    return loss
    
