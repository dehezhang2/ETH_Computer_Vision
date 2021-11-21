import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # number of groups for group-wise correlation
        self.G = 8
        self.feature = FeatureNet()
        self.similarity_regularization = SimlarityRegNet(self.G)


    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        D = depth_values.size(1)
        V = len(imgs)

        # feature extraction
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]

        # do the warping, compute and integrate matching similarity across source views
        B,C,H,W = ref_feature.size()
        similarity_sum = torch.zeros((B, self.G, D, H, W), dtype=torch.float32, device=ref_feature.device)

        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped src feature
            warped_src_feature = warping(src_fea, src_proj, ref_proj, depth_values)
            # group-wise correlation
            similarity = group_wise_correlation(ref_feature, warped_src_feature, self.G)
            similarity_sum = similarity_sum + similarity

        # aggregate matching similarity from all the source views by averaging
        similarity_sum = similarity_sum.div_(V)

        # regularization
        similarity_reg = self.similarity_regularization(similarity_sum)
        prob_volume = F.softmax(similarity_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), 
                                            pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(D, 
                                            device=prob_volume.device, dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        return {
            "depth": depth, 
            "photometric_confidence": photometric_confidence
        }
    