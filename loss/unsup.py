import torch
import torch.nn.functional as F

from misc.chamfer_distance import ChamferDistance

def chamfer_mask(chamfer_dist, x, y_hat0, thres_static=0.2, thres_dist=0.1):
    # Use the source frame, matching FlowBasedPointFiltering/ConvertToRefinedMMWavePointCloud.
    dists_x, dists_y = chamfer_dist(x[:, 0].to(torch.float), y_hat0[:, 0].to(torch.float))
    del dists_x

    mask_dist_pos = (dists_y < thres_dist).unsqueeze(1).unsqueeze(-1).detach()  # B 1 J 1
    mask_dist_neg = (dists_y > thres_static).unsqueeze(1).unsqueeze(-1).detach()  # B 1 J 1
    return mask_dist_pos, mask_dist_neg

class UnsupLoss(torch.nn.Module):
    def __init__(self, thres_static=0.2, thres_dynamic=None, thres_dist=None, thres_loss=0.05):
        super().__init__()
        self.thres_static = thres_static
        # Preserve the old config key while using the dgmmwave naming internally.
        self.thres_dist = thres_dist if thres_dist is not None else thres_dynamic
        if self.thres_dist is None:
            self.thres_dist = 0.1
        self.thres_loss = thres_loss
        self.chamfer_dist = ChamferDistance()

    def forward(self, x, y_hat0, y_hat1):
        # x: B T N C
        # y_hat0: B 1 J 3
        # y_hat1: B 1 J 3

        mask_dist_pos, mask_dist_neg = chamfer_mask(
            self.chamfer_dist,
            x[:, :1, :, :3],
            y_hat0,
            self.thres_static,
            self.thres_dist,
        )
        
        displacement = y_hat1 - y_hat0

        dynamic_disp = displacement[mask_dist_pos.expand_as(displacement)].view(-1, displacement.shape[-1])
        if dynamic_disp.numel() == 0:
            loss_dynamic = displacement.new_zeros(())
        else:
            dynamic_norm = torch.norm(dynamic_disp, p=2, dim=-1)
            loss_dynamic = torch.relu(self.thres_loss - dynamic_norm).mean()
        
        static_disp = displacement[mask_dist_neg.expand_as(displacement)].view(-1, displacement.shape[-1])
        if static_disp.numel() == 0:
            loss_static = displacement.new_zeros(())
        else:
            loss_static = F.mse_loss(static_disp, torch.zeros_like(static_disp))

        return loss_dynamic, loss_static
