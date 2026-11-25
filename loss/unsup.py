import torch

def chamfer_mask(x, y_hat, thres_static=0.2, thres_dist=0.05):
    # Thresholds are in meters. The old CUDA Chamfer operator returned squared
    # distances, and compared the first input frame to the last predicted pose.
    with torch.no_grad():
        dists_y = torch.cdist(y_hat[:, 0].float(), x[:, -1, :, :3].float()).amin(dim=-1)

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

    def forward(self, x, y_hat0, y_hat1):
        # x: B T N C
        # y_hat0: B 1 J 3
        # y_hat1: B 1 J 3

        mask_dist_pos, mask_dist_neg = chamfer_mask(
            x,
            y_hat1,
            self.thres_static,
            self.thres_dist,
        )
        
        displacement = y_hat1 - y_hat0

        dynamic_disp = displacement[mask_dist_pos.expand_as(displacement)].view(-1, displacement.shape[-1])
        if dynamic_disp.numel() == 0:
            loss_dynamic = displacement.sum() * 0
        else:
            dynamic_norm = torch.norm(dynamic_disp, p=2, dim=-1)
            loss_dynamic = torch.relu(self.thres_loss - dynamic_norm).mean()
        
        static_disp = displacement[mask_dist_neg.expand_as(displacement)].view(-1, displacement.shape[-1])
        if static_disp.numel() == 0:
            loss_static = displacement.sum() * 0
        else:
            loss_static = torch.norm(static_disp, p=2, dim=-1).mean()

        return loss_dynamic, loss_static
