import torch
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt

from model.metrics import calculate_error
from misc.lr_scheduler import LinearWarmupCosineAnnealingLR
from misc.utils import torch2numpy, import_with_str, exists_and_is_true
from misc.skeleton import ITOPSkeleton
from misc.vis import visualize_sample

def create_model(model_name, model_params):
    if model_params is None:
        model_params = {}
    model_class = import_with_str('model', model_name)
    model = model_class(**model_params)
    return model

def create_loss(loss_name, loss_params):
    if loss_params is None:
        loss_params = {}
    if loss_name == 'UnsupLoss':
        from loss.unsup import UnsupLoss
        loss_class = UnsupLoss
        loss = loss_class(**loss_params)
    else:
        loss_class = import_with_str('torch.nn', loss_name)
        loss = loss_class(**loss_params)
    return loss

def create_optimizer(optim_name, optim_params, mparams):
    if optim_params is None:
        optim_params = {}
    optim_class = import_with_str('torch.optim', optim_name)
    optimizer = optim_class(mparams, **optim_params)
    return optimizer
    
def create_scheduler(sched_name, sched_params, optimizer):
    if sched_params is None:
        sched_params = {}
    if sched_name == 'LinearWarmupCosineAnnealingLR':
        sched_class = LinearWarmupCosineAnnealingLR
    else:
        sched_class = import_with_str('torch.optim.lr_scheduler', sched_name)
    scheduler = sched_class(optimizer, **sched_params)
    return scheduler

class LitModel(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.model = create_model(self.hparams.model_name, self.hparams.model_params)
        self.use_teacher = self.hparams.train_dataset['name'] == 'LiDARAssistedPseudoLabelingDataset'

        if self.use_teacher:
            self.model_teacher = create_model(self.hparams.model_name, self.hparams.model_params)
            if hasattr(hparams, 'teacher_checkpoint_path') and hparams.teacher_checkpoint_path is not None:
                teacher_state_dict = torch.load(hparams.teacher_checkpoint_path, map_location=self.device)['state_dict']
                cleaned_state_dict = {}
                for key, value in teacher_state_dict.items():
                    if key.startswith('model_teacher.'):
                        cleaned_state_dict[key[len('model_teacher.'):]] = value
                    elif key.startswith('model.'):
                        cleaned_state_dict[key[len('model.'):]] = value
                if cleaned_state_dict:
                    self.model_teacher.load_state_dict(cleaned_state_dict, strict=False)

        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'], strict=False)
        self.loss_fn = create_loss(self.hparams.loss_name, self.hparams.loss_params)

    def _recover_data(self, data, centroid, radius):
        data[..., :3] = data[..., :3] * radius.unsqueeze(-2).unsqueeze(-2) + centroid.unsqueeze(-2).unsqueeze(-2)
        return torch2numpy(data)
    
    def _recover_all(self, x, y, y_hat, c, r):
        x = self._recover_data(x.clone().detach(), c, r)
        y = self._recover_data(y.clone().detach(), c, r)
        y_hat = self._recover_data(y_hat.clone().detach(), c, r)
        return x, y, y_hat

    def _calculate_loss(self, batch):
        if self.use_teacher:
            x_sup, y_sup = batch['point_clouds'], batch['keypoints']
            x_unsup = batch['point_clouds_unsup']
            x_lidar, y_lidar = batch['point_clouds_ref'], batch['keypoints_ref']

            yt_hat_sup = self.model_teacher(x_sup)
            y_hat_lidar = self.model(x_lidar)
            y_hat = y_hat_lidar

            if hasattr(self.hparams, 'teacher_checkpoint_path') and self.hparams.teacher_checkpoint_path is not None:
                loss_sup = F.mse_loss(y_hat_lidar, y_lidar)
            else:
                loss_sup = F.mse_loss(yt_hat_sup, y_sup) + F.mse_loss(y_hat_lidar, y_lidar)

            x_unsup0 = x_unsup[:, :-1]
            x_unsup1 = x_unsup[:, 1:]

            yt_hat_unsup0 = self.model_teacher(x_unsup0)
            yt_hat_unsup1 = self.model_teacher(x_unsup1)

            y_hat_unsup0 = self.model(x_unsup0)
            y_hat_unsup1 = self.model(x_unsup1)

            loss_pseudo = F.mse_loss(y_hat_unsup0, yt_hat_unsup0.detach()) + F.mse_loss(y_hat_unsup1, yt_hat_unsup1.detach())
            # Keep the motion priors on teacher pseudo predictions, but do not detach them:
            # this preserves the intended target branch while still allowing gradients to flow.
            loss_dynamic, loss_static = self.loss_fn(x_unsup, yt_hat_unsup0, yt_hat_unsup1)

            loss = loss_sup + \
                   self.hparams.w_pseudo * loss_pseudo + \
                   self.hparams.w_dynamic * loss_dynamic + \
                   self.hparams.w_static * loss_static
            
            loss_dict = {'loss_sup': loss_sup.item(),
                         'loss_pseudo': loss_pseudo.item(), 'loss_dynamic': loss_dynamic.item(), 
                         'loss_static': loss_static.item(), 'loss': loss.item()}

        else:
            x, y = batch['point_clouds'], batch['keypoints']
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            loss_dict = {'loss': loss.item()}

        return loss, loss_dict, y_hat
    
    def _visualize(self, x, y, y_hat):
        sample = x[0][0][:, [0, 2, 1]], y[0][0][:, [0, 2, 1]], y_hat[0][0][:, [0, 2, 1]]
        fig = visualize_sample(sample, edges=ITOPSkeleton.bones, point_size=2, joint_size=25, linewidth=2, padding=0.1)
        tb = self.logger.experiment
        tb.add_figure('val_sample', fig, global_step=self.global_step)
        plt.close(fig)
        plt.clf()

    def training_step(self, batch, batch_idx):
        del batch_idx
        if self.use_teacher:
            x = batch['point_clouds_ref']
            y = batch['keypoints_ref']
            c = batch['centroid_ref']
            r = batch['radius_ref']
        else:
            x = batch['point_clouds']
            y = batch['keypoints']
            c = batch['centroid']
            r = batch['radius']
        loss, loss_dict, y_hat = self._calculate_loss(batch)

        log_dict = {f'train_{k}': v for k, v in loss_dict.items()}
        x_rec, y_rec, y_hat_rec = self._recover_all(x, y, y_hat, c, r)
        mpjpe, pampjpe = calculate_error(y_hat_rec, y_rec)
        log_dict = {**log_dict, 'train_mpjpe': mpjpe, 'train_pampjpe': pampjpe}

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, c, r = batch['point_clouds'], batch['keypoints'], batch['centroid'], batch['radius']
        y_hat = self.model(x)

        x_rec, y_rec, y_hat_rec = self._recover_all(x, y, y_hat, c, r)
        mpjpe, pampjpe = calculate_error(y_hat_rec, y_rec)
        log_dict = {'val_mpjpe': mpjpe, 'val_pampjpe': pampjpe}

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)

        if batch_idx == 10:
            self._visualize(x_rec, y_rec, y_hat_rec)

    def test_step(self, batch, batch_idx):
        x, y, c, r = batch['point_clouds'], batch['keypoints'], batch['centroid'], batch['radius']
        y_hat = self.model(x)

        x_rec, y_rec, y_hat_rec = self._recover_all(x, y, y_hat, c, r)
        mpjpe, pampjpe = calculate_error(y_hat_rec, y_rec)
        log_dict = {'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)

        if batch_idx == 10:
            self._visualize(x_rec, y_rec, y_hat_rec)

    def predict_step(self, batch, batch_idx):
        del batch_idx
        x, c, r, si, gi = batch['point_clouds'], batch['centroid'], batch['radius'], batch['sequence_index'], batch['global_index']
        y_hat = self.model(x)
        y_hat = self._recover_data(y_hat, c, r)
        return y_hat, si, gi

    def configure_optimizers(self):
        if self.use_teacher:
            all_params = list(self.model.parameters()) + list(self.model_teacher.parameters())
        else:
            all_params = self.model.parameters()
        optimizer = create_optimizer(self.hparams.optim_name, self.hparams.optim_params, all_params)
        scheduler = create_scheduler(self.hparams.sched_name, self.hparams.sched_params, optimizer)
        return [optimizer], [scheduler]
