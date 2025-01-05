import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from kornia.losses import ssim_loss


class ReconsLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, delta=1.0, epsilon=0.2, zeta=0.2, eta=0.2):
        super(ReconsLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon  # PSNR weight
        self.zeta = zeta        # SSIM weight
        self.eta = eta          # MSE weight

    def forward(self, pred, target):
        pred_mito = pred[:, 0:1, :, :]
        pred_micro = pred[:, 1:2, :, :]
        true_mito = target[:, 0:1, :, :]
        true_micro = target[:, 1:2, :, :]

        # 1. Reconstruction Loss
        recon_loss = F.mse_loss(pred_mito, true_mito) + F.mse_loss(pred_micro, true_micro)

        # 2. Consistency Loss
        consist_loss = F.mse_loss(pred_mito + pred_micro, true_mito + true_micro)

        # 3. Contrast Loss (corrected version)
        contrast_loss = self.contrast_loss(pred_mito, pred_micro)

        # 4. Structure Preservation Loss
        struct_loss = self.structure_loss(pred_mito, true_mito) + self.structure_loss(pred_micro, true_micro)

        # 5. PSNR Loss
        psnr_loss = 60 - self.psnr(pred_mito, true_mito) - self.psnr(pred_micro, true_micro)

        # 6. MSE Loss
        mse_loss = F.mse_loss(pred, target)

        # Total Loss
        total_loss = self.alpha * recon_loss + \
                     self.beta * consist_loss + \
                     self.gamma * contrast_loss + \
                     self.delta * struct_loss + \
                     self.epsilon * psnr_loss + \
                     self.zeta * ssim_loss + \
                     self.eta * mse_loss

        return total_loss


    def contrast_loss(self, pred_mito, pred_micro):
        similarity = F.cosine_similarity(pred_mito.flatten(1), pred_micro.flatten(1), dim=1)
        return 1 - similarity.mean()


    def structure_loss(self, pred, true):
        def gradient(x):
            dx = torch.zeros_like(x)
            dx[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
            dy = torch.zeros_like(x)
            dy[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]
            return torch.cat([dx, dy], dim=1)

        return F.mse_loss(gradient(pred), gradient(true))


    def psnr(self, pred, target):
        mse = F.mse_loss(pred, target)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))


class ImprovedCellLoss(nn.Module):
    def __init__(self, temperature=0.5, alpha=0.5, focal_alpha=0.25, focal_gamma=2.0):
        super(ImprovedCellLoss, self).__init__()
        self.mse_weight = nn.Parameter(torch.tensor(1.0))
        self.ssim_weight = nn.Parameter(torch.tensor(1.0))
        self.gradient_weight = nn.Parameter(torch.tensor(0.5))
        self.contrastive_weight = nn.Parameter(torch.tensor(0.1))
        self.psnr_weight = nn.Parameter(torch.tensor(0.1))
        self.focal_weight = nn.Parameter(torch.tensor(0.1))
        self.temperature = temperature
        self.alpha = alpha
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, pred, target):
        pred_mito = pred[:, 0:1, :, :]
        pred_micro = pred[:, 1:2, :, :]
        true_mito = target[:, 0:1, :, :]
        true_micro = target[:, 1:2, :, :]

        mito_loss = self.calculate_losses(pred_mito, true_mito)
        micro_loss = self.calculate_losses(pred_micro, true_micro)

        total_loss = mito_loss + micro_loss

        # Add L1 regularization to weights
        l1_reg = sum(p.abs().sum() for p in self.parameters())
        total_loss += 1e-5 * l1_reg

        return total_loss

    def calculate_losses(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        ms_ssim_loss = 1 - ssim_loss(pred, target, window_size=11, reduction='mean')
        grad_loss = self.gradient_loss(pred, target)
        contrastive_loss = self.contrastive_loss(pred, target)
        psnr_loss = self.psnr_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)

        total_loss = (
                self.mse_weight * mse_loss +
                self.ssim_weight * ms_ssim_loss +
                self.gradient_weight * grad_loss +
                self.psnr_weight * psnr_loss +
                self.contrastive_weight * contrastive_loss +
                self.focal_weight * focal_loss
        )

        return total_loss

    def gradient_loss(self, pred, target):
        pred_grad = kornia.filters.spatial_gradient(pred)
        target_grad = kornia.filters.spatial_gradient(target)
        return F.l1_loss(pred_grad, target_grad)


    def contrastive_loss(self, pred, target):
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        sim_matrix = F.cosine_similarity(pred_flat.unsqueeze(1), target_flat.unsqueeze(0), dim=2) / self.temperature
        labels = torch.eye(pred.size(0), device=pred.device)
        loss = F.cross_entropy(sim_matrix, torch.argmax(labels, dim=1))
        return loss


    def psnr_loss(self, pred, target):
        mse = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2, 3))
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return torch.max(torch.tensor(0.0, device=pred.device), 35 - psnr.mean())


    def focal_loss(self, pred, target):
        # Ensure pred is in [0, 1] range
        pred = torch.sigmoid(pred)

        # Flatten pred and target
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        # Compute focal loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss

        return focal_loss.mean()

    def adaptive_weight_update(self):
        with torch.no_grad():
            total_weight = self.mse_weight + self.ssim_weight + self.gradient_weight + self.psnr_weight + self.contrastive_weight + self.focal_weight
            self.mse_weight.div_(total_weight)
            self.ssim_weight.div_(total_weight)
            self.gradient_weight.div_(total_weight)
            self.psnr_weight.div_(total_weight)
            self.contrastive_weight.div_(total_weight)
            self.focal_weight.div_(total_weight)