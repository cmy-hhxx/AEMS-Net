import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import os
from skimage import io
from scipy.ndimage import gaussian_filter
from archs.improvedAemsn import ImprovedAemsn
from archs.unet import UNet
from mpl_toolkits.axes_grid1 import make_axes_locatable


class UNetReconstructionCAM:
    def __init__(self, model: nn.Module, target_layers: List[str], smooth_sigma: float = 0.05, threshold: float = 0.05, output_dir='original_cams'):
        self.model = model
        self.target_layers = target_layers
        self.smooth_sigma = smooth_sigma
        self.threshold = threshold
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self._register_hooks()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cam_counter = 0

    def _register_hooks(self):
        def save_activation(name: str):
            def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
                self.activations[name] = output
            return hook

        def save_gradient(name: str):
            def hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
                self.gradients[name] = grad_output[0]
            return hook

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(save_activation(name))
                module.register_full_backward_hook(save_gradient(name))  # 使用 register_full_backward_hook

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.max()

    def get_reconstruction_cam(self, x: torch.Tensor) -> Tuple[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
                                                               Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
                                                               torch.Tensor,
                                                               torch.Tensor]:
        x = self._normalize_input(x)
        output = self.model(x)
        mitochondria, microtubules = output[:, 0:1], output[:, 1:2]

        mito_cams = self._compute_cam_for_target(mitochondria)
        micro_cams = self._compute_cam_for_target(microtubules)

        return mito_cams, micro_cams, mitochondria, microtubules

    def _compute_cam_for_target(self, target: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self.model.zero_grad()
        target.sum().backward(retain_graph=True)
        positive_cams, negative_cams = self._compute_cam()
        return positive_cams, negative_cams

    def _compute_cam(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        positive_cams, negative_cams = {}, {}
        for layer_name in self.target_layers:
            activations = self.activations[layer_name]
            grads = self.gradients[layer_name]

            positive_cams[layer_name] = self._compute_single_cam(activations, F.relu(grads), f"positive_{layer_name}")
            negative_cams[layer_name] = self._compute_single_cam(activations, F.relu(-grads), f"negative_{layer_name}")
        self.cam_counter += 1  # 每次完成一组CAM的计算后增加计数器

        return positive_cams, negative_cams

    def _save_original_cam(self, cam: torch.Tensor, cam_type: str):
        # 将CAM转换为numpy数组
        cam_np = cam.detach().cpu().numpy()

        # 对每个批次中的图像进行循环
        for i, single_cam in enumerate(cam_np):
            # 去除多余的维度
            single_cam = np.squeeze(single_cam)

            # 创建图像
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(single_cam, cmap='jet')
            # plt.colorbar()
            # plt.title(f'Original CAM - Image {self.cam_counter}')
            plt.axis('off')
            plt.tight_layout(pad=0.5)

            # 保存图像
            save_path = os.path.join(self.output_dir, f'original_cam_{cam_type}_call_{self.cam_counter}_image_{i+1}.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
        print(f"Original CAMs saved in {self.output_dir}")

    def _compute_single_cam(self, activations: torch.Tensor, grads: torch.Tensor, cam_type: str) -> torch.Tensor:
        cam = torch.sum(grads * activations, dim=1, keepdim=True)
        cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
        cam = self._normalize(cam)
        cam = self._apply_smoothing(cam)
        cam = self._apply_thresholding(cam)
        return cam

    @staticmethod
    def _normalize(cam: torch.Tensor) -> torch.Tensor:
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def _apply_smoothing(self, cam: torch.Tensor) -> torch.Tensor:
        cam_np = cam.detach().cpu().numpy()
        smoothed_cam = np.stack([gaussian_filter(c[0], sigma=self.smooth_sigma) for c in cam_np])
        return torch.from_numpy(smoothed_cam).unsqueeze(1).to(cam.device)

    def _apply_thresholding(self, cam: torch.Tensor) -> torch.Tensor:
        return F.threshold(cam, self.threshold, 0)


def apply_unet_reconstruction_cam(unet_model: nn.Module, input_image: np.ndarray, target_layers: List[str]) -> Tuple[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
                                                                                                                     Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
                                                                                                                     torch.Tensor,
                                                                                                                     torch.Tensor]:
    cam_model = UNetReconstructionCAM(unet_model, target_layers)
    input_tensor = torch.from_numpy(input_image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(next(unet_model.parameters()).device)
    return cam_model.get_reconstruction_cam(input_tensor)


def visualize_and_save_all_layers(input_image: np.ndarray,
                                  mito_cams: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
                                  micro_cams: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
                                  mitochondria: torch.Tensor, microtubules: torch.Tensor, save_dir: str,
                                  filename_prefix: str = "sample"):
    os.makedirs(save_dir, exist_ok=True)
    individual_activations_dir = os.path.join(save_dir, "individual_activations")
    os.makedirs(individual_activations_dir, exist_ok=True)

    mito_positive_cams, mito_negative_cams = mito_cams
    micro_positive_cams, micro_negative_cams = micro_cams

    def enhance_cam(cam: np.ndarray, percentile: float = 97) -> np.ndarray:
        """Enhance CAM by thresholding and normalizing"""
        threshold = np.percentile(cam, percentile)
        cam = np.clip(cam, 0, threshold)
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def visualize_overview(cams_dict: Dict[str, torch.Tensor], title_prefix: str, filename_suffix: str,
                           original_image: np.ndarray):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for i, (layer_name, cam) in enumerate(cams_dict.items()):
            ax = axes.flatten()[i]
            cam_np = cam.detach().cpu().numpy().squeeze()
            enhanced_cam = enhance_cam(cam_np)

            # Overlay CAM on original image
            ax.imshow(original_image, cmap='gray')
            im = ax.imshow(enhanced_cam, cmap='jet', alpha=0.6)
            ax.set_title(f'{title_prefix} - {layer_name}')
            ax.axis('off')

            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_{filename_suffix}_overview.png"), dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    visualize_overview(mito_positive_cams, 'Mito Positive CAM', 'mito_positive', input_image)
    visualize_overview(mito_negative_cams, 'Mito Negative CAM', 'mito_negative', input_image)
    visualize_overview(micro_positive_cams, 'Micro Positive CAM', 'micro_positive', input_image)
    visualize_overview(micro_negative_cams, 'Micro Negative CAM', 'micro_negative', input_image)

    def visualize_single_layer_cam(cam: torch.Tensor, title: str, filename: str, original_image: np.ndarray):
        plt.figure(figsize=(12, 5))

        # Original image
        # plt.subplot(1, 2, 1)
        # plt.imshow(original_image, cmap='gray')
        # plt.title('Original Image')
        # plt.axis('off')

        # Enhanced CAM overlay
        plt.subplot(1, 1, 1)
        cam_np = cam.detach().cpu().numpy().squeeze()
        enhanced_cam = enhance_cam(cam_np)
        plt.imshow(original_image, cmap='gray')
        im = plt.imshow(enhanced_cam, cmap='jet', alpha=0.6)
        # plt.title(title)
        plt.axis('off')

        # plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_activations_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    for cams_dict, prefix in [(mito_positive_cams, 'mito_positive'),
                              (mito_negative_cams, 'mito_negative'),
                              (micro_positive_cams, 'micro_positive'),
                              (micro_negative_cams, 'micro_negative')]:
        for layer_name, cam in cams_dict.items():
            title = f'{prefix.capitalize()} CAM - {layer_name}'
            filename = f"{filename_prefix}_{prefix}_{layer_name}.png"
            visualize_single_layer_cam(cam, title, filename, input_image)

    print(f"Saved visualizations to {save_dir}")


def process_image(model: nn.Module, image_path: str, target_layers: List[str], save_directory: str):
    image = io.imread(image_path)
    assert image.shape == (256, 256), f"Expected image shape (256, 256), but got {image.shape}"

    mito_cams, micro_cams, mitochondria, microtubules = apply_unet_reconstruction_cam(model, image, target_layers)

    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    image_save_dir = os.path.join(save_directory, image_filename)

    visualize_and_save_all_layers(
        image, mito_cams, micro_cams, mitochondria, microtubules, image_save_dir,
        filename_prefix=image_filename
    )


def load_model(model_path: str, n_channels: int, n_classes: int) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedAemsn(n_channels=n_channels, n_classes=n_classes)
    # model = UNet(n_channels=n_channels, n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}")
    return model


def main():
    model_path = r"best_model.pth"
    n_channels, n_classes = 1, 2
    loaded_model = load_model(model_path, n_channels, n_classes)

    input_directory = r""
    save_directory = r""
    target_layers = ['down1', 'down2', 'down3', 'down4', 'up1', 'up2', 'up3', 'up4']

    for filename in os.listdir(input_directory):
        if filename.endswith('.tif'):
            image_path = os.path.join(input_directory, filename)
            process_image(loaded_model, image_path, target_layers, save_directory)


if __name__ == "__main__":
    main()