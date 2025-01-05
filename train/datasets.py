import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class MitoMTDataset(Dataset):
    def __init__(self, root_dir, phase='training', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform

        self.image_dir = os.path.join(root_dir, phase)
        self.gt_dir = os.path.join(root_dir, 'gt', phase)

        self.images = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Extract cell and sample information from the image name
        parts = img_name.split('_')
        cell_nums = parts[1].split('&')
        sample_nums = parts[3].split('&')

        # Construct the corresponding gt file names
        mito_gt_name = f"cell_{cell_nums[0]}_sample_{sample_nums[0]}.tif"
        mt_gt_name = f"cell_{cell_nums[1]}_sample_{sample_nums[1]}.tif"
        mito_gt_path = os.path.join(self.gt_dir, 'Mito', mito_gt_name)
        mt_gt_path = os.path.join(self.gt_dir, 'MTs', mt_gt_name)

        # Load image (16-bit grayscale)
        image = Image.open(img_path)
        image = np.array(image, dtype=np.float32)
        max_intensity = np.max(image)
        min_intensity = np.min(image)
        # (x - x_min) / (x_max - x_min + 1e-8)
        # image = (image - min_intensity) / (max_intensity - min_intensity)  # Normalize by its own max value
        image = image / max_intensity  # Normalize by its own max value

        # Load Mito gt
        mito_gt = Image.open(mito_gt_path)
        mito_gt = np.array(mito_gt, dtype=np.float32)
        mito_gt_max = np.max(mito_gt)
        mito_gt_min = np.min(mito_gt)
        # mito_gt = (mito_gt - mito_gt_min) / (mito_gt_max - mito_gt_min )  # Normalize by its own max value
        mito_gt = mito_gt / mito_gt_max if mito_gt_max > 0 else mito_gt

        # Load MT gt
        mt_gt = Image.open(mt_gt_path)
        mt_gt = np.array(mt_gt, dtype=np.float32)
        mt_gt_max = np.max(mt_gt)
        mt_gt_min = np.min(mt_gt)
        # mt_gt = (mt_gt - mt_gt_min) / (mt_gt_max - mt_gt_min)
        mt_gt = mt_gt / mt_gt_max if mt_gt_max > 0 else mt_gt

        # Combine Mito and MT gt
        gt = np.stack([mito_gt, mt_gt], axis=-1)  # Shape: (H, W, 2)

        if self.transform:
            augmented = self.transform(image=image, mask=gt)
            image = augmented['image']
            gt = augmented['mask']

        # Convert to torch tensors and adjust dimensions
        image = torch.from_numpy(image).unsqueeze(0).float()  # Shape: (1, H, W)
        gt = torch.from_numpy(gt).permute(2, 0, 1).float()  # Shape: (2, H, W)

        # Create a dictionary to store max values
        max_values = {
            'overlay': max_intensity,
            'mito': mito_gt_max,
            'mt': mt_gt_max
        }

        return image, gt, max_values, img_name

def get_dataloaders(root_dir, batch_size=4, num_workers=4):
    # 可以根据需要添加数据增强
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        # A.ElasticTransform(p=0.5),
        # A.GridDistortion(p=0.5),
        # A.OpticalDistortion(p=0.5),
        # A.ShiftScaleRotate(p=0.5),
    ])

    train_dataset = MitoMTDataset(root_dir, phase='training', transform=train_transform)
    val_dataset = MitoMTDataset(root_dir, phase='validating')
    # test_dataset = MitoMTDataset(root_dir, phase='testing')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    root_dir = '/root/autodl-tmp/without_normalization/output'

    train_loader, val_loader = get_dataloaders(root_dir)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    # print(f"Number of testing batches: {len(test_loader)}")

    # 检查一个批次
    images, gt, max_value, _ = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch mask shape: {gt.shape}")
