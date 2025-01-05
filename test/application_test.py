import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
from archs.unet import UNet
from archs.improvedAemsn import ImprovedAemsn
import tifffile
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/output_application', help='output dir')
    parser.add_argument('--data_dir', default='/root/autodl-tmp/Application', help='input images dir')
    parser.add_argument('--model_name', default='AEMSN', choices=['UNet', 'AEMSN'])
    parser.add_argument('--input_channels', default=1, type=int, help='input channels')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    return parser.parse_args()


class MultiCellTifDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cell_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_paths = []
        for cell_dir in self.cell_dirs:
            cell_path = os.path.join(root_dir, cell_dir)
            self.image_paths.extend([os.path.join(cell_path, f) for f in os.listdir(cell_path) if f.endswith('.tif')])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = tifffile.imread(img_path)

        # Convert to float32 and normalize
        image = image.astype(np.float32)
        max_value = np.max(image)
        if max_value > 0:
            image = image / max_value

        # Add channel dimension
        image = np.expand_dims(image, axis=0)

        return torch.from_numpy(image), img_path, max_value


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    if args.name is None:
        args.name = f'{args.model_name}_inference'

    config_path = r'/root/autodl-tmp/output_recons/models/MitoMts_ImprovedUKan/tmp_res_1/config.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("=> creating model", config['model_name'])
    if config['model_name'] == 'UNet':
        model = UNet(n_channels=config['input_channels'], n_classes=config['num_classes']).to(device)
    elif config['model_name'] == 'AEMSN':
        model = ImprovedAemsn(n_channels=config['input_channels'], n_classes=config['num_classes'], device=device).to(device)
    else:
        raise NotImplementedError('Model not implemented')

    ckpt = torch.load('/root/autodl-tmp/output_recons/models/MitoMts_ImprovedUKan/model_final/epoch_80.pth')
    model.load_state_dict(ckpt)
    model.eval()

    test_dataset = MultiCellTifDataset(args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for inputs, img_paths, max_values in tqdm(test_loader, total=len(test_loader)):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Separate mitochondria and microtubules outputs
            mito_output = outputs[:, 0, :, :].cpu().numpy()
            micro_output = outputs[:, 1, :, :].cpu().numpy()

            # Scale back using the original max value
            max_value = max_values.item()  # Convert to Python scalar
            mito_output = (mito_output * max_value).astype(np.uint16)
            micro_output = (micro_output * max_value).astype(np.uint16)

            # Save outputs
            img_path = img_paths[0]
            cell_dir = os.path.basename(os.path.dirname(img_path))
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            output_cell_dir = os.path.join(args.output_dir, args.name, 'inference_results', cell_dir)
            os.makedirs(output_cell_dir, exist_ok=True)

            tifffile.imwrite(os.path.join(output_cell_dir, f'{base_name}_mito.tif'), mito_output[0])
            tifffile.imwrite(os.path.join(output_cell_dir, f'{base_name}_micro.tif'), micro_output[0])

    print(f"Inference completed. Results saved in {os.path.join(args.output_dir, args.name, 'inference_results')}")


if __name__ == '__main__':
    main()