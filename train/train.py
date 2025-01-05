import argparse
from collections import OrderedDict
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from archs.unet import UNet
from archs.improvedAemsn import ImprovedAemsn
from torch.optim import lr_scheduler
from tqdm import tqdm
import loss
from datasets import get_dataloaders
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter
import os


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=101, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--dataseed', default=42, type=int,
                        help='')

    # dataset
    parser.add_argument('--dataset', default='MitoMts', help='dataset name')
    parser.add_argument('--data_dir', default='/root/autodl-tmp/without_normalization/output', help='dataset dir')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/output_recons', help='output dir')

    # model
    parser.add_argument('--model_name', default='AEMSN', choices=['UNet', 'AEMSN'])
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')

    # loss
    parser.add_argument('--loss', default='ImprovedCellLoss')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=50, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))

    for input, target, max_values, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
        else:
            output = model(input)
            loss = criterion(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, max_values, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
            else:
                output = model(input)
                loss = criterion(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
    ])


def main():
    # seed_torch(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['model_name'])
        else:
            config['name'] = '%s_%s' % (config['dataset'], config['model_name'])
    os.makedirs('/root/autodl-tmp/output_recons/models/%s' % config['name'], exist_ok=True)
    my_writer = SummaryWriter(f'/root/autodl-tmp/output_recons/models/%s/tf_logs' % config['name'])

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('/root/autodl-tmp/output_recons/models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = getattr(loss, config['loss'])().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['model_name'])
    if config['model_name'] == 'UNet':
        model = UNet(n_channels=config['input_channels'], n_classes=config['num_classes']).to(device)
    elif config['model_name'] == 'AEMSN':
        model = ImprovedAemsn(n_channels=config['input_channels'], n_classes=config['num_classes'], device=device).to(device)
    else:
        raise NotImplementedError('Model not implemented')

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    train_loader, val_loader = get_dataloaders(config['data_dir'])

    # 初始化 log 字典
    log = {'loss': [], 'epoch': []}

    # 添加验证集相关的键
    val_keys = ['val_loss']
    log.update({key: [] for key in val_keys})

    best_loss = float('inf')
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f' % train_log['loss'])

        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['val_loss'].append(val_log['loss'])

        pd.DataFrame(log).to_csv(f'/root/autodl-tmp/output_recons/models/%s/log.csv' % config['name'], index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)

        trigger += 1

        if val_log['loss'] < best_loss:
            torch.save(model.state_dict(), '/root/autodl-tmp/output_recons/models/%s/best_model.pth' % config['name'])
            best_loss = val_log['loss']
            print("=> saved best model=================================================")
            print(f'Best Loss: {best_loss:.4f}')
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            torch.save(model.state_dict(), '/root/autodl-tmp/output_recons/models/%s/last_model.pth' % config['name'])
            break

        if epoch % 30 == 0 :
            torch.save(model.state_dict(), '/root/autodl-tmp/output_recons/models/%s/epoch_%d.pth' % (config['name'], epoch))

        if epoch == config['epochs'] - 1:
            torch.save(model.state_dict(), '/root/autodl-tmp/output_recons/models/%s/last_model.pth' % config['name'])
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
