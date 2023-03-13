# python -W ignore main2_multi.py --save --batch-size 8 -d /home/dyf/database/NYUv2/aftercrop/ -lr 1e-4 --epoch 1000
import cv2
import os
import sys
sys.path.append('../')
import time
import torch
import criteria
import argparse
import NYUdataset2 as Nd
from torch.utils.data import DataLoader
import torch.optim as optim
from net import *
from imageio import imread, imsave
from torch.utils.tensorboard import SummaryWriter   
from net_bak import *
import shutil


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Example training script')
    # yapf: disable
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default='/home/dyf/database/NYUv2/aftercrop/',
        help='Training dataset'
    )
    parser.add_argument(
        '--cuda',
        type=int,
        nargs='+',
        default=0,
        help='cuda device number'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='cuda device number'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=1000,
        help='epoch nums'
    )
    parser.add_argument(
        '-lr',
        '--learning-rate',
        default=1e-4,
        type=float,
        help='Learning rate (default: %(default)s)'
    )
    parser.add_argument(
        '-r',
        '--resolution',
        type=int,
        nargs=2,
        default=(384, 384),
        help='input image resolution'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save model to disk'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Only test'
    )
    parser.add_argument(
        '--savepic',
        action='store_true',
        help='Test save pictures'
    )
    parser.add_argument(
        '--optim',
        action='store_true',
        help='Load optimizer'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='second_checkpoint_best_loss.pth.tar',
        help='load weights'
    )
    args = parser.parse_args(argv)
    return args


def save_checkpoint(state, is_best, filename='checkpoint_best_loss.pth.tar'):
    if is_best:
        if os.path.exists(filename):
            shutil.copy(filename, filename+'copy')
        torch.save(state, filename)


def get_params(model, name='net'):
    print('The parameters of model ' + name)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

def TensorToDepthimg(input, filename, ColorMap=False):
    foldername, name = os.path.split(filename)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    input = input.clone().detach()
    input = input.to(torch.device('cpu'))
    input = input.squeeze(dim=0)
    input = input/input.max()
    output = torch.tensor(input*65535, dtype=torch.int)
    output = output.numpy().astype(np.uint16)
    if ColorMap:
        output_u8 = (output/256).astype(np.uint8)
        output_color = cv2.applyColorMap(output_u8, colormap=cv2.COLORMAP_HSV)
        colorname = filename[0:-4] + '_color.png'
    imsave(filename, output)
    cv2.imwrite(colorname, output_color)
    print('save ' + filename)

def TensorToRGB(input, filename):
    input = input.clone().detach()
    input = input.to(torch.device('cpu'))
    input = input.squeeze(dim=0)
    input = input.permute(1 ,2 ,0)
    output = torch.tensor(input * 256, dtype=torch.int)
    output = output.numpy().astype(np.uint8)
    # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    imsave(filename, output)
    print('save ' + filename)

def train_epoch(epoch, train_dataloader, model, model2, criterion, optimizer):
    device = next(model.parameters()).device
    model.eval()
    model2.train()
    Loss = AverageMeter()
    for i, d in enumerate(train_dataloader):
        gt = d[0].to(device)
        raw = d[1].to(device)
        rgb = d[2].to(device)
        mask = (raw>0).type(torch.float32)
        raw = raw.unsqueeze(dim=1)
        with torch.no_grad():
            dense_self = model(raw)
            dense_self = dense_self*(1-mask.unsqueeze(dim=1)) + raw * mask.unsqueeze(dim=1)
        # batch = {"color":rgb, "raw_depth":dense_self, "mask":mask.unsqueeze(dim=1)}
        dense = model2(dense_self, rgb)

        dense = dense.squeeze(dim=1)
        dense = dense * (1-mask) + raw.squeeze(dim=1) * mask
        raw_mask = (raw==0)
        optimizer.zero_grad()
        loss = criterion(dense.unsqueeze(dim=1)[raw_mask], gt.unsqueeze(dim=1)[raw_mask])
        loss.backward()
        
        optimizer.step()
        Loss.update(loss)
        if i % 20 == 0:
            log_data = f'Train epoch {epoch}:' \
                f'{i*d[0].size()[0]}/{len(train_dataloader.dataset)}'\
                f'\tloss: {loss.item():.6f}({Loss.avg:.6f})'
            print(log_data)
    return Loss.avg

def test_epoch(epoch, dataloader, model, model2, criterion, save=False):
    device = next(model.parameters()).device
    model.eval()
    model2.eval()
    Loss = AverageMeter()
    RMSEloss = AverageMeter()
    MAEloss = AverageMeter()
    iMAEloss = AverageMeter()
    iRMSEloss = AverageMeter()
    RelLoss = AverageMeter()
    SSIMLoss = AverageMeter()
    rmseloss = criteria.MaskedMSELoss()
    maeloss = criteria.MaskedL1Loss()
    irmseloss = criteria.MaskediRMSELoss()
    imaeloss = criteria.MaskediMAELoss()
    relloss = criteria.Rel()
    ssimloss = criteria.SSIM()

    deltaloss = criteria.Getdelta([1.01,1.03,1.05,1.15,1.25,1.25**2,1.25**3])
    with torch.no_grad():
        for i, d in enumerate(dataloader):
            gt = d[0].to(device)
            raw = d[1].to(device)
            rgb = d[2].to(device)
            name = d[3]
            mask = (raw>0).type(torch.float32)
            raw = raw.unsqueeze(dim=1)
            dense_self = model(raw)
            dense_self = dense_self*(1-mask) + raw.squeeze(dim=1) * mask
            dense = model2(dense_self, rgb)

            dense = torch.squeeze(dense, dim=1)

            dense = dense * (1-mask) + raw.squeeze(dim=1) * mask
            
            if save:
                name = name[0][0:-4]
                dense_name = 'savepic/dense/' + name + '.png'
                # gt_name = 'savepic/gt/' + name + '.png'
                # raw_name = 'savepic/raw/' + name + '.png'
                TensorToDepthimg(dense, dense_name, ColorMap=True)
                # TensorToDepthimg(gt, gt_name, ColorMap=True)
                # TensorToDepthimg(raw.squeeze(dim=1), raw_name, ColorMap=True)

            dense_mm = torch.tensor(dense*256*1000)
            dense_km = torch.tensor(dense*256/1000)
            gt_mm = torch.tensor(gt*256*1000)
            gt_km = torch.tensor(gt*256/1000)

            loss = criterion(dense.unsqueeze(dim=1), gt.unsqueeze(dim=1))
            rmse = torch.sqrt(rmseloss(dense_mm, gt_mm))
            mae = maeloss(dense_mm, gt_mm)
            irmse = irmseloss(dense_km, gt_km)
            imae = imaeloss(dense_km, gt_km)
            
            rel = relloss(dense, gt)
            deltaloss.calculate(dense, gt)
            ssim = ssimloss(dense.unsqueeze(dim=1), gt.unsqueeze(dim=1))


            Loss.update(loss)
            RMSEloss.update(rmse)
            MAEloss.update(mae)
            iMAEloss.update(imae)
            iRMSEloss.update(irmse)
            RelLoss.update(rel)
            deltaloss.update()
            SSIMLoss.update(ssim)

    print(f'Test epoch {epoch}: Average:'
          f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}'
          f'\tLoss: {Loss.avg:.6f}'
          f'\tRMSE: {RMSEloss.avg:.3f}(mm)'
          f'\tMAE: {MAEloss.avg:.3f}(mm)'
          f'\tiRMSE: {iRMSEloss.avg:.3f}(km)'
          f'\tiMAE: {iMAEloss.avg:.3f}(km)'
          f'\tRel: {RelLoss.avg:.4f}'
          f'\tSSIM: {SSIMLoss.avg:.5f}')

    for delta, value in deltaloss.avg_dict.items():
        print(f'{delta},{value.avg:.5f}')

    return Loss.avg


def main(argv):
    args = parse_args(argv)
    
    train_dataset = Nd.NYUv2Dataset(dataroot=args.dataset, phase='train', resolution=args.resolution, random_crop=True, artifical_mask=False, extname='.npy', exp=10.1)
    val_dataset = Nd.NYUv2Dataset(dataroot=args.dataset, phase='val', resolution=(384,384), random_crop=False, artifical_mask=False, extname='.npy', exp=10.1)

    train_dataloader = DataLoader(train_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=5, 
                                shuffle=True, 
                                pin_memory=True)
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=1, 
                            num_workers=5, 
                            shuffle=False, 
                            pin_memory=True)

    if args.test:
        test_dataset = Nd.NYUv2Dataset(dataroot=args.dataset, phase='test', resolution=(384,384), random_crop=False, artifical_mask=False, extname='.npy', exp=10.1)
        test_dataloader = DataLoader(test_dataset, 
                        batch_size=1, 
                        num_workers=5, 
                        shuffle=False, 
                        pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    device_ids = list(args.cuda)
    if device=='cuda':
        torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
    print('temp gpu device number:')
    print(torch.cuda.current_device())

    net = SelfCompletion()
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    net = net.cuda(device=device_ids[0])

    net2 = UnionComp(window_size=6, embed_dim=96, down_depths=[2,2,6,2], catstep=2, up_depths=[6,2,2], drop_rate=0.1)
    net2 = torch.nn.DataParallel(net2, device_ids=device_ids)
    net2 = net2.cuda(device=device_ids[0])

    optimizer = optim.Adam(net2.parameters(), lr=args.learning_rate)

    # 严格加载
    if os.path.exists("checkpoint_best_loss.pth.tar"):
        model = torch.load("checkpoint_best_loss.pth.tar", map_location=lambda storage, loc: storage)
        model.keys()
        net.module.load_state_dict(model['state_dict'], strict=True)
        # epoch_now = model['epoch']
        del model
        print('load model self completion ok')
    else:
        # epoch_now = -1
        print('train from none')

    # pretrained_dict = torch.load("/home/dyf/dev/MyDepth/uwork_sun/history_2211192031_unet_dmlrn/second_checkpoint_best_loss.pth.tar")['state_dict']
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net2.module.state_dict()}
    # for k, v in pretrained_dict.items():
    #     print(k)
    # raise ValueError('stop')

    if os.path.exists(args.weights):
        model2 = torch.load(args.weights, map_location=lambda storage, loc: storage)
        model2.keys()
        net2.module.load_state_dict(model2['state_dict'], strict=True)
        if args.optim:
            print('load optimizer ok')
            optimizer.load_state_dict(model2['optimizer'])
        epoch_now = model2['epoch']
        del model2
        print('load model2 ok, weights file is {}'.format(args.weights))
    else:
        epoch_now = -1
        # pretrained_dict = torch.load("/home/dyf/dev/MyDepth/unet_swin_sun/second_checkpoint_best_loss.pth.tar")['state_dict']
        # net2_dict = net2.module.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net2_dict}
        # net2_dict.update(pretrained_dict)
        # del pretrained_dict
        # net2.module.load_state_dict(net2_dict)
        # del net2_dict
        print('train from none')

    net = net.to(device)
    net2 = net2.to(device)
    criterion = criteria.MixedLoss(lmbda=0, lmbda2=2)
    # criterion = criteria.LogMaskedL1loss()


    get_params(net, 'net')
    get_params(net2, 'net2')

    best_loss = 1e10
    loss = test_epoch(epoch_now, val_dataloader, net, net2, criterion)
    best_loss = min(loss, best_loss)

    if args.test:
        loss = test_epoch(epoch_now, test_dataloader, net, net2, criterion, save=args.savepic)
        return loss

    writer = SummaryWriter('./boardlog2')
    for epoch in range(epoch_now+1, args.epoch+epoch_now+1):
        train_loss = train_epoch(epoch, train_dataloader, net, net2, criterion, optimizer)
        loss = test_epoch(epoch, val_dataloader, net, net2, criterion)
        writer.add_scalar('train_loss', train_loss, global_step=epoch, walltime=None)
        writer.add_scalar('val_loss', loss, global_step=epoch, walltime=None)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        print(f'is_best: {is_best:}')
        if args.save:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': net2.module.state_dict(),
                    'loss': loss,
                    'optimizer': optimizer.state_dict(),
                }, is_best, filename=args.weights)


if __name__ == '__main__':
    main(sys.argv[1:])

    



