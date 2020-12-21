import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision
import os
import argparse
import csnet
from torchvision import datasets,transforms
from torch.autograd import Variable
from torch.nn import init
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='CSNet')
parser.add_argument('--dataset',default='own_image')
parser.add_argument('--trainpath',default='../Image/train/')
parser.add_argument('--valpath',default='../Image/test/')
parser.add_argument('--batch-size',type=int,default=1,metavar='N')
parser.add_argument('--image-size',type=int,default=256,metavar='N')
parser.add_argument('--start_epoch',type=int,default=0,metavar='N')#加载checkpoint即会改变
parser.add_argument('--epochs',type=int,default=100,metavar='N')
parser.add_argument('--lr',type=float,default=1e-3,metavar='LR')
parser.add_argument('--cuda',action='store_true',default=True)
parser.add_argument('--ngpu',type=int,default=1,metavar='N')
parser.add_argument('--seed',type=int,default=1,metavar='S')
parser.add_argument('--log-interval',type=int,default=100,metavar='N')
parser.add_argument('--outf',default='./results')
parser.add_argument('--cr',type=int,default=20)
parser.add_argument('--resume',action='store_true',default=True)
opt = parser.parse_args()

if torch.cuda.is_available() and not opt.cuda:
    print("please run with GPU")
if opt.seed is None:
    opt.seed = np.random.randint(1,10000)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True

if not os.path.exists('%s/cr%s/model' % (opt.outf,opt.cr)):
    os.makedirs('%s/cr%s/model' % (opt.outf,opt.cr))
if not os.path.exists('%s/cr%s/image' % (opt.outf,opt.cr)):
    os.makedirs('%s/cr%s/image' % (opt.outf,opt.cr))
if not os.path.exists('%s/cr%s/log' % (opt.outf,opt.cr)):
    os.makedirs('%s/cr%s/log' % (opt.outf,opt.cr))
log_dir = '%s/cr%s/log' % (opt.outf,opt.cr)
writer = SummaryWriter(log_dir=log_dir)

def data_loader():
    kwopt = {'num_workers': 4, 'pin_memory': True} if opt.cuda else {}
    transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(opt.image_size),#随机剪裁
                    torchvision.transforms.RandomHorizontalFlip(),#依照概率水平翻转
                    torchvision.transforms.RandomVerticalFlip(),#依照概率垂直翻转
                    torchvision.transforms.ToTensor(),#转化为tensor
                    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                        ])
    train_dataset = torchvision.datasets.ImageFolder(opt.trainpath,transform=transforms)
    val_dataset = torchvision.datasets.ImageFolder(opt.valpath,transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = opt.batch_size,shuffle = True,**kwopt)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = opt.batch_size,shuffle = True,**kwopt)
    return train_loader, val_loader

def train(start_epoch,epochs,trainloader, valloader):
    input, _ = trainloader.__iter__().__next__()
    input = input.numpy()
    sz_input = input.shape#128*1*256*256
    channels = sz_input[1]#通道数（3）
    img_size = sz_input[3]#256

    input = torch.FloatTensor(opt.batch_size,channels,img_size,img_size)

    CSnet = csnet.CSNET(channels,opt.cr)

    for m in CSnet.modules():
        if isinstance(m, (nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in',nonlinearity='relu')

    optimizer = optim.Adam(CSnet.parameters(),lr=opt.lr,betas=(0.9,0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [4000], gamma = 0.1, last_epoch=-1)
   
    criterion_mse = nn.MSELoss()
    cudnn.benchmark = True

    if opt.cuda:
        device_id = [0]
        CSnet = nn.DataParallel(CSnet.cuda(),device_ids = device_id)
        criterion_mse.cuda()
        input = input.cuda()
    
    if opt.resume:
        if os.path.isfile('%s/checkpoint' % (opt.outf)):
            checkpoint = torch.load('%s/checkpoint' % (opt.outf))
            start_epoch = checkpoint['epoch'] + 1
            G.load_state_dict(checkpoint['model'])
            optimizer_G.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    min_loss = 100000
    for epoch in range(epochs):
        for idx, (input, _) in enumerate(trainloader, 0):
            if input.size(0) != opt.batch_size:
                continue
            CSnet.train()

            CSnet.zero_grad()
            output = CSnet(input,opt.batch_size,opt.ngpu)
            csnet_mse = criterion_mse(output,input.cuda())
            csnet_mse.backward()
            optimizer.step()
            scheduler.step()

            if idx % opt.log_interval == 0:
                print('[%d/%d][%d/%d] mse:%.4f' % (epoch,epochs,idx,len(trainloader),csnet_mse.item()))

        writer.add_scalar('train/mse',csnet_mse, epoch)
        a = vutils.make_grid(input[:1],normalize=True,scale_each=True)
        b = vutils.make_grid(output[:1],normalize=True,scale_each=True)

        writer.add_image('orin',a,epoch)
        writer.add_image('recon',b,epoch)

        CSnet.eval()
        average_mse = val(epoch,channels,valloader,input,CSnet,criterion_mse)

        if average_mse < min_loss:
            min_loss = average_mse
            print("save model")
            torch.save(CSnet.state_dict(),'%s/cr%s/model/CSnet.pth' % (opt.outf,opt.cr))

def val(epoch,channels,valloader,input,CSnet,criterion_mse):
    csnet_mse_total = 0
    average_mse = 0
    for idx, (input, _) in enumerate(valloader, 0):
        if input.size(0) != opt.batch_size:
            continue

        with torch.no_grad():
            output = CSnet(input,opt.batch_size,opt.ngpu)

            csnet_mse = criterion_mse(output,input.cuda())
            csnet_mse_total += csnet_mse
            average_mse = csnet_mse_total.item() / len(valloader)

        if idx % 20 == 0:
            print('Test:[%d][%d/%d] mse:%.4f \n' % (epoch,idx,len(valloader),csnet_mse.item()))

    print('Test:[%d] average mse:%.4f,' % (epoch,csnet_mse_total.item() / len(valloader)))
    writer.add_scalar('test/mse_loss_epoch', csnet_mse_total.item() / len(valloader), epoch)

    return average_mse

def main():
    train_loader,val_loader = data_loader()
    train(opt.start_epoch,opt.epochs,train_loader,val_loader)

if __name__ == '__main__':
    main()