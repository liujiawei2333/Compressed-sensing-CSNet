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
parser.add_argument('--trainpath',default='./128image/train/')
parser.add_argument('--valpath',default='./128image/test/')
parser.add_argument('--batch-size',type=int,default=128,metavar='N')
parser.add_argument('--image-size',type=int,default=256,metavar='N')
parser.add_argument('--epochs',type=int,default=100,metavar='N')
parser.add_argument('--lr',type=float,default=1e-3,metavar='LR')
parser.add_argument('--cuda',action='store_true',default=True)
parser.add_argument('--ngpu',type=int,default=1,metavar='N')
parser.add_argument('--seed',type=int,default=1,metavar='S')
parser.add_argument('--log-interval',type=int,default=100,metavar='N')
parser.add_argument('--outf',default='./results')
parser.add_argument('--cr',type=int,default=20)
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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    net.apply(weights_init_normal)

def data_loader():
    kwopt = {'num_workers': 8, 'pin_memory': True} if opt.cuda else {}
    transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(opt.image_size),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.CenterCrop(opt.image_size),
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                        ])
    train_dataset = torchvision.datasets.ImageFolder(opt.trainpath,transform=transforms)
    val_dataset = torchvision.datasets.ImageFolder(opt.valpath,transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = opt.batch_size,shuffle = True,**kwopt)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = opt.batch_size,shuffle = True,**kwopt)
    return train_loader, val_loader

def train(epochs,trainloader, valloader):
    input, _ = trainloader.__iter__().__next__()
    input = input.numpy()
    sz_input = input.shape
    channels = sz_input[1]
    img_size = sz_input[3]

    target = torch.FloatTensor(opt.batch_size,channels,img_size,img_size)
    input = torch.FloatTensor(opt.batch_size,channels,img_size,img_size)

    CSnet = csnet.CSNET(channels,opt.cr)
    weights_init(CSnet,init_type='normal')
    optimizer = optim.Adam(CSnet.parameters(),lr=opt.lr,betas=(0.9,0.999))
    criterion_mse = nn.MSELoss()
    cudnn.benchmark = True

    if opt.cuda:
        device_id = [0]
        CSnet = nn.DataParallel(CSnet.cuda(),device_ids = device_id)
        criterion_mse.cuda()
        target = target.cuda()
        input = input.cuda()

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 80], gamma = 0.1, last_epoch=-1)

    for epoch in range(epochs):
        scheduler.step()
        for idx, (data, _) in enumerate(trainloader, 0):
            if data.size(0) != opt.batch_size:
                continue
            CSnet.train()
            data_array = data.numpy()
            for i in range(opt.batch_size):
                target[i] = torch.from_numpy(data_array[i])

                for j in range(channels):
                    input[i,j,:,:] = target[i,j,:,:]

            target_var = Variable(target)
            input_var = Variable(input)

            CSnet.zero_grad()
            output = CSnet(input_var,opt.batch_size,opt.ngpu)
            csnet_mse = criterion_mse(output,target_var)
            csnet_mse.backward()
            optimizer.step()

            if idx % opt.log_interval == 0:
                print('[%d/%d][%d/%d] mse:%.4f' % (epoch,epochs,idx,len(trainloader),csnet_mse.item()))

        torch.save(CSnet.state_dict(),'%s/cr%s/model/%d.pth' % (opt.outf,opt.cr,epoch))
        vutils.save_image(target_var.data,'%s/cr%s/image/_%03d_real.png' % (opt.outf,opt.cr,epoch),normalize=True)
        vutils.save_image(output.data,'%s/cr%s/image/_%03d_fake.png' % (opt.outf,opt.cr,epoch),normalize=True)
        writer.add_scalar('train/mse_loss_epoch', csnet_mse, epoch)

        CSnet.eval()
        val(epoch,channels,valloader,target,input,CSnet,criterion_mse)

def val(epoch,channels,valloader,target,input,CSnet,criterion_mse):
    csnet_mse_total = 0
    for idx, (data, _) in enumerate(valloader, 0):
        if data.size(0) != opt.batch_size:
            continue
        data_array = data.numpy()
        for i in range(opt.batch_size):
            target[i] = torch.from_numpy(data_array[i])

            for j in range(channels):
                input[i,j,:,:] = target[i,j,:,:]

        with torch.no_grad():
            target_var = Variable(target)
            input_var = Variable(input)
            output = CSnet(input_var,opt.batch_size,opt.ngpu)

            csnet_mse = criterion_mse(output,target_var)
            csnet_mse_total += csnet_mse

        if idx % 20 == 0:
            print('Test:[%d][%d/%d] mse:%.4f \n' % (epoch,idx,len(valloader),csnet_mse.item()))

    print('Test:[%d] average mse:%.4f,' % (epoch,csnet_mse_total.item() / len(valloader)))
    vutils.save_image(target_var.data,'%s/cr%s/image/test_%03d_real.png' % (opt.outf,opt.cr,epoch),normalize=True)
    vutils.save_image(output.data,'%s/cr%s/image/test_%03d_fake.png' % (opt.outf,opt.cr,epoch),normalize=True)
    writer.add_scalar('test/mse_loss_epoch', csnet_mse_total.item() / len(valloader), epoch)

def main():
    train_loader,val_loader = data_loader()
    train(opt.epochs,train_loader,val_loader)

if __name__ == '__main__':
    main()