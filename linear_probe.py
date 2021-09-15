import torch
import argparse
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from builder import Builder
import os
import pandas as pd

torch.manual_seed(1)


parser = argparse.ArgumentParser()

parser.add_argument('data', type=str, help='path to finetune dataset')
parser.add_argument('weights', type=str, help='path to pretrained model weights')
parser.add_argument('--base_model',
                        default='ViT16-S',
                        help='model architecture',
                        choices=["ViT16-S", "ViT16-B", "ViT32-S", "ViT32-B", "Swin-S", "Swin-T", "Swin-B"])

parser.add_argument('--d_subs', default=200, type=int, help='k subnets')
parser.add_argument('--hidden_size', default=128, type=int,
                        help='subnetworks hidden layer size')
parser.add_argument('--bn', dest='bn', default=True, action='store_true', help=('wether to use batch norm in subs'))
parser.add_argument('--img_size', type=int, default=32, help='size of images')
parser.add_argument('--breg_dim', type=int, default=128, help='representation dimension')
parser.add_argument('--linear_dim', type=int, default=384, help='size of linear probing dimension')
parser.add_argument('--classes', type=int, default=10, help='number of image classes')
parser.add_argument('--channels', type=int, default=3, help='number of input channels')
parser.add_argument('--epochs', type=int, default=90, help='number of fine tune epochs')
parser.add_argument('--batch', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for linear probing')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for linear probe')
parser.add_argument('--workers', type=int, default=8, help='number of workers')
parser.add_argument('--device', default='cuda', help='wether to use gpu')
parser.add_argument('--dataset', default='cifar10', help='which dataset to use', choices=['cifar10',
                                                                                          'cifar100',
                                                                                          'stl10'])
parser.add_argument('--resize', dest='resize', default=False, action='store_true', help='resizes all images to 224 for Swin Transformer')
parser.add_argument('--save_path', default=None, help='path to save best model')

args = parser.parse_args()

if args.save_path is not None:
    path = args.save_path
else:
   if os.path.isdir('Linear_Runs'):
       path = os.getcwd() + '/Linear_Runs'
   else:
       os.mkdir('Linear_Runs')
       path = os.getcwd() + '/Linear_Runs'
       
    


if args.dataset == 'cifar10':
   normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
   
elif args.dataset == 'stl10':
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std = [0.2471, 0.2435, 0.2616])
    
elif args.dataset == 'cifar100':
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

if args.resize:
    train_transform = transforms.Compose([
        transforms.Resize(224),
        #transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224), normalize])
   
else:
    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([transforms.Resize(args.img_size),
                                        transforms.CenterCrop(args.img_size),
                                        transforms.ToTensor(), normalize])
    

if args.dataset == 'cifar10':
   train = torchvision.datasets.CIFAR10(args.data, train=True, transform=train_transform)
   val = torchvision.datasets.CIFAR10(args.data, train=False, transform=val_transform)
   
elif args.dataset == 'stl10':
    train = torchvision.datasets.STL10(args.data, split='train', transform=train_transform)
    val = torchvision.datasets.STL10(args.data, split='test', transform=val_transform)
    
elif args.dataset == 'cifar100':
    train = torchvision.datasets.CIFAR100(args.data, train=True, transform=train_transform)
    val = torchvision.datasets.CIFAR100(args.data, train=False, transform=val_transform)
    
train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers)
val_loader = DataLoader(val, batch_size=256, shuffle=False, num_workers=args.workers)


###############################
#### Load Model

builder = Builder(args.base_model, args.img_size, args.breg_dim, args.classes,
                 args.d_subs,args.hidden_size, args.bn,
                 args.device)

model = builder.model

model.load_state_dict(torch.load(args.weights))
    
encoder = model.backbone

if args.base_model in ["Swin-S", "Swin-T", "Swin-B"]:
   del encoder.head

   linear_layer = torch.nn.Linear(args.linear_dim, args.classes).cuda()

   encoder.head = linear_layer
   encoder.head.weight.data.normal_(mean=0.0, std=0.01)
   encoder.head.bias.data.zero_()

   for name, param in encoder.named_parameters():
        if name not in ['head.weight', 'head.bias']:
            param.requires_grad = False
            
else:
   del encoder.mlp_head

   linear_layer = torch.nn.Linear(args.linear_dim, args.classes).cuda()

   encoder.mlp_head = linear_layer
   encoder.mlp_head.weight.data.normal_(mean=0.0, std=0.01)
   encoder.mlp_head.bias.data.zero_()

   for name, param in encoder.named_parameters():
        if name not in ['mlp_head.weight', 'mlp_head.bias']:
            param.requires_grad = False
    



def train_linear_probe(encoder, train_loader, val_loader, lr, epochs, classes, path):
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    #norm = torch.nn.BatchNorm1d(linear_dim).to('cuda')
    encoder.eval()
    

    val_acc = []
    for e in range(epochs):
        epoch_acc = []
        for i, data in enumerate(train_loader, 0):
            imgs, labels = data
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')
            
            
            with torch.cuda.amp.autocast():
                 outputs = encoder(imgs)
                 loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1, 1)
            optimizer.step()
            
            batch_acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_acc.append(batch_acc)
            
        scheduler.step()
        
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for imgs, labels in val_loader:
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')
                
                with torch.cuda.amp.autocast():
                    val_outputs = encoder(imgs)
                    val_loss = criterion(val_outputs, labels)

                acc = (val_outputs.argmax(dim=1) == labels).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)
                
        save(model, path, args.base_model, args.dataset, args.d_subs, args.lr,
             args.bn, val_acc,  epoch_val_accuracy, e)
        
        val_acc.append(epoch_val_accuracy.item())
                    

                
                
            
        print('Epoch %d: train accuracy = %.2f, val accuracy = %.2f' %(e, 100 * sum(epoch_acc)/len(epoch_acc),
                                                                    100 * epoch_val_accuracy))
        
        
def save(model, path, base_model, dataset, d_subs, lr, bn, acc_ls, acc, epoch):
    save_name_pre = '{}_D{}_{}_{}_{}'.format(
        dataset, d_subs,
        base_model, lr,
        bn)
    
    if os.path.isdir('Linear_Runs/{}'.format(save_name_pre)):
       path = os.getcwd() + '/Linear_Runs'.format(save_name_pre)
    else:
       os.mkdir('Linear_Runs/{}'.format(save_name_pre))
       path = os.getcwd() + '/Linear_Runs'.format(save_name_pre)
    model_dir = os.path.join(path, '{}_model.pth'.format(save_name_pre))
    csv_dir = os.path.join(path, '{}_stats.csv'.format(save_name_pre))
    
    data_frame = pd.DataFrame(data=[acc_ls], index=range(1, epoch + 1))
    data_frame.to_csv(csv_dir, index_label='epoch')
    
    if epoch > 1 and acc > max(acc_ls):
        state_dict = model.state_dict()
        torch.save(state_dict, model_dir)
        print('Best Validation Accuracy So Far !! Saving Model !! ACC: %.2f' %(100 * acc))
    
    
    
if __name__=='__main__':
    train_linear_probe(encoder, train_loader, val_loader, 
                       args.lr, args.epochs, args.classes, path)

