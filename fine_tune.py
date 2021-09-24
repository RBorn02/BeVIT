import torch
import argparse
import random
import math
import torchvision
from data_aug.transformation import transform
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from builder import Builder
import os
import pandas as pd

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

torch.manual_seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

parser = argparse.ArgumentParser()

parser.add_argument('data', type=str, help='path to finetune dataset')
parser.add_argument('weights', type=str, help='path to pretrained model weights')
parser.add_argument('--base_model',
                        default='Swin-B',
                        help='model architecture',
                        choices=["ViT16-S", "ViT16-B", "ViT32-S", "ViT32-B", "Swin-S", "Swin-T", "Swin-B"])
parser.add_argument('--d_subs', default=200, type=int, help='k subnets')
parser.add_argument('--hidden_size', default=32, type=int,
                        help='subnetworks hidden layer size')
parser.add_argument('--bn', dest='bn', default=True, action='store_true', help=('wether to use batch norm in subs'))
parser.add_argument('--img_size', type=int, default=224, help='size of images')
parser.add_argument('--breg_dim', type=int, default=128, help='representation dimension')
parser.add_argument('--linear_dim', type=int, default=1024, help='size of linear probing dimension')
parser.add_argument('--channels', type=int, default=3, help='number of input channels')
parser.add_argument('--epochs', type=int, default=90, help='number of fine tune epochs')
parser.add_argument('--warmup', default=20, type=int, help='number of linear warmup steps for lr scheduler')
parser.add_argument('--batch', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for linear probing')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for linear probe')
parser.add_argument('--workers', type=int, default=8, help='number of workers')
parser.add_argument('--device', default='cuda', help='wether to use gpu')
parser.add_argument('--save_path', default=None, help='path to save best model')
parser.add_argument('--fraction', default=0.1, type=float, help='percent of dataset used for finetuning')

args = parser.parse_args()

if args.save_path is not None:
    path = args.save_path
else:
   if os.path.isdir('Fine_Tune_Runs'):
       path = os.getcwd() + '/Fine_Tune_Runs'
   else:
       os.mkdir('Fine_Tune_Runs')
       path = os.getcwd() + '/Fine_Tune_Runs'

####################################
#### Load and prepare Data
       
train_transform, val_transform = transform(args.img_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

dataset_train = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_transform)
dataset_val = datasets.ImageFolder(os.path.join(args.data, "val"), transform=val_transform)


f = int(args.fraction * len(dataset_train))

subset_idx = random.sample(range(0,len(dataset_train)), f)

    
train_loader = DataLoader(dataset_train, batch_size=args.batch, sampler=SubsetRandomSampler(subset_idx),
                          num_workers=args.workers)
val_loader = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=args.workers)

num_cls = len(val_loader.dataset.classes)


###############################
#### Load Model

builder = Builder(args.base_model, args.img_size, args.breg_dim,
                 args.d_subs, args.hidden_size, args.bn,
                 args.device)

model = builder.model

model.load_state_dict(torch.load(args.weights))
    
encoder = model.backbone


if args.base_model in ["Swin-S", "Swin-T", "Swin-B"]:
   del encoder.head

   linear_layer = torch.nn.Linear(args.linear_dim, num_cls).cuda()

   encoder.head = linear_layer
   encoder.head.weight.data.normal_(mean=0.0, std=0.01)
   encoder.head.bias.data.zero_()

else:
   del encoder.mlp_head

   linear_layer = torch.nn.Linear(args.linear_dim, num_cls).cuda()

   encoder.mlp_head = linear_layer
   encoder.mlp_head.weight.data.normal_(mean=0.0, std=0.01)
   encoder.mlp_head.bias.data.zero_()


if torch.cuda.device_count() > 1:
    print("We have", torch.cuda.device_count(), "GPUs available!")
    encoder = torch.nn.DataParallel(encoder, device_ids=[0,1,2,3,4,5,6,7]) 
    

def fine_tune(encoder, train_loader, val_loader, lr, epochs, classes, path):
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=args.wd)
    scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=args.warmup,
                    max_epochs=args.epochs)
    #norm = torch.nn.BatchNorm1d(linear_dim).to('cuda')
    

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
                
        save(model, path, args.base_model, args.data, args.d_subs, args.lr,
             args.bn, val_acc,  epoch_val_accuracy, e)
        
        val_acc.append(epoch_val_accuracy.item())
                    

                
                
            
        print('Epoch %d: train accuracy = %.2f, val accuracy = %.2f' %(e, 100 * sum(epoch_acc)/len(epoch_acc),
                                                                    100 * epoch_val_accuracy))
        
        
def save(model, path, base_model, data, d_subs, lr, bn, acc_ls, acc, epoch):
    save_name_pre = '{}_D{}_{}_{}_{}'.format(
        'test', d_subs,
        base_model, lr,
        bn)
    
    if os.path.isdir('Fine_Tune_Runs/{}'.format(save_name_pre)):
       path = os.getcwd() + '/Fine_Tune_Runs'.format(save_name_pre)
    else:
       os.mkdir('Fine_Tune_Runs/{}'.format(save_name_pre))
       path = os.getcwd() + '/Fine_Tune_Runs'.format(save_name_pre)
    model_dir = os.path.join(path, '{}_model.pth'.format(save_name_pre))
    csv_dir = os.path.join(path, '{}_stats.csv'.format(save_name_pre))
    
    data_frame = pd.DataFrame(data=[acc_ls], index=range(1, epoch + 1))
    data_frame.to_csv(csv_dir, index_label='epoch')
    
    if epoch > 1 and acc > max(acc_ls):
        state_dict = model.state_dict()
        torch.save(state_dict, model_dir)
        print('Best Validation Accuracy So Far !! Saving Model !! ACC: %.2f' %(100 * acc))
    
    
    
if __name__=='__main__':
    fine_tune(encoder, train_loader, val_loader, 
                       args.lr, args.epochs, num_cls, path)

    