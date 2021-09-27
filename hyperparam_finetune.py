from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from functools import partial

import torch
import argparse
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from builder import Builder
import os
import pandas as pd

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
parser.add_argument('--batch', type=int, default=512, help='batch size')
parser.add_argument('--workers', type=int, default=8, help='number of workers')
parser.add_argument('--device', default='cuda', help='wether to use gpu')
parser.add_argument('--mode', default='probe', help='choose to finetune whole model\
                    or only linear probe')
parser.add_argument('--num_samples', default=4, type=int, help='number of processes')
parser.add_argument('--gpus_per_trial', default=0.25, type=float, help='gpus per process')

args = parser.parse_args()


def get_data_loader():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(args.img_size * 256/224), interpolation=3),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_transform)
    dataset_val = datasets.ImageFolder(os.path.join(args.data, "val"), transform=val_transform)
    
    train_loader = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=args.workers)
    
    num_cls = len(val_loader.dataset.classes)
    
    return train_loader, val_loader, num_cls


    
def train_breg(config, checkpoint_dir=None):
    
    
    train_loader, val_loader, num_cls = get_data_loader()
    
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
    
       if args.mode == 'probe':   
           for name, param in encoder.named_parameters():
                if name not in ['head.weight', 'head.bias']:
                    param.requires_grad = False
            
                
    else:
       del encoder.mlp_head
    
       linear_layer = torch.nn.Linear(args.linear_dim, num_cls).cuda()
    
       encoder.mlp_head = linear_layer
       encoder.mlp_head.weight.data.normal_(mean=0.0, std=0.01)
       encoder.mlp_head.bias.data.zero_()
       
       if args.mode == 'probe':
           for name, param in encoder.named_parameters():
                if name not in ['mlp_head.weight', 'mlp_head.bias']:
                    param.requires_grad = False

    if torch.cuda.device_count() > 1:
        print("We have", torch.cuda.device_count(), "GPUs available!")
        encoder = torch.nn.DataParallel(encoder, device_ids=[0,1,2,3,4,5,6,7])
        
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config['lr'], weight_decay=config['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    if args.mode == 'probe':
        encoder.eval()
        
    
    for e in range(args.epochs):
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
            val_loss = 0
            val_steps = 0
            for imgs, labels in val_loader:
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')
                
                with torch.cuda.amp.autocast():
                    val_outputs = encoder(imgs)
                    loss = criterion(val_outputs, labels)
                    
                val_loss += loss.cpu().numpy()
                val_steps += 1
                acc = (val_outputs.argmax(dim=1) == labels).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                
        tune.report(loss=(val_loss/val_steps), accuracy=epoch_val_accuracy)
        
        
def main(num_samples, epochs, gpus_per_trail):
    config = {
        'lr': tune.loguniform(1e-4, 1e-1),
        'wd': tune.loguniform(1e-6, 1e-1)
        }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    result = tune.run(
        partial(train_breg),
        resources_per_trial={'cpu':2, 'gpu':gpus_per_trail},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
if __name__ == '__main__':
    main(args.num_samples, args.epochs, args.gpus_per_trial)
                
        
        
        
       
    
    
        
    
