import argparse
import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


from data_aug.data_loader import CustomDataLoader
from trainer import Trainer
from builder import Builder


from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--breg_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--length', default=1.5, type=float, help='value for lenth scale parameter of RBF kernel')
    parser.add_argument('--case', default=2, type=int, help='how to determine similarity')
    parser.add_argument('--k_nn', default=200, type=int, help='k in knn')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    parser.add_argument('--d_subs', default=200, type=int, help='d subnets')
    parser.add_argument('--hidden_size', default=32, type=int,
                        help='subnetworks hidden layer size')
    parser.add_argument('--bn', dest='bn', default=True, action='store_true', help=('wether to use batch norm in subs'))
    parser.add_argument('--lr', default=1e-3, type=float,help='initial learning rate')
    parser.add_argument('--warmup', default=20, type=int, help='number of linear warmup steps for lr scheduler')
    parser.add_argument('--wd', default=5e-2, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', default=10, type=int, help='seed for initializing training.')
    parser.add_argument('--workers', default=64, type=int, help='number of data loading workers')
    parser.add_argument('--lmbda', default=5, type=float, help='sets lambda for weighting the mixed loss')
    parser.add_argument('--margin', default=1, type=float, help='size of margin for margin loss')
    parser.add_argument('--use_margin', dest='use_margin', default=False, action='store_true', 
                        help='use margin loss')
    parser.add_argument('--base_model',
                        default='Swin-B',
                        help='model name',
                        choices=["ViT16-S", "ViT16-B", "ViT32-S", "ViT32-B",
                                 "Swin-T", "Swin-S", "Swin-B"])
    parser.add_argument('--img_size', default=224, type=int, help='size of images(currently set for ImageNet)')
   # parser.add_argument('--resize', dest='resize', default=False, action='store_true', help='resizes all images to 224 for Swin Transformer')
    
    
    parser.add_argument('--dataset_name', default='imagenet',
                    help='dataset name', choices=['stl10',
                                                  'cifar10',
                                                  'cifar100',
                                                  'brain',
                                                  'skin',
                                                  'retina',
                                                  'iris',
                                                  'imagenet'
                                                  'coco',
                                                  'tiny-imagenet'])
    # args parse
    args = parser.parse_args()
    

    # model setup and optimizer config
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        
        
    # create a tensorboard writer
    writer = SummaryWriter()
    # save config file
    save_config_file(writer.log_dir, args)
############################################################]
### Load Datasets and Dataloaders
    dl = CustomDataLoader()
    train_loader, memory_loader, test_loader = dl.get_loader(args.dataset_name, args.batch_size, args.workers, args.resize)
    
    num_cls = len(test_loader.dataset.classes)
    
###########################################################
### Initiate Model, Optimizer and Scheduler  
    builder = Builder(args.base_model, args.img_size, args.breg_dim,
                 args.d_subs, args.hidden_size, args.bn,
                 args.device)
    
    model = builder.model
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=args.warmup,
                    max_epochs=args.epochs)
    
    if torch.cuda.device_count() > 1:
            print("We have", torch.cuda.device_count(), "GPUs available!")
            model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    
    
    trainer = Trainer(model,
                      optimizer,
                      scheduler,
                      args.temperature,
                      args.length,
                      args.case,
                      num_cls,
                      args.epochs,
                      args.device,
                      args.margin,
                      args.use_margin)

    # training loop
    results = {'train_loss': [],
               'breg_loss': [],
               'NTXent_loss': [],
               'test_acc@1': [],
               'test_acc@5': []
              }
    save_name_pre = '{}_K{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.dataset_name, args.d_subs,
        args.base_model, args.lr,
        args.breg_dim, args.temperature,
        args.k_nn, args.batch_size, args.epochs)
    csv_dir = os.path.join(writer.log_dir, '{}_stats.csv'.format(save_name_pre))
    model_dir = os.path.join(writer.log_dir, '{}_model.pth'.format(save_name_pre))
    fig_dir = os.path.join(writer.log_dir, '{}_loss_acc.png'.format(save_name_pre))
    
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, bloss, NTXent = trainer.train(train_loader, epoch, args.lmbda)
        results['train_loss'].append(train_loss)
        results['breg_loss'].append(bloss)
        results['NTXent_loss'].append(NTXent)
        writer.add_scalar('loss/train', results['train_loss'][-1], epoch)
        
        test_acc_1, test_acc_5 = trainer.test(memory_loader, test_loader, args.k_nn, epoch)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        writer.add_scalar('acc@1/test', results['test_acc@1'][-1], epoch)
        writer.add_scalar('acc@5/test', results['test_acc@5'][-1], epoch)
        
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(csv_dir, index_label='epoch')
        
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, model_dir)
    
    # plotting loss and accuracies
    df = pd.read_csv(csv_dir)
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(20,5))
    axes[0].set_title('Loss/Train')
    axes[1].set_title('acc@1/test')
    axes[2].set_title('acc@5/test')
    sns.lineplot(ax=axes[0], x="epoch", y="train_loss", data=df)
    sns.lineplot(ax=axes[1], x="epoch", y="test_acc@1", data=df)
    sns.lineplot(ax=axes[2], x="epoch", y="test_acc@5", data=df)
    
    fig.savefig(fig_dir)
