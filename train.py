import torch
import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import copy

from train_configs import get_train_args
from dataset_configs import get_dataset_args
from datasets import get_dataset
from models import get_model
from loss_functions import SpatialEmbLoss

train_args = get_train_args()
dataset_args = get_dataset_args()

if train_args['save']:
    if not os.path.exists(train_args['save_dir']):
        os.makedirs(train_args['save_dir'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if train_args['display']:
    plt.ion() # script continues even after a plot is drawn on the screen
else:
    plt.ioff()
    plt.switch_backend('agg')

# Create dataloaders
dataset_name = dataset_args['name']

train_dataset = get_dataset(
                    dataset_name,
                    dataset_args['train_dataset'][dataset_name]
                )
train_dataloader = DataLoader(
                        train_dataset,
                        batch_size = dataset_args['train_dataset']['batch_size'],
                        shuffle = True, 
                        drop_last = True, 
                        num_workers = dataset_args['train_dataset']['workers'], 
                        pin_memory = torch.cuda.is_available()
                    )

val_dataset = get_dataset(
                    dataset_name,
                    dataset_args['val_dataset'][dataset_name]
                )
val_dataloader = DataLoader(
                        val_dataset,
                        batch_size = dataset_args['val_dataset']['batch_size'],
                        shuffle = False, 
                        drop_last = True, 
                        num_workers = dataset_args['val_dataset']['workers'], 
                        pin_memory = torch.cuda.is_available()
                    )

# Define model
model = get_model(train_args['model']['name'], train_args['model']['options'])
model.init_output(train_args['loss_opts']['n_sigma'])
#print(model)

# Define loss functions
criterion = SpatialEmbLoss(**train_args['loss_opts'])

# set optimizer
optimizer = optim.Adam(model.parameters(), lr=train_args['lr'], weight_decay=1e-4)

def lambda_(epoch):
    return pow((1-((epoch)/train_args['n_epochs'])), 0.9)

scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_,)

start_epoch = 0
best_iou = 0


def train_model(num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        epoch_loss = 0
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-'*10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                for param_group in optimizer.param_groups:
                    #print("LR", param_group['lr'])
                    # Set model to training mode
                    model.train()
            else:
                dataloader = val_dataloader
                # Set model to evaluation mode
                model.eval()

            epoch_samples = 0

            for samples in dataloader:
                imgs = samples['img']
                instances = samples['instance_mask'].squeeze()
                class_labels = samples['semantic_mask'].squeeze()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    output = model(imgs)
                    loss = criterion(output, instances, class_labels, **train_args['loss_w'])
                    epoch_loss +=  loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()


                    epoch_samples += imgs.size(0)
                    epoch_loss = epoch_loss/epoch_samples

            # Deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("Saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'best_model.pth')
        print(f'Train loss: {epoch_loss}')
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))

    print('Best val loss: {:4f}'.format(best_loss))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model


train_model()
