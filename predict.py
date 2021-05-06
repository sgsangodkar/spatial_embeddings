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

from test_configs import get_test_args
from dataset_configs import get_dataset_args
from datasets import get_dataset
from models import get_model
from loss_functions import SpatialEmbLoss
from utils import Visualizer, Cluster, Logger
from torchvision import transforms

test_args = get_test_args()
dataset_args = get_dataset_args()

if test_args['save']:
    if not os.path.exists(test_args['save_dir']):
        os.makedirs(test_args['save_dir'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if test_args['display']:
    plt.ioff() # script continues even after a plot is drawn on the screen
else:
    plt.ioff()
    plt.switch_backend('agg')

# Create dataloaders
dataset_name = dataset_args['name']

test_dataset = get_dataset(
                    dataset_name,
                    dataset_args['test_dataset'][dataset_name]
                )
test_dataloader = DataLoader(
                        test_dataset,
                        batch_size = dataset_args['test_dataset']['batch_size'],
                        shuffle = False, 
                        drop_last = False, 
                        num_workers = dataset_args['test_dataset']['workers'], 
                        pin_memory = torch.cuda.is_available()
                    )

# Define model
model = get_model(test_args['model']['name'], test_args['model']['options'])

model.eval()

# cluster module
cluster = Cluster()


if os.path.exists(test_args['checkpoints']):
    state = torch.load(test_args['checkpoints'])
    #model.load_state_dict(state, strict=True)
else:
    assert False, 'checkpoint_path {} does not exist!'.format(test_args['checkpoints'])

sample = next(iter(test_dataloader))
#print(sample['img'][0].size())

im = sample['img'][0].unsqueeze(0)
print(f'image size {im.size()}')
instances = sample['instance_mask'][0]
print(f'instances size {instances.size()}')
class_labels = sample['semantic_mask'][0]
print(f'class_labels size {class_labels.size()}')


im_save = transforms.ToPILImage()(im.squeeze(0))
im_save.save('outputs/input.png')

with torch.no_grad():       
        output = model(im)
        instance_map, predictions = cluster.cluster(
                                        output.squeeze(0), 
                                        class_labels,
                                        threshold=0.005, 
                                        num_classes=test_args['num_classes']
                                    )

        if test_args['save']:
            txt_file = os.path.join(test_args['save_dir'], 'output' + '.txt')
            with open(txt_file, 'w') as f:
                # loop over instances
                for id, pred in enumerate(predictions):
                    im_name = 'output' + '_{:02d}.png'.format(id)
                    im = transforms.ToPILImage()(
                        pred['mask'].unsqueeze(0))
                    # write image
                    im.save(os.path.join(test_args['save_dir'], im_name))

                    # write to file
                    cl = 26
                    score = pred['score']
                    f.writelines("{} {} {:.02f}\n".format(im_name, cl, score))
