import copy
import os
from PIL import Image


args = dict(
    save = True,
    save_dir = 'outputs',
    checkpoints = 'checkpoints/best_model.pth',
    resume_path = 'saved_models',
    display = True,
    num_classes = 6,
    model = dict(
        name = 'erfnet', # unet or erfnet
        options = dict(
            decoder_channels = [3, 6] # [2+n_sigma, num_classes]
        )
    )
)

def get_test_args():
    return copy.deepcopy(args)
