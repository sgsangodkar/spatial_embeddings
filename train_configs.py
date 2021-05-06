import copy
import os
from PIL import Image


args = dict(
    save = True,
    save_dir = 'checkpoints',
    resume_path = 'saved_models',
    display = False,
    num_classes = 6,
    model = dict(
        name = 'erfnet', # unet or erfnet
        options = dict(
            decoder_channels = [3, 6] # [2+n_sigma, num_classes]
        )
    ),
    lr = 5e-4,
    n_epochs = 200,

    # loss options
    loss_opts = dict(
        to_center = True,
        n_sigma = 1,
        foreground_weight = 10
    ),
    loss_w = dict(
        w_inst = 1,
        w_var = 10,
        w_seed = 1
    )
)

def get_train_args():
    return copy.deepcopy(args)

#a = dict(num = 9)
#def dummy2(num):
#    print(num)
#def dummy(a):
#    dummy2(a)
#    dummy2(*a)
#    dummy2(**a)
#dummy(a)
#output:
#{'num': 9}
#num
#9