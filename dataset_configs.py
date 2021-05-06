import copy
import os
CITYSCAPES_DIR = ''

args = dict(
    name = 'synthetic', # or cityscape
    train_dataset = dict(
            synthetic = dict(
                img_shape = (128,128),
                dataset_size = 32,
                num_objects = 6,
                max_object_instances = 5
            ),
            cityscape = dict(
                root_dir = CITYSCAPES_DIR,
                type = 'crops',
                size = 3000
            ),
            batch_size = 2,
            workers = 1
    ),

    val_dataset = dict(
            synthetic = dict(
                img_shape = (128,128),
                dataset_size = 8,
                num_objects = 6,
                max_object_instances = 5
            ),
            cityscape = dict(
                root_dir = CITYSCAPES_DIR,
                type = 'crops',
                size = 3000
            ),
            batch_size = 4,
            workers = 1
    ),

    test_dataset = dict(
            synthetic = dict(
                img_shape = (128,128),
                dataset_size = 8,
                num_objects = 6,
                max_object_instances = 5
            ),
            cityscape = dict(
                root_dir = CITYSCAPES_DIR,
                type = 'crops',
                size = 3000
            ),
            batch_size = 4,
            workers = 1
    )
)

def get_dataset_args():
    return copy.deepcopy(args)