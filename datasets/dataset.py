import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from datasets.cityscape_data import CityscapesDataset
from datasets.synthetic_data import SyntheticDataset

def get_dataset(name, dataset_opts):
    if name == "cityscape": 
        return CityscapesDataset(**dataset_opts)
    if name == "synthetic":
        return SyntheticDataset(**dataset_opts) ## why double star
    else:
        raise RuntimeError("Dataset {} not available".format(name))
