from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch

class SyntheticDataset(Dataset):
    class_names = ('triangle', 'square', 'hexagon', 'circle', 'ellipse', 'pie_slice')
    class_ids = (1, 2, 3, 4, 5, 6)

    def __init__(
            self, 
            img_shape, 
            dataset_size, 
            num_objects=4,
            max_object_instances=5, 
            transform=None
        ):
        self.img_shape = img_shape
        self.dataset_size = dataset_size
        self.num_objects = num_objects
        self.max_object_instances = max_object_instances

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            #Convert a PIL Image or numpy.ndarray to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([transforms.ToTensor()])
        self.seed = 1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        sample = {}
        img, instance_mask, semantic_mask = gen_img_and_annotations(
                            self.img_shape, 
                            self.num_objects, 
                            self.max_object_instances,
                            self.seed
                    )

        if self.img_transform:
            # changes order from HWC to CHW
            sample['img'] = self.img_transform(img)
        if self.mask_transform:
            sample['instance_mask'] = torch.from_numpy(np.array(instance_mask))
            sample['semantic_mask'] = torch.from_numpy(np.array(semantic_mask))
        return sample


def gen_img_and_annotations(
        img_shape, 
        num_objects = 6, 
        max_object_instances = 1,
        seed = 0
    ):
    '''
    Function to generate synthetic images and instance annotations
    Input:
        img_shape: Required shape (width, height) of the image 
            and the corresponding mask
        num_objects: Number of unique objects in each 
            image-mask pair
        max_object_instances: Max count of each distinct object
    Output:
        Synthetic image-mask pair
    '''

    object_list = ['triangle', 'square', 'hexagon', 'circle', 'ellipse', 'pie_slice']
    cids = [1,2,3,4,5,6]
    assert num_objects <= len(object_list), 'num_objects > len(object_list)'
    #random.Random(seed).shuffle(cids)
    cids = cids[:num_objects]
    num_instances = [random.randint(1,max_object_instances) for _ in range(len(cids))]
    img, instance_mask, semantic_mask = gen_sample(img_shape, cids, num_instances)

    return img, instance_mask, semantic_mask

def gen_sample(img_shape, cids, num_instances):
    img = Image.new('RGB', img_shape, (255, 255, 255))
    img_draw = ImageDraw.Draw(img)

    instance_mask = Image.new('L', img_shape)
    instance_draw = ImageDraw.Draw(instance_mask)

    semantic_mask = Image.new('L', img_shape)
    semantic_draw = ImageDraw.Draw(semantic_mask)

    instance_id = 0
    i = 0
    for cid in cids:
        if cid == 1:
            for _ in range(num_instances[i]):
                xx,yy,s = get_location_and_size(*img_shape)
                img_draw.regular_polygon((xx,yy,s),n_sides=3,fill=(20,220,0))
                instance_id = instance_id + 1
                instance_draw.regular_polygon((xx,yy,s),n_sides=3,fill=instance_id)
                semantic_draw.regular_polygon((xx,yy,s),n_sides=3,fill=cid)

        if cid == 2:
            for _ in range(num_instances[i]):
                xx,yy,s = get_location_and_size(*img_shape)
                img_draw.rectangle((xx-s/2,yy-s/2,xx+s/2,yy+s/2), fill=(60,180,50))
                instance_id = instance_id + 1
                instance_draw.rectangle((xx-s/2,yy-s/2,xx+s/2,yy+s/2), fill=instance_id)
                semantic_draw.rectangle((xx-s/2,yy-s/2,xx+s/2,yy+s/2), fill=cid)

        if cid == 3:
            for _ in range(num_instances[i]):
                xx,yy,s = get_location_and_size(*img_shape)
                img_draw.regular_polygon((xx,yy,s),n_sides=6,fill=(100,140,100))
                instance_id = instance_id + 1
                instance_draw.regular_polygon((xx,yy,s),n_sides=6,fill=instance_id)
                semantic_draw.regular_polygon((xx,yy,s),n_sides=6,fill=cid)

        if cid == 4:
            for _ in range(num_instances[i]):
                xx,yy,s = get_location_and_size(*img_shape)
                img_draw.ellipse((xx-s/2,yy-s/2,xx+s/2,yy+s/2), fill=(140,100,150))
                instance_id = instance_id + 1
                instance_draw.ellipse((xx-s/2,yy-s/2,xx+s/2,yy+s/2), fill=instance_id)
                semantic_draw.ellipse((xx-s/2,yy-s/2,xx+s/2,yy+s/2), fill=cid)

        if cid == 5:
            for _ in range(num_instances[i]):
                xx,yy,s = get_location_and_size(*img_shape)
                img_draw.ellipse((xx,yy,xx+s*2,yy+s), fill=(180,60,200))
                instance_id = instance_id + 1
                instance_draw.ellipse((xx,yy,xx+s*2,yy+s), fill=instance_id)
                semantic_draw.ellipse((xx,yy,xx+s*2,yy+s), fill=cid)

        if cid == 6:
            for _ in range(num_instances[i]):
                xx,yy,s = get_location_and_size(*img_shape)
                img_draw.pieslice((xx-s/2,yy-s/2,xx+s/2,yy+s/2), 30, 330, fill=(220,20,250))   
                instance_id = instance_id + 1
                instance_draw.pieslice((xx-s/2,yy-s/2,xx+s/2,yy+s/2), 30, 330, fill=instance_id)
                semantic_draw.pieslice((xx-s/2,yy-s/2,xx+s/2,yy+s/2), 30, 330, fill=cid)
        i = i + 1
    return img, instance_mask, semantic_mask



def get_location_and_size(shape_x, shape_y):
    x = int(shape_x*random.uniform(0.1,0.9))
    y = int(shape_y*random.uniform(0.1,0.9))
    size = int(min(shape_x, shape_y)*random.uniform(0.06, 0.12))
    return x,y,size

if __name__ == '__main__':
    dataset = SyntheticDataset(img_shape=(128,128), dataset_size=4, num_objects = 2)
    print(dataset[0]['img'].shape)
    print(dataset[0]['instance_mask'].shape)
    print(dataset[0]['semantic_mask'].shape)
    print(np.unique(dataset[0]['instance_mask']))
