import torch
import numpy as np
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import torchvision.transforms as transforms
import pickle
from tqdm import tqdm
# import clip
import random
      

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, preprocess, mode='train', size=None):
        # self.path_prefix = '../../diploma/'
        self.lengths = []
        self.followings = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.preprocess = preprocess
        
        # load list of image paths depending on mode (train / val / test)
        self.image_paths_list = list()
        if mode == 'train':
            with open("pickles/image_paths_list_train.pkl", 'rb') as f:
                self.image_paths_list = pickle.load(f)
            self.ids = [i for i in range(len(self.image_paths_list))]
            # self.ids = np.sort(train_ids)
        elif mode =='val':
            with open("pickles/image_paths_list_val.pkl", 'rb') as f:
                self.image_paths_list = pickle.load(f)
            self.ids = [i for i in range(len(self.image_paths_list))]
            # self.ids = np.sort(val_ids)
        elif mode =='test':
            with open("pickles/image_paths_list_test.pkl", 'rb') as f:
                self.image_paths_list = pickle.load(f)
            self.ids = [i for i in range(len(self.image_paths_list))]
            # self.ids = np.sort(test_ids)
        else:
            raise ValueError
        
        # if no size was given use the whole dataset
        if not size:
            size = len(self.ids)
        self.ids = [self.ids[i] for i in range(size)]         

    def sample_image(self, im, se=None):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        if not se:
            se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):
        # get image id
        img_id = self.ids[item]

        # get image path
        img_path = self.image_paths_list[img_id]

        image = self.sample_image(Image.open(img_path).convert('RGB'))
        # image.show()

        return self.preprocess(image)

    def __len__(self):
        return len(self.ids)
    
def main():
    train_transform = transforms.Compose(
        [transforms.Resize((64,64)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    dataset = ImageDataset(img_folder='../data/pororo_png',
                           tokenizer=1,
                           preprocess=train_transform,
                           mode='train',
                           size=100)
    for i in range(10):
        print(dataset[i])

if __name__ == "__main__":
    main()