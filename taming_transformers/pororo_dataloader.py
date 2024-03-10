import os, re
import csv
import nltk
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch.utils.data
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import json
import pickle as pkl
import pandas as pd
from transformers import PreTrainedTokenizerFast
from typing import Union
import numpy as np
from numpy.random import Generator, PCG64

def mapper(text, character_names=['pororo', 'eddy', 'crong', 'loopy', 'harry', 'petty', 'poby', 'tongtong', 'rody', 'tutu']):
    words = text.split(' ')
    res = []
    for word in words:
        res = (res + [word] + [f"{word}{i}" for i in range(7)]) if word in character_names else res + [word]
    return " ".join(res)

# from story dalle
class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            img_folder, 
            preprocess, 
            mode='train', 
            video_len=4, 
            size=None, 
            eval_classifier=False,
            seed=False
    ):
        self.lengths = []
        self.followings = []
        self.images = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.video_len = video_len
        self.eval_classifier = eval_classifier
        self.preprocess = preprocess

        if seed:
            self.rng = Generator(PCG64(seed=42))
        else:
            self.rng = Generator(PCG64())

        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)

        if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
            self.counter = ''
        else:
            counter = np.load(os.path.join(img_folder, 'frames_counter.npy'), allow_pickle=True).item()
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(self.img_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > counter[v_name] - (self.video_len-1):
                    continue
                following_imgs = []
                for i in range(self.video_len-1):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(self.img_folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(self.img_folder, 'img_cache4.npy'), self.images)
            np.save(os.path.join(self.img_folder, 'following_cache4.npy'), self.followings)

        # ids by mode
        if mode == 'train':
            self.ids = np.sort(train_ids)
        elif mode =='val':
            self.ids = np.sort(val_ids)
        elif mode =='test':
            self.ids = np.sort(test_ids)
        else:
            raise ValueError
        
        if size:
            self.ids = [self.ids[i] for i in range(size)]

    def return_random(self):
        res = self.rng.integers(0, 100, 10)
        return res           

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        # se = np.random.randint(0,video_len, 1)[0]
        se = self.rng.integers(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))
    

    def __len__(self):
        return len(self.ids)            
            
    def __getitem__(self, item):
        # for indexing
        src_img_id = self.ids[item]
        src_img_path = os.path.join(self.img_folder, str(self.images[src_img_id])[2:-1])
        tgt_img_paths = [str(self.followings[src_img_id][i])[2:-1] for i in range(self.video_len)]

        # src_img_path_id = str(src_img_path).replace(self.img_folder, '').replace('.png', '')[1:]
        src_img_path_id = str(src_img_path).replace(self.img_folder, '').replace('.png', '')[1:]  

        tgt_img_ids = [str(tgt_img_path).replace(self.img_folder, '').replace('.png', '') for tgt_img_path in tgt_img_paths]

        tgt_images = [self.preprocess(self.sample_image(Image.open(os.path.join(self.img_folder, tgt_img_path))).convert('RGB')) for tgt_img_path in tgt_img_paths]
        src_image = self.preprocess(self.sample_image(Image.open(src_img_path).convert('RGB')))
        images = [src_image] + tgt_images

        # get masks
        masks = []
        for index in [src_img_path_id] + tgt_img_ids:
            filename = index.replace('/','_')
            with open(f"pikles/character_masks/{filename}.pkl", 'rb') as f:
                mask = pickle.load(f)
                masks.append(torch.tensor(mask))
        masks = torch.cat(masks)

        # when evaluating with character classifier
        if self.eval_classifier:
            labels = [self.labels[index] for index in [src_img_path_id]+tgt_img_ids]
            # print(f"labels: {torch.tensor(np.vstack(labels)).shape}")
            return torch.stack(images), torch.tensor(np.vstack(labels)), masks

        return torch.stack(images)
        
    
def main():
    dataset = StoryImageDataset(img_folder='../data/pororo_png',
                                tokenizer=None,
                                preprocess=transforms.Compose(
                                    [transforms.Resize((64,64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                ))
    # print(dataset.descriptions.keys())
    item = dataset[10]
    print(f"images : {item[0].shape}, text embeddings : {item[1].shape}")

if __name__ == "__main__":
    main()                      
                                
