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
import re
import random


def mapper(text, character_names=['pororo', 'eddy', 'crong', 'loopy', 'harry', 'petty', 'poby', 'tongtong', 'rody', 'tutu']):
    words = text.split(' ')
    res = []
    for word in words:
        res = (res + [word] + [f"{word}{i}" for i in range(7)]) if word in character_names else res + [word]
    return " ".join(res)

def emoji_filter(caption):
    emojis = ["🌧", "🏠", "🍞", "🍪", "🛷", "🌙", "✨", "🎶", "🦖", "🐧", "🐻", "🚀", "😊", "🕺", "🍽", "🖖", "⚽", "🍳", "🥪", "⭐"]
    pattern = ""
    for emoji in emojis:
        pattern += emoji + "|"
    pattern = pattern[:-1]
    caption = re.sub(pattern, "", caption)
    return caption

def get_background_concepts(caption):
    import re
    import numpy as np

    concepts = [["house"], ["door"], ["table"], ["car"], ["snow"], ["sky"], ["tree", "forest", "woods"]]

    caption = re.split(pattern=",|\.| ", string=caption.strip().lower())

    res = []
    for concept in concepts:
        is_present = 0.
        for alt in concept:
            for word in caption:
                if alt in word:
                    is_present = 1.
        res.append(is_present)
    
    return np.array(res)

# from story dalle
class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            img_folder, 
            tokenizer:Union[PreTrainedTokenizerFast,int], 
            preprocess, 
            mode='train', 
            video_len=4, 
            out_img_folder=None, 
            return_labels=False, 
            size=None, 
            return_captions=False, 
            return_tokenized_captions=False,
            character_names=['pororo', 'eddy', 'crong', 'loopy', 'harry', 'petty', 'poby', 'tongtong', 'rody', 'tutu'],
            max_sentence_length=100,
            text_encoder=None,
            eval_classifier=False,
            seed=False,
            character_emphasis=False,
            return_token_indices=False,
            use_chatgpt_captions=False
    ):
        self.lengths = []
        self.followings = []
        self.images = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.video_len = video_len
        self.descriptions_original = np.load(os.path.join(img_folder, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
        # self.descriptions = np.load(os.path.join(img_folder, 'descriptions_vec.npy'), allow_pickle=True, encoding='latin1').item() # used in the eccv camera-ready version
        self.descriptions = pkl.load(open(os.path.join(img_folder, 'descriptions_vec_512.pkl'), 'rb'))
        self.return_captions = return_captions
        self.return_tokenized_captions = return_tokenized_captions
        self.character_names = character_names
        self.eval_classifier = eval_classifier
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.return_labels = return_labels
        self.out_img_folder = out_img_folder
        self.return_token_indices = return_token_indices
        self.use_chatgpt_captions = use_chatgpt_captions

        if seed:
            self.rng = Generator(PCG64(seed=42))
        else:
            self.rng = Generator(PCG64())

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

        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)

        if os.path.exists("pickles/my_description_dict.pkl"):
            with open("pickles/my_description_dict.pkl",'rb') as f:
                self.my_description_dict = pickle.load(f)
        else:
            self.my_description_dict = self.get_source_image_descriptions() 

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

        #load embeddings
        if text_encoder != None:
            with open(f"pickles/preloaded_captions_{text_encoder}_{mode}", 'rb') as f:
                self.preloaded_text_embeddings = pickle.load(f)

        # with open(f"pikles/preloaded_captions_{mode}.pkl", 'rb') as f:
        #     self.preloaded_text_embeddings = pickle.load(f)

        # with open(f"pikles/image_paths_list_{mode}.pkl", 'rb') as f:
        #         image_paths_list = pickle.load(f)

        # self.image_paths_dict = {}
        # for i in range(len(image_paths_list)):
        #     self.image_paths_dict[image_paths_list[i]] = i

        # self.text_encoder = 'clip'
        # if text_encoder == 't5':
        #     self.text_encoder = 't5'
        #     with open(f"pikles/preloaded_captions_t5_{mode}", 'rb') as f:
        #         self.t5_embeddings = pickle.load(f)

        # if return_tokenized_captions:
        #     # pretokenize text
        #     all_ids, all_captions = [], []
        #     for item in tqdm(range(len(self.ids))):
        #         src_img_id = self.ids[item]

        #         src_img_path = os.path.join(self.img_folder, str(self.images[src_img_id])[2:-1])
        #         tgt_img_paths = [str(self.followings[src_img_id][i])[2:-1] for i in range(self.video_len)]
        #         # src_img_path_id = str(src_img_path).replace(self.img_folder, '').replace('.png', '')[1:]
        #         src_img_path_id = str(src_img_path).replace(self.img_folder, '').replace('.png', '')  

        #         tgt_img_ids = [str(tgt_img_path).replace(self.img_folder, '').replace('.png', '') for tgt_img_path in tgt_img_paths]

        #         image_captions = [self.descriptions_original[src_img_path_id]]+[self.descriptions_original[other_id] for other_id in tgt_img_ids]
        #         image_captions = [desc[0].lower() for desc in image_captions]

        #         if character_emphasis:
        #             image_captions = map(mapper, image_captions)

        #         all_ids += [src_img_path_id] + tgt_img_ids
        #         all_captions += image_captions
            
        #     caption_ids = self.tokenizer(all_captions, padding=True, return_tensors='pt', truncation=True, max_length=max_sentence_length).input_ids
        #     print(f"caption ids shape {caption_ids[0].shape}")
        #     self.caption_ids_dict = dict()
        #     for i in range(len(caption_ids)):
        #         self.caption_ids_dict[all_ids[i]] = caption_ids[i]

        # load captions if needed
        if return_tokenized_captions:
            # pretokenize text

            with open('pickles/chagpt_captions_dict.pkl', 'rb') as f:
                chagpt_captions_dict = pickle.load(file=f)

            all_ids, all_captions, chatgpt_captions = [], [], []
            for item in tqdm(range(len(self.ids))):
                src_img_id = self.ids[item]

                src_img_path = os.path.join(self.img_folder, str(self.images[src_img_id])[2:-1])
                tgt_img_paths = [str(self.followings[src_img_id][i])[2:-1] for i in range(self.video_len)]
                # src_img_path_id = str(src_img_path).replace(self.img_folder, '').replace('.png', '')[1:]
                src_img_path_id = str(src_img_path).replace(self.img_folder, '').replace('.png', '')  

                tgt_img_ids = [str(tgt_img_path).replace(self.img_folder, '').replace('.png', '') for tgt_img_path in tgt_img_paths]

                image_captions = [self.descriptions_original[src_img_path_id]]+[self.descriptions_original[other_id] for other_id in tgt_img_ids]
                image_captions = [desc[0].lower() for desc in image_captions]

                if character_emphasis:
                    image_captions = map(mapper, image_captions)

                all_ids += [src_img_path_id] + tgt_img_ids
                all_captions += image_captions
                chatgpt_captions += [emoji_filter(chagpt_captions_dict[img_id.replace('/','_')].lower()) for img_id in [src_img_path_id]+tgt_img_ids]
            print(len(all_captions), len(chatgpt_captions))
            caption_ids = self.tokenizer(all_captions+chatgpt_captions, padding=True, return_tensors='pt', truncation=True, max_length=max_sentence_length).input_ids
            print(f"caption ids shape {caption_ids[0].shape}")
            self.caption_ids_dict = dict()
            self.chatgpt_ids_dict = dict()
            for i in range(len(caption_ids)//2):
                self.caption_ids_dict[all_ids[i]] = caption_ids[i]
                self.chatgpt_ids_dict[all_ids[i]] = caption_ids[i+len(caption_ids)//2]
        
        # background concepts 
        self.concept_labels = dict()
        for item in tqdm(range(len(self.ids))):
            src_img_id = self.ids[item]

            src_img_path = os.path.join(self.img_folder, str(self.images[src_img_id])[2:-1])
            tgt_img_paths = [str(self.followings[src_img_id][i])[2:-1] for i in range(self.video_len)]
            # src_img_path_id = str(src_img_path).replace(self.img_folder, '').replace('.png', '')[1:]
            src_img_path_id = str(src_img_path).replace(self.img_folder, '').replace('.png', '')  

            tgt_img_ids = [str(tgt_img_path).replace(self.img_folder, '').replace('.png', '') for tgt_img_path in tgt_img_paths]
            self.concept_labels[src_img_path_id] = get_background_concepts(self.descriptions_original[src_img_path_id][0])
            for other_id in tgt_img_ids:
                self.concept_labels[other_id] = get_background_concepts(self.descriptions_original[other_id][0])
    
    def return_random(self):
        res = self.rng.integers(0, 100, 10)
        return res           

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        # se = np.random.randint(0,video_len, 1)[0]
        se = self.rng.integers(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))
    
    def sample_caption(self, img_id):
        r = random.randint(0,1)
        if r==0:
            return self.caption_ids_dict[img_id]
        return self.chatgpt_ids_dict[img_id]
    
    def get_source_image_descriptions(self):
        file = open('../data/pororo_png/descriptions.csv','r')
        text = [line.replace('\"', '') for line in file.readlines()]
        file.close()
        description_dict = dict()
        for line in text:
            episode, frame_number, desc = line.split(',',maxsplit=2)
            # handle multiple discriptions for the same photo
            key = f"{episode}_{frame_number}"
            if key in description_dict.keys():
                current_list = description_dict[key]
                current_list.append(desc)
                description_dict[key] = current_list
            else:
                description_dict[key] = [desc]
        return description_dict

    def __len__(self):
        return len(self.ids)            
            
    def __getitem__(self, item):
        # for indexing
        src_img_id = self.ids[item]
        src_img_path = os.path.join(self.img_folder, str(self.images[src_img_id])[2:-1])
        tgt_img_paths = [str(self.followings[src_img_id][i])[2:-1] for i in range(self.video_len)]

        # src_img_path_id = str(src_img_path).replace(self.img_folder, '').replace('.png', '')[1:]
        src_img_path_id = str(src_img_path).replace(self.img_folder, '').replace('.png', '')  

        tgt_img_ids = [str(tgt_img_path).replace(self.img_folder, '').replace('.png', '') for tgt_img_path in tgt_img_paths]

        # get images 
        if self.out_img_folder != None:
            tgt_images = [self.preprocess(Image.open(os.path.join(self.out_img_folder, 'gen_%s_%s.png' % (item, frame_idx))).convert('RGB')) for frame_idx in range(1,5)]
            src_image = self.preprocess(Image.open(os.path.join(self.out_img_folder, 'gen_%s_%s.png' % (item, 0))).convert('RGB'))
            # tgt_images = [self.preprocess(Image.open(os.path.join(self.out_img_folder, 'img-%s-%s.png' % (item, frame_idx))).convert('RGB')) for frame_idx in range(1,5)]
            # src_image = self.preprocess(Image.open(os.path.join(self.out_img_folder, 'img-%s-%s.png' % (item, 0))).convert('RGB'))
        else:
            tgt_images = [self.preprocess(self.sample_image(Image.open(os.path.join(self.img_folder, tgt_img_path))).convert('RGB')) for tgt_img_path in tgt_img_paths]
            src_image = self.preprocess(self.sample_image(Image.open(src_img_path).convert('RGB')))
        images = [src_image] + tgt_images

        # # get masks
        # masks = []
        # for index in [src_img_path_id] + tgt_img_ids:
        #     filename = index.replace('/','_')
        #     with open(f"pikles/character_masks/{filename}.pkl", 'rb') as f:
        #         mask = pickle.load(f)
        #         masks.append(torch.tensor(mask))
        # masks = torch.cat(masks)
        masks = torch.tensor(1)
        
        # when evaluating with character classifier
        if self.eval_classifier:
            labels = [self.labels[index] for index in [src_img_path_id]+tgt_img_ids]
            if self.return_captions:
                # get captions
                image_captions = [self.descriptions_original[src_img_path_id]]+[self.descriptions_original[other_id] for other_id in tgt_img_ids]
                image_captions = [desc[0].lower() for desc in image_captions]
                return torch.stack(images), torch.tensor(np.vstack(labels)), masks, image_captions
            return torch.stack([src_image]+tgt_images), torch.tensor(np.vstack(labels))

        # get tokenized captions or embeddings
        if self.return_tokenized_captions:
            if self.use_chatgpt_captions:
                text_embeddings = [self.sample_caption(img_id) for img_id in [src_img_path_id]+tgt_img_ids]
            else:
                text_embeddings = [self.caption_ids_dict[src_img_path_id]] + [self.caption_ids_dict[i] for i in tgt_img_ids]
        else:
            text_embeddings = [self.preloaded_text_embeddings[src_img_path_id]] + [self.preloaded_text_embeddings[tgt_id] for tgt_id in tgt_img_ids]
        
        token_inds = None
        if self.return_token_indices:
            with open(os.path.join('pickles','image_tokens',f"{src_img_path_id}.pkl"), 'rb') as f:
                token_inds = pickle.load(f)

        # labels
        labels = [self.labels[index] for index in [src_img_path_id]+tgt_img_ids] 
        labels = torch.tensor(np.vstack(labels))

        # concept labels
        concept_labels = [self.concept_labels[index] for index in [src_img_path_id]+tgt_img_ids] 
        concept_labels = torch.tensor(np.vstack(concept_labels))


        if self.return_captions:
            # get captions
            image_captions = [self.descriptions_original[src_img_path_id]]+[self.descriptions_original[other_id] for other_id in tgt_img_ids]
            image_captions = [desc[0].lower() for desc in image_captions]
            if self.return_token_indices:
                return torch.stack(images), torch.stack(text_embeddings), token_inds, masks, image_captions 
            else:
                return torch.stack(images), torch.stack(text_embeddings), labels, concept_labels, masks, image_captions
        
        if self.return_token_indices:
            return torch.stack(images), torch.stack(text_embeddings), token_inds, labels, concept_labels, masks
        else:
            return torch.stack(images), torch.stack(text_embeddings), labels, concept_labels, masks


      
        
    
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
                                