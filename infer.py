import argparse
from maskgst.maskGST import maskGST
from omegaconf import OmegaConf
import yaml
import torch
import pororo_dataloader 
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from torchvision.utils import save_image
from tqdm import tqdm
import os
import numpy as np

def load_vqgan(config_path):
    from taming_transformers.taming.models.vqgan import VQModel
    config = OmegaConf.load(config_path)
    print(yaml.dump(OmegaConf.to_container(config)), flush=True)
    vae = VQModel(**config.model.params)
    return vae

def infer(model, dataset, batch_size, num_workers, timesteps, outfolder, fmap_size, train_text_embeddings=True,cg_scale=0.2):
    model.eval()
    inv_transform = transforms.Normalize(mean = [ -1, -1, -1 ],std = [ 1/0.5, 1/0.5, 1/0.5 ])

    os.makedirs(f"{outfolder}/timesteps_{timesteps}", exist_ok=True)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    with torch.no_grad():
        for item, (images, text_embeddings, character_labels, concept_labels, _) in enumerate(tqdm(dataloader)):
            if not train_text_embeddings:
                text_embeddings = text_embeddings.squeeze(2)
            
            text_embeddings = text_embeddings.to('cuda')
            character_labels=character_labels.to('cuda')
            concept_labels = concept_labels.to('cuda')

            generated_stories = model.generate_story(
                loop=None,
                story_text_inputs=text_embeddings, 
                timesteps=timesteps, 
                can_remask_prev_masked=True,
                fmap_size=fmap_size,
                character_labels=character_labels,
                cg_scale=cg_scale
            )  
            generated_stories = generated_stories.split(5)
            for index in range(len(generated_stories)):
                generated_story = generated_stories[index]
                images = [inv_transform(generated_story[i]) for i in range(5)]

                for i in range(5):
                    save_image(images[i], fp=f"{outfolder}/timesteps_{timesteps}/gen_{item*batch_size+index}_{i}.png")
                
            


    

def main(args):
    # for reproducibility 1 3 6 7 8 9
    torch.manual_seed(10)
    
    vq_vae = load_vqgan(config_path=args.vq_vae_config)
      
    if args.train_text_embeddings:
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=args.tokenizer_path, 
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        vocab_size = len(wrapped_tokenizer.get_vocab())
    else:
        wrapped_tokenizer, vocab_size = None, None

    # prepare maskGST model
    model = maskGST(
        # universal arguments  
        vq_vae,
        num_embeddings=args.num_embeddings, 
        num_transformer_layers=args.num_transformer_layers, 
        num_transformer_heads=args.num_transformer_heads,
        latent_image_size=args.latent_image_size, 
        text_dim=args.text_dim, 
        input_dim=args.input_dim, 
        output_dim=args.output_dim, 
        no_mask_token_prob=0.25,
        # story length
        images_per_story=5,
        train_text_embeddings=args.train_text_embeddings,
        vocab_size=vocab_size,
        max_story_length=args.max_story_length,
        transformer_type=args.transformer_type,
        # story_first=args.story_first,
        # story_last=args.story_last,
    )

    state_dict = torch.load(args.model_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict=state_dict['state_dict'], strict=False)
    print(f"missing: {missing}\nunexpected: {unexpected}", flush=True)  
    model = model.to('cuda')

    # print(f"model: {model}", flush=True)

    # prepare dataset(s) and Dataloaders
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.image_size),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    test_dataset = pororo_dataloader.StoryImageDataset(
        img_folder='../data/pororo_png/',
        tokenizer=wrapped_tokenizer, 
        preprocess=train_transform, 
        mode='test', 
        return_tokenized_captions=args.train_text_embeddings,
        text_encoder=args.text_encoder,
        # character_emphasis=args.character_emphasis, 
        max_sentence_length=args.dataset_max_sentence_length
    )

    print(f"test dataset size : {len(test_dataset)}", flush=True) 

    infer(
        model=model, 
        dataset=test_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        timesteps=args.timesteps, 
        outfolder=args.outfolder,
        fmap_size=args.fmap_size,
        train_text_embeddings=args.train_text_embeddings,
        cg_scale=args.cg_scale
    )




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='arguments for model training')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer/tokenizer_vocab2500_aug.json')
    parser.add_argument('--vq_vae_config', type=str, default='taming_transformers/configs/custom/f8_128.yaml')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--num_embeddings', type=int, default=128)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--num_transformer_heads', type=int, default=8)
    parser.add_argument('--latent_image_size', type=int, default=64)
    parser.add_argument('--text_dim', type=int, default=2048)
    parser.add_argument('--input_dim', type=int, default=2048, help="dimension of the transformer")
    parser.add_argument('--output_dim', type=int, default=128, help="number of tokens in the vqvae")
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--no_mask_token_prob', type=float, default=0.25)
    parser.add_argument('--train_text_embeddings', type=str, default="True")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--timesteps', type=int, required=True)
    parser.add_argument('--outfolder', type=str, required=True)
    parser.add_argument('--max_story_length', type=int, default=345)
    # parser.add_argument('--character_emphasis', required=True, type=str)
    parser.add_argument('--dataset_max_sentence_length', type=int, default=69)
    parser.add_argument('--fmap_size', type=int, default=8)
    parser.add_argument('--text_encoder', type=str, default='None')
    parser.add_argument('--transformer_type', type=str, default='baseline')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cg_scale', type=float, required=False, default=0.2)

    
    args = parser.parse_args()
    args.train_text_embeddings = True if args.train_text_embeddings == 'True' else False
    # args.character_emphasis = True if args.character_emphasis == 'True' else False
    args.dataset_max_sentence_length = None if args.dataset_max_sentence_length < 0 else args.dataset_max_sentence_length
    args.text_encoder = None if args.text_encoder == 'None' else args.text_encoder
    main(args)
