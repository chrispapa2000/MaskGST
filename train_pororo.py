import argparse
from maskgst.maskGST import maskGST
from omegaconf import OmegaConf
import yaml
import torch
from pytorch_lightning import Trainer
import pororo_dataloader 
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.plugins.training_type import DDPPlugin
from transformers import PreTrainedTokenizerFast

def load_vqgan(config_path, vq_vae_path):
    from taming_transformers.taming.models.vqgan import VQModel
    config = OmegaConf.load(config_path)
    print(yaml.dump(OmegaConf.to_container(config)), flush=True)
    vae = VQModel(**config.model.params)
    sd = torch.load(vq_vae_path, map_location="cpu")["state_dict"]
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    return vae

def main(args):
    print(f"available gpus: {torch.cuda.device_count()}", flush=True)

    vq_vae = load_vqgan(config_path=args.vq_vae_config, vq_vae_path=args.vq_vae_path)
   
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_path, 
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    vocab_size = len(wrapped_tokenizer.get_vocab())


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
        train_text_embeddings=True,
        vocab_size=vocab_size,
        max_story_length=args.max_story_length,
        transformer_type=args.transformer_type,
    )

    print(f"model: {model}", flush=True)

    # prepare dataset(s) and Dataloaders
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    train_dataset = pororo_dataloader.StoryImageDataset(img_folder='../data/pororo_png/', tokenizer=wrapped_tokenizer, preprocess=train_transform, mode='train', return_tokenized_captions=True, text_encoder=args.text_encoder, character_emphasis=False,    max_sentence_length=None, use_chatgpt_captions=args.use_chatgpt_captions)
    eval_dataset = pororo_dataloader.StoryImageDataset(img_folder='../data/pororo_png/', tokenizer=wrapped_tokenizer, preprocess=train_transform, mode='val', return_tokenized_captions=True, text_encoder=args.text_encoder, character_emphasis=False, max_sentence_length=args.max_story_length//5, use_chatgpt_captions=args.use_chatgpt_captions)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"train dataset size : {len(train_dataset)}, eval dataset size: {len(eval_dataset)}", flush=True)  #, eval dataset size : {len(eval_dataset)}", flush=True)


    # prepare trainer
    trainer = Trainer(gpus=args.num_gpus, 
                      accelerator='ddp',  
                      default_root_dir=args.default_root_dir, 
                    #   resume_from_checkpoint=args.resume_from_checkpoint, 
                      max_epochs=args.max_epochs
    )

    # train
    trainer.fit(model, train_dataloader, eval_dataloader)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='arguments for model training')
    parser.add_argument('--num_nodes', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--num_gpus', type=int)
    parser.add_argument('--default_root_dir', type=str, required=True)
    parser.add_argument('--resume_from_checkpoint', type=str, default='None')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer/tokenizer_vocab2500_aug.json')
    parser.add_argument('--vq_vae_config', type=str, default='taming_transformers/configs/custom/f8_128.yaml')
    parser.add_argument('--vq_vae_path', type=str, required=True)   
    parser.add_argument('--batch_size', type=int, default=6)

    parser.add_argument('--max_story_length', type=int, default=345)
    parser.add_argument('--num_embeddings', type=int, default=128)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--num_transformer_heads', type=int, default=8)
    parser.add_argument('--latent_image_size', type=int, default=64)
    parser.add_argument('--text_dim', type=int, default=2048)
    parser.add_argument('--input_dim', type=int, default=2048, help="dimension of the transformer")
    parser.add_argument('--output_dim', type=int, default=128, help="number of tokens in the vqvae")
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--no_mask_token_prob', type=float, default=0.25)
    parser.add_argument('--vqvae_mode', type=str, default='ddp')
    parser.add_argument('--text_encoder', type=str, default='None')
    parser.add_argument('--max_epochs', type=int, default=250)
    parser.add_argument('--transformer_type', type=str, default='baseline')
    parser.add_argument('--use_chatgpt_captions', type=str, default='True')


    args = parser.parse_args()
    args.resume_from_checkpoint = None if args.resume_from_checkpoint == 'None' else args.resume_from_checkpoint 
    args.text_encoder = None if args.text_encoder == 'None' else args.text_encoder
    args.batch_size = args.batch_size // args.num_gpus
    args.use_chatgpt_captions = True if args.use_chatgpt_captions=='True' else False
    print(f"batch size per gpu : {args.batch_size}", flush=True)
    main(args)
