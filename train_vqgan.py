from taming_transformers.taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import yaml
import pororo_dataloader as data
import torchvision.transforms as transforms
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
from pytorch_lightning.plugins.training_type import DDPPlugin

def main(args):
    config = OmegaConf.load(args.config_path)

    print(yaml.dump(OmegaConf.to_container(config)))

    vae = VQModel(**config.model.params)

    dataset = data.StoryImageDataset(
    	img_folder='../data/pororo_png/',
    	preprocess=transforms.Compose(
        	[transforms.Resize(64),
        	transforms.ToTensor(),
        	transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]
    	),
        tokenizer=None,
    	eval_classifier=True,
    )

    eval_dataset = data.StoryImageDataset(
        img_folder='../data/pororo_png/',
        preprocess=transforms.Compose(
                [transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]
        ),
        tokenizer=None,
        eval_classifier=True,
        mode='val'
    )

    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    trainer = pl.Trainer(
        accelerator='ddp',
        gpus=args.num_gpus,
        plugins=DDPPlugin(find_unused_parameters=True),
        sync_batchnorm=True, 
        default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs,
        # resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.fit(vae, train_dataloader, eval_dataloader)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='arguments for model training')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--default_root_dir', required=True, type=str)
    parser.add_argument('--resume_from_checkpoint', type=str, default='None')
    parser.add_argument('--config_path', required=True, type=str)
    parser.add_argument('--max_epochs', required=True, type=int)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    main(args)
