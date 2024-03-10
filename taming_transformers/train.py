from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import yaml
import pororo_image_dataloader as data
import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
from pytorch_lightning.plugins.training_type import DDPPlugin

# class Dummy_Cnn(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)

#     def training_step(self, batch, batch_idx):
#         out = self.conv(batch)
#         criterion = torch.nn.MSELoss()
#         loss = criterion(out,batch)
#         return loss
    
#     def configure_optimizers(self):
#         # lr = self.learning_rate
#         lr = 4.5e-06
#         opt = torch.optim.Adam(self.parameters(), lr=lr)
#         return opt

def main():
    # vgg_pretrained = torchvision.models.vgg16(pretrained=True)
    # torch.save(
    #     vgg_pretrained.state_dict(),
    #     "checkpoints/vgg_pretrained/ckpt.pth"
    # )
    # print(vgg_pretrained.features, flush=True)

    config_path = 'configs/my_custom.yaml'
    config = OmegaConf.load(config_path)

    print(yaml.dump(OmegaConf.to_container(config)))

    vae = VQModel(**config.model.params)

    dataset = data.ImageDataset(
        img_folder='../../data/pororo_png/',
        mode='train',
        size=None, 
        preprocess= transforms.Compose(
            [transforms.Resize((64,64)),
            transforms.ToTensor()]
        )
    )

    eval_dataset = data.ImageDataset(
        img_folder='../../data/pororo_png/',
        mode='val',
        size=None, 
        preprocess= transforms.Compose(
            [transforms.Resize((64,64)),
            transforms.ToTensor()]
        )
    )

    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=16
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=16
    )

    trainer = pl.Trainer(accelerator='ddp', gpus=2, plugins=DDPPlugin(find_unused_parameters=True))

    trainer.fit(vae, train_dataloader, eval_dataloader)

if __name__=='__main__':
    main()