# Official PyTorch Implementation of MaskGST (Masked Generative Story Transformer with Character Guidance and Caption Augmentation)

![example generations for our model](/assets/example.png "example generations for our model")

## Setup
This project was developed in `Python3.8` using PyTorch `v1.8.0`

Start by setting up a virtual environment:
```
virtualenv -p /usr/bin/python3.8 venv
source venv/bin/activate
```

Install PyTorch:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install the remaining dependencies:
```
pip install -r requirements.txt
```

## Prepare Data
Dowload the Pororo-SV dataset from [StoryDALL-E](https://github.com/adymaharana/storydalle/tree/main?tab=readme-ov-file) and extract it under ```../data/```

## Training for Pororo-SV 
### Training VQ-GAN 
```
python train_vqgan.py --default_root_dir <LIGHTNING_DIR> \
    --max_epochs <MAX_EPOCHS> \
    --config_path taming_transformers/configs/custom/f8_128.yaml
```

### Training MaskGST
```
python train_pororo.py --num_nodes <NUM_NODES> \ 
    --num_workers <NUM_WORKERS> \
    --num_gpus <GPUS_PER_NODE> \
    --default_root_dir <LIGHTNING_DIR> \ 
    --vq_vae_config taming_transformers/configs/custom/f8_128.yaml \ 
    --vq_vae_path <PRETRAINED_VQ-GAN_PATH> \
    --batch_size <BATCH_SIZE> \
```


## Inference
```
python infer.py --num_workers <NUM_WORKERS> \
    --timesteps 20 \
    --outfolder <OUTFOLDER> \
    --model_path <PRETRAINED_MASKGST_PATH>
```

## Acknowledgements
- VQ-GAN's implementation from [Taming Transformers](https://github.com/CompVis/taming-transformers) is used
- The code for the Masked Generative Transformer is adapted from this open source implementation of [MUSE](https://github.com/lucidrains/muse-maskgit-pytorch)

## Citation
```
@misc{papadimitriou2024masked,
      title={Masked Generative Story Transformer with Character Guidance and Caption Augmentation}, 
      author={Christos Papadimitriou and Giorgos Filandrianos and Maria Lymperaiou and Giorgos Stamou},
      year={2024},
      eprint={2403.08502},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
