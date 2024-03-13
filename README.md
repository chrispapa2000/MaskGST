# Official PyTorch Implementation of MaskGST (Masked Generative Story Transformer with Character Guidance and Caption Augmentation)

![example generations for our model](/assets/example.png "example generations for our model")

## Prepare Data
Dowload the Pororo-SV dataset from the [StoryDALL-E](https://github.com/adymaharana/storydalle/tree/main?tab=readme-ov-file) and extract it under ```../data/```

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
