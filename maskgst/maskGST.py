import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import math
import pytorch_lightning as pl
from typing import Optional
from random import random
from torch import einsum
from typing import  List
from einops import rearrange, repeat
import os
from torchvision.transforms import transforms
import numpy as np

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return gate * F.gelu(x)

def FeedForward(dim, mult = 4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        cross_attend = False,
        scale = 8
    ):
        super().__init__()
        self.scale = scale
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        assert not (exists(context) ^ self.cross_attend)

        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            context_mask = F.pad(context_mask, (1, 0), value = True)

            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~context_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        layer_cross_attention = [True, True, False, False, False, False]
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
 
        for _ in range(max(depth-len(layer_cross_attention), 0)):
            layer_cross_attention.append(False)

        for i in range(depth):
            if layer_cross_attention[i]:
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, dim_head = dim_head, heads = heads),
                    Attention(dim = dim, dim_head = dim_head, heads = heads, cross_attend = True),
                    FeedForward(dim = dim, mult = ff_mult)
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, dim_head = dim_head, heads = heads),
                    FeedForward(dim = dim, mult = ff_mult)
                ]))

        self.norm = LayerNorm(dim)

        self.layer_cross_attention = layer_cross_attention

    def forward(self, x, context = None, context_mask = None):
        for i, block in enumerate(self.layers):
            if self.layer_cross_attention[i]:
                attn, cross_attn, ff = block 

                x = attn(x) + x
                x = cross_attn(x, context = context, context_mask = context_mask) + x
                x = ff(x) + x

            else:
                attn, ff = block 

                x = attn(x) + x
                x = ff(x) + x

        return self.norm(x)

# transformer - it's all we need

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        dim_out = None,
        add_mask_id = False,
        text_embed_dim = 512,
        heads=8,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len

        self.transformer_blocks = TransformerBlocks(dim = dim, heads=heads, **kwargs)
        # self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)

        # text conditioning
        self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias = False) if text_embed_dim != dim else nn.Identity() 

    def forward_with_cond_scale(
        self,
        *args,
        return_embed = False,
        **kwargs
    ):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        character_embeds: Optional[torch.Tensor] = None,
        conditioning_embeddings: Optional[torch.Tensor] = None,
        conditioning_embeddings_mask: Optional[torch.Tensor] = None,
    ):
        device, b, n = x.device, *x.shape
        assert n <= self.seq_len

        # prepare texts
        assert exists(texts) ^ exists(text_embeds)

        if exists(texts):
            text_embeds = self.encode_text(texts)
        
        # concat conditioning embeddings if needed (character embeddings)
        if exists(conditioning_embeddings):
            assert exists(conditioning_embeddings_mask)
            # context = torch.cat((context, conditioning_embeddings), dim=-2)
            # context_mask = torch.cat((context_mask, conditioning_embeddings_mask), dim=-1)
            context = conditioning_embeddings
            context_mask = conditioning_embeddings_mask
        else:
            context, context_mask = None, None
 
        # embed tokens

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device = device))
        x = torch.cat([x,self.text_embed_proj(text_embeds), character_embeds], dim=-2)

        embed = self.transformer_blocks(x, context = context, context_mask = context_mask)
        return embed 

  
class Character_Embeddings(nn.Module):
    def __init__(
        self, 
        n_chars=9,
        dim=1024,
        mult_dim=8,     
    ):
        super().__init__()
        self.character_embeddings = nn.Embedding(num_embeddings=n_chars, embedding_dim=dim*mult_dim)
        self.not_character_embeddings = nn.Embedding(num_embeddings=n_chars, embedding_dim=dim*mult_dim)
        self.mult_dim = mult_dim
        self.character_seperator = nn.Embedding(num_embeddings=n_chars, embedding_dim=dim)
        self.n_chars = n_chars
    
    def forward(self,character_labels: torch.Tensor):
        b, device = character_labels.shape[0],  character_labels.device
        
        # prepare character context
        character_indices = torch.arange(self.n_chars, device=device).unsqueeze(0)
        character_indices = character_indices.repeat_interleave(repeats=b,dim=0)
        
        character_embs = self.character_embeddings(character_indices)
        not_character_embs = self.not_character_embeddings(character_indices)
        character_embs = torch.where(character_labels.unsqueeze(-1).expand(character_embs.shape)==1, character_embs, not_character_embs)
        character_embs = rearrange(character_embs, '... (mult_dim dim) -> ... mult_dim dim', mult_dim=self.mult_dim)
        
        seperators = self.character_seperator(torch.tensor([[i for i in range(self.n_chars)] for _ in range(b)], device=device)).unsqueeze(-2).repeat_interleave(repeats=self.mult_dim, dim=-2)
        character_embs = character_embs + seperators
        character_embs = rearrange(character_embs, 'b n_char n_tok ...-> b (n_char n_tok) ...')
       
        return character_embs


# classifier free guidance functions

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device = None):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return uniform(shape, device = device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

# mask utilities from: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/muse_maskgit_pytorch.py
def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(min, max)

# noise schedules
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

# maskGST class
class maskGST(pl.LightningModule):
    def __init__(
            self,
            # universal arguments  
            vq_vae,
            num_embeddings=64, 
            num_transformer_layers=6, 
            num_transformer_heads=8,
            latent_image_size=64, 
            text_dim=2048, 
            input_dim=2048, 
            output_dim=64, 
            noise_schedule=cosine_schedule,
            no_mask_token_prob=0.25,
            images_per_story=5, 
            train_text_embeddings=False,
            vocab_size=None,
            max_story_length=345, # maximum number of tokens in a story description
            transformer_type:Optional[str]='baseline',
            n_chars=9,
    ):
        super(maskGST, self).__init__()
        self.validation_step_outputs = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_embeddings = num_embeddings+1
        self.mask_id = num_embeddings
        self.noise_schedule = noise_schedule
        self.no_mask_token_prob = no_mask_token_prob
        self.text_dim=text_dim
        self.max_story_length=max_story_length
        self.images_per_story = images_per_story
        self.latent_image_size = latent_image_size
        self.transformer_type=transformer_type
        self.n_chars=n_chars    

        self.vq_vae = vq_vae
        for param in vq_vae.parameters():
            param.requires_grad = False
        
        transformer_out_dim = output_dim
        self.to_logits = torch.nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)

        # for Character Guidance
        self.character_embeddings = Character_Embeddings(dim=input_dim,n_chars=n_chars)
        self.null_condition = torch.nn.Embedding(1,embedding_dim=input_dim)

        self.transformer = Transformer(
            num_tokens=num_embeddings,
            dim=input_dim,
            seq_len=latent_image_size,
            dim_out=transformer_out_dim,
            add_mask_id=True,
            depth=num_transformer_layers,
            text_embed_dim=text_dim,
            heads=num_transformer_heads, # number of transformer heads
        )

        self.train_text_embeddings = False
        if train_text_embeddings:
            self.train_text_embeddings = True
            self.vocab_size=vocab_size
            self.text_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=text_dim)
        
        self.pos_text_embeddings= nn.Embedding(num_embeddings=max_story_length, embedding_dim=text_dim)
        
        self.images_per_story = images_per_story

    def get_masked_indices(self, indices, batch, seq_len, prob=None):
        # prepare mask
        rand_time = uniform((batch,), device = self.device)
        rand_mask_probs = self.noise_schedule(rand_time, )
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min = 1)

        mask_id = self.mask_id
        batch_randperm = torch.rand((batch, seq_len), device = self.device).argsort(dim = -1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        x = torch.where(mask, mask_id, indices)
        return x, mask
    
    def set_device(self, device):
        self.device = device
    
    @torch.no_grad()
    def decode_from_ids(self, ids, labels, fmap_size=16):
        device = ids.device
        ids = ids.flatten()
        codebook_entries = self.vq_vae.quantize.get_codebook_entry(indices=ids, shape=(-1,fmap_size,fmap_size,256))
        recon = self.vq_vae.decode(codebook_entries)
        return recon

    @torch.no_grad()
    def generate_story(
        self, 
        story_text_inputs, 
        character_labels:Optional[torch.Tensor]=None,
        fmap_size = 16,
        temperature = 1.,
        topk_filter_thres = 0.9,
        can_remask_prev_masked = False,
        timesteps = 18,  # ideal number of steps is 18 in maskgit paper
        loop=None,
        cg_scale=0.2
    ):
        # get the universal story encoding
        story_text_embedding = self.encode_story(text_inputs=story_text_inputs)
        
        # reshape the text inputs : (b x n_images) x 1 x text_dim 
        if self.train_text_embeddings:
            batch, n_texts, _= story_text_inputs.shape
            text_inputs = story_text_inputs.view(batch*n_texts, -1)
        else:
            batch, n_texts, _, _ = story_text_inputs.shape
            text_inputs = rearrange(story_text_inputs, 'b n_images n_tokens dim -> (b n_images) n_tokens dim')
        
        if exists(character_labels):
            character_labels = rearrange(character_labels, 'b n_images ...-> (b n_images) ...')

        return self.generate_image(
            texts=text_inputs,
            story_text_embedding=story_text_embedding,
            character_labels=character_labels,
            fmap_size = fmap_size,
            temperature = temperature,
            topk_filter_thres = topk_filter_thres,
            can_remask_prev_masked = can_remask_prev_masked,
            timesteps = timesteps,  
            loop=loop,
            cg_scale=cg_scale
        )
       
    @torch.no_grad()
    def generate_image(
        self,
        texts,
        story_text_embedding:Optional[torch.Tensor]=None, 
        character_labels:Optional[torch.Tensor]=None,
        fmap_size = 16,
        temperature = 1.,
        topk_filter_thres = 0.9,
        can_remask_prev_masked = False,
        timesteps = 18,  # ideal number of steps is 18 in maskgit paper
        loop=None,
        cg_scale=0.2
    ):
        
        sampled_indices = self.generate(
            texts=texts,
            story_text_embedding=story_text_embedding,
            character_labels=character_labels,
            fmap_size = fmap_size,
            temperature = temperature,
            topk_filter_thres = topk_filter_thres,
            can_remask_prev_masked = can_remask_prev_masked,
            timesteps = timesteps,
            loop=loop,
            cg_scale=cg_scale
            )

        return self.decode_from_ids(sampled_indices, character_labels, fmap_size)
    
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        texts,
        story_text_embedding:Optional[torch.Tensor]=None, 
        character_labels:Optional[torch.Tensor]=None,
        fmap_size = 16,
        temperature = 1.,
        topk_filter_thres = 0.9,
        can_remask_prev_masked = False,
        timesteps = 18,  # ideal number of steps is 18 in maskgit paper
        loop=None,
        cg_scale=0.2
    ):
        # begin with all image token ids masked
        device = next(self.parameters()).device

        seq_len = fmap_size ** 2

        batch_size = texts.shape[0]

        shape = (batch_size, seq_len)

        ids = torch.full(shape, self.mask_id, dtype = torch.long, device = device)
        scores = torch.zeros(shape, dtype = torch.float32, device = device)

        starting_temperature = temperature

        if self.train_text_embeddings:
            text_inputs = self.text_embeddings(texts)
            n_words = text_inputs.shape[1]
            text_inputs = text_inputs + self.pos_text_embeddings(torch.arange(n_words, device=self.device))
            text_embeds = text_inputs
        else:
            text_embeds = texts

        # condition on characters if needed 
        if exists(character_labels):
            character_embeddings = self.character_embeddings.forward(character_labels)
            neg_character_embeddings = self.character_embeddings.forward(
                torch.where(character_labels==1, 
                torch.tensor(0, dtype=torch.float, device=device), 
                torch.tensor(1,dtype=torch.float, device=device)))

        demask_fn = self.transformer.forward_with_cond_scale

        # condition on story if needed
        if exists(story_text_embedding):
            cond_embed = story_text_embedding
            cond_embed_mask = torch.full(size=cond_embed.shape[:2], fill_value=True, dtype=torch.bool, device=device)

        # autoregressive transformer inference for the specified number of timesteps
        if loop == 'verbose':
            loop_obj = tqdm(zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))), total = timesteps)
        else:
            loop_obj = zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps)))


        # character guidance
        null_text_embeds, is_drop_condition = self.drop_condition(text_embeds, prob=1.)
        null_cond_embed, _ = self.drop_condition(cond_embed, is_drop_condition=is_drop_condition)
        null_cond_embed_mask = torch.where(is_drop_condition[:,None].expand(cond_embed_mask.shape), False, cond_embed_mask)

        for timestep, steps_until_x0 in loop_obj:

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim = -1).indices

            ids = ids.scatter(1, masked_indices, self.mask_id)

            embeds =  demask_fn(
                x = ids,
                text_embeds = text_embeds,
                character_embeds=character_embeddings,
                conditioning_embeddings=cond_embed, # story and embeddings
                conditioning_embeddings_mask=cond_embed_mask, # mask for the embeddings above
            )
            logits = self.to_logits(embeds[:,:seq_len,:])
            
            char_embeds = demask_fn(
                x = ids,
                text_embeds = null_text_embeds,
                character_embeds=character_embeddings,
                conditioning_embeddings=null_cond_embed, # story and embeddings
                conditioning_embeddings_mask=null_cond_embed_mask, # mask for the embeddings above
            )
            char_logits = self.to_logits(char_embeds[:,:seq_len,:])
            
            neg_char_embeds = demask_fn(
                x = ids,
                text_embeds = null_text_embeds,
                character_embeds=neg_character_embeddings,
                conditioning_embeddings=null_cond_embed, # story and embeddings
                conditioning_embeddings_mask=null_cond_embed_mask, # mask for the embeddings above
            )
            neg_char_logits = self.to_logits(neg_char_embeds[:,:seq_len,:])
            
            # logits, char_logits, neg_char_logits = logits.split(5)

            logits = (1-cg_scale) * logits + 2 * cg_scale * char_logits - cg_scale * neg_char_logits
            # ids, _, _ = ids.split(5)

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            is_mask = ids == self.mask_id
            ids = torch.where(
                is_mask,
                pred_ids,
                ids
            )

            probs_without_temperature = logits.softmax(dim = -1)
            scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')

            if not can_remask_prev_masked:
                scores = scores.masked_fill(~is_mask, -1e5)
            else:
                assert self.no_mask_token_prob > 0., 'without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token'

        # get ids
        ids = rearrange(ids, 'b (i j) -> b i j', i = fmap_size, j = fmap_size)
        ids = rearrange(ids, 'b i j -> b (i j)')
        return ids
        

    def training_step(self, batch, batch_idx):
        if self.train_text_embeddings:
            story_images, story_texts, character_labels, _, _ = batch
        else:   
            story_images, story_texts, character_labels, _, _ = batch
            story_texts = story_texts.squeeze(2)

        loss = self(story_text_inputs=story_texts, story_images=story_images, character_labels=character_labels)        

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.train_text_embeddings:
            story_images, story_texts, character_labels,  _, _ = batch
        else:
            story_images, story_texts, character_labels,  _, _ = batch
            story_texts = story_texts.squeeze(2)
            
        loss = self(story_text_inputs=story_texts, story_images=story_images, character_labels=character_labels)  
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, sync_dist=True)
    
    def on_train_epoch_end(self, outputs):
        # manually checkpoint every 10 epochs
        if self.current_epoch != 0 and self.current_epoch % 10 == 0:
            root_dir = self.trainer.default_root_dir
            filename = f"epoch={self.current_epoch}.ckpt"
            self.trainer.save_checkpoint(filepath=os.path.join(root_dir,filename))

    def on_validation_epoch_end(self):
        losses = torch.stack(self.validation_step_outputs)
        print(f"Validation Ended | Average loss: {losses.mean().item()}")
       
        self.validation_step_outputs.clear()  # free memory
    
    def configure_optimizers(self, lr=3e-5):               
        trainable_params = [param for name, param in self.named_parameters() if ("vq_vae" not in name)] # we train all parameters except for the vqvae's

        # optimizer
        optim = torch.optim.Adam(trainable_params, lr=lr)
        return optim
    
    @torch.no_grad()
    def vqvae_encode(self, image, labels):
        out, _, info = self.vq_vae.encode(image)
        indices = info[2]        
        return out, indices
    
    def encode_story(self, text_inputs):
        """
        generate a story embedding from the text inputs 
        """
        if self.train_text_embeddings:
            # text inputs are tokenization ids
            # batch_size = text_inputs.shape[0]
            text_ids_repeated = text_inputs.repeat_interleave(repeats=torch.tensor([self.images_per_story], device=self.device), dim=0)
            text_ids_repeated = rearrange(text_ids_repeated, 'b n1 n2 ... -> b (n1 n2) ...')
            text_embeddings_repeated = self.text_embeddings(text_ids_repeated)
            if text_embeddings_repeated.shape[-2] > self.max_story_length:
                text_embeddings_repeated = text_embeddings_repeated[:,:self.max_story_length,:]
            n_words = text_embeddings_repeated.shape[1]
            text_embeddings_repeated = text_embeddings_repeated + self.pos_text_embeddings(torch.arange(n_words, device=self.device))

            if self.transformer_type=='baseline':
                text_embeddings_repeated = self.transformer.text_embed_proj(text_embeddings_repeated)
            else:
                text_embeddings_repeated = self.transformer.base_transformer.text_embed_proj(text_embeddings_repeated)

            return text_embeddings_repeated
 
        text_embeddings_repeated = text_inputs.repeat_interleave(repeats=torch.tensor([self.images_per_story], device=self.device), dim=0)
        text_embeddings_repeated = rearrange(text_embeddings_repeated, '(b n1) n2 a n -> (b n1) (n2 a) n', n1=self.images_per_story, n2=self.images_per_story)
        text_embeddings_repeated = text_embeddings_repeated + self.pos_text_embeddings(torch.arange(text_embeddings_repeated.shape[1], device=self.device))
        text_embeddings_repeated = self.transformer.text_embed_proj(text_embeddings_repeated)
        return text_embeddings_repeated


    def forward(
        self, 
        story_text_inputs, 
        story_images, 
        character_labels:Optional[torch.Tensor]=None
    ):
        """
        text_inputs : batch x 5 x 1 x text_dim
        story : batch x 5 x 3 x 64 x 64 
        story_character_indices : batch x 5 x num_characters (0/1s)
        conditioning_image_token_ids : batch x 5 x n_ids x 1
        """
        # get the universal story encoding
        story_text_embedding = self.encode_story(text_inputs=story_text_inputs)
        
        # reshape the text inputs : (b x n_images) x 1 x text_dim 
        if self.train_text_embeddings:
            text_inputs = rearrange(story_text_inputs, 'b n_images n_tokens -> (b n_images) n_tokens')
        else:
            text_inputs = rearrange(story_text_inputs, 'b n_images n_tokens dim -> (b n_images) n_tokens dim')

        # reshape the images : (b x n_images) x 3 x 64 x 64
        story_images = rearrange(story_images, 'b n_images ... -> (b n_images) ...')

        if exists(character_labels):
            character_labels = rearrange(character_labels, 'b n_images ...-> (b n_images) ...')

        # pass the inputs through the transformer
        # print(f"reshape images: {story_images.shape}", flush=True)
        return self.forward_image(
            text_inputs=text_inputs,
            image=story_images,
            story_text_embedding=story_text_embedding,
            character_labels=character_labels
        )
 

    def forward_image(
        self, 
        text_inputs, 
        image, 
        story_text_embedding:Optional[torch.Tensor]=None, 
        character_labels:Optional[torch.Tensor]=None,
    ):
        """
        text_inputs: batch x 1 x text_dim OR batch x num_indices
        image: batch x 3 x 64 x 64 
        """
        # pass through the vae
        with torch.no_grad():
            image_tokens, encoding_indices = self.vqvae_encode(image, character_labels)
            if len(encoding_indices.shape)<=1:
                b, _, _, _ = image_tokens.shape
                encoding_indices = encoding_indices.view(b,-1)

        batch, _, h, w = image_tokens.shape
        seq_len = h*w

        # get the indices
        encoding_indices = rearrange(encoding_indices, 'b ... -> b (...)')

        masked_encoding_indices, mask = self.get_masked_indices(indices=encoding_indices, batch=batch, seq_len=seq_len)

        cond_embed, cond_embed_mask = None, None

        # -- MASKGIT TRANSFORMER --

        if self.train_text_embeddings:
            text_ids = text_inputs
            text_inputs = self.text_embeddings(text_inputs)
            n_words = text_inputs.shape[1]
            text_inputs = text_inputs + self.pos_text_embeddings(torch.arange(n_words, device=self.device))

        # condition on story if needed
        if exists(story_text_embedding):
            cond_embed = story_text_embedding
            cond_embed_mask = torch.full(size=cond_embed.shape[:2], fill_value=True, dtype=torch.bool, device=self.device)

        # condition on characters if needed 
        if exists(character_labels):
            character_embeddings = self.character_embeddings.forward(character_labels)

        # drop some text conditions
        text_inputs, is_drop_condition = self.drop_condition(text_inputs)
        cond_embed, _ = self.drop_condition(cond_embed, is_drop_condition=is_drop_condition)
        cond_embed_mask = torch.where(is_drop_condition[:,None].expand(cond_embed_mask.shape), False, cond_embed_mask)
        
        # pass through the transformer
        embeds = self.transformer(
            x=masked_encoding_indices, 
            text_embeds=text_inputs,
            character_embeds=character_embeddings, 
            conditioning_embeddings=cond_embed, 
            conditioning_embeddings_mask=cond_embed_mask,
        )

        embeds = embeds[:,:seq_len,:]
        logits = self.to_logits(embeds)

        out = logits
        out = rearrange(out, 'b n c -> b c n')
        target = torch.where(masked_encoding_indices==self.mask_id, encoding_indices, self.mask_id)
        ce_loss = F.cross_entropy(input=out, target=target, ignore_index=self.mask_id, reduction='mean')
        
        return ce_loss

    def drop_condition(self, tensor:torch.Tensor, prob=0.2, is_drop_condition:Optional[torch.Tensor]=None):
        if not exists(is_drop_condition):
            is_drop_condition = prob_mask_like(shape=tensor.shape[0], prob=prob, device=tensor.device)
        nulls = self.null_condition(torch.zeros(size=tensor.shape[:-1], dtype=torch.long, device=tensor.device))
        tensor = torch.where(
            is_drop_condition[:,None,None].expand(tensor.shape),
            nulls,
            tensor
        )
        return tensor, is_drop_condition

