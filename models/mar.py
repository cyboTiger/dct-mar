from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss
import torch_dct as dct

from models.dct_layer import DCT2DLayer, LinearDCT
from models.lin_emb import BottleneckPatchEmbed, FinalLayer

import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import seaborn as sns

def safe_log_transform(x):
    # use log1p to ensure numerical stability
    return torch.sign(x) * torch.log1p(torch.abs(x))

def safe_exp_transform(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, in_channels=3, bottleneck_dim=768, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 hybrid_prob=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 recon_lambda=0.2
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # patchify specifics
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        # 
        self.seq_h = self.seq_w = self.grid_size = img_size // patch_size
        self.seq_len = self.seq_h * self.seq_w
        # a patch is transformed into a token
        self.token_embed_dim = encoder_embed_dim
        self.grad_checkpointing = grad_checkpointing
        self.linear_embed = nn.Sequential(nn.Linear(patch_size**2*in_channels, bottleneck_dim), 
                                          nn.GELU(),
                                          nn.Linear(bottleneck_dim, encoder_embed_dim))
        # self.linear_embed = BottleneckPatchEmbed(img_size=img_size, 
        #                                          patch_size=patch_size, 
        #                                          in_chans=in_channels,
        #                                          pca_dim=bottleneck_dim,
        #                                          embed_dim=encoder_embed_dim,
        #                                          bias=True)

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # DCT/IDCT layer
        self.dct_layer = DCT2DLayer(size_h=img_size, size_w=img_size, direction='dct', norm='ortho')
        self.idct_layer = DCT2DLayer(size_h=img_size, size_w=img_size, direction='idct', norm='ortho')
        # self.ln = nn.LayerNorm(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        self.hybrid_prob = hybrid_prob
        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        self.recon_lambda = recon_lambda

        # Final Layer from latent token to 1-D image patches
        self.final_layer = FinalLayer(hidden_size=self.token_embed_dim,
                                      patch_size=patch_size,
                                      out_channels=in_channels)

        # decoding order related
        # Zig-zag order starting from top-left
        self.zigzag_order = self._generate_frequency_order()

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        for m in self.linear_embed.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, LinearDCT):
            return
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, c*p*p]

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        bsz, t, _= x.shape
        p = self.patch_size
        c = self.in_channels
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c*p**2)
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def _generate_frequency_order(self):
        """sorted based on euclidean distance"""
        order = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt(i**2 + j**2) 
                order.append((i * self.grid_size + j, dist))
        
        # ascending order
        order.sort(key=lambda x: x[1])
        return np.array([x[0] for x in order])
    
    def sample_orders_zigzag(self, bsz, hybrid_prob=None):
        if hybrid_prob is not None:
            if random.random() > hybrid_prob:
                return self.sample_orders(bsz)
        order = torch.from_numpy(self.zigzag_order).long().cuda()
        return order.unsqueeze(0).repeat(bsz, 1)

    def random_masking_zigzag(self, x, orders, is_zigzag=True):
        if not is_zigzag:
            return self.random_masking(x, orders)
        bsz, seq_len, _ = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0] 
        num_keep = int(np.ceil(seq_len * (1 - mask_rate)))
        
        mask = torch.ones(bsz, seq_len, device=x.device)
        #  num_keep 个是最重要的低频 Token，不遮掩 (mask=0)
        # 后面的高频 Token 被遮掩 (mask=1)
        for i in range(bsz):
            mask[i, orders[i, :num_keep]] = 0
            
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss, pred_xstart = self.diffloss(z=z, target=target, mask=mask)
        return loss, pred_xstart

    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels)

        # DCT transform into hi/lo frequency token space
        x = self.dct_layer(imgs) # [B, c, h, w]

        x = safe_log_transform(x) # approxiamted range: [-12, 12]
        # x = self.ln(x) # not sure whether to add or not

        # convert to patch token dim
        x = self.patchify(x) # [B, l, c*p*p]
        gt_dct = x.clone().detach() # [B, l, c*p*p]
        gt_dct = gt_dct.reshape(-1, gt_dct.size(-1)) # [B*l, c*p*p]
        # patchify & linear embedding
        x = self.linear_embed(x) # [B*l, embed_dim]
        
        gt_latents = x.clone().detach()
        orders = self.sample_orders_zigzag(bsz=x.size(0), hybrid_prob=self.hybrid_prob)
        mask = self.random_masking_zigzag(x, orders)

        # mae encoder
        x = self.forward_mae_encoder(x, mask, class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        # diffloss & reconstructed latent
        loss, pred_xstart = self.forward_loss(z=z, target=gt_latents, mask=mask)

        # pred_xstart shape: [B*l*diffusion_batch_mul, embed_dim]
        recon_dct = self.final_layer(pred_xstart) # [B*l*diffbm, c*p*p]
        gt_dct = gt_dct.repeat(self.diffusion_batch_mul, 1)
        mask_flat = mask.reshape(-1).repeat(self.diffusion_batch_mul)
        
        # only compute masked positions
        recon_loss = F.mse_loss(
            recon_dct[mask_flat==1],
            gt_dct[mask_flat==1]
        )

        bsz = imgs.shape[0]
        recon_dct_slice = recon_dct[:bsz*self.seq_len].reshape(bsz, self.seq_len, -1) # [B*l, c*p*p]
        recon_dct_slice = self.unpatchify(recon_dct_slice)
        recon_dct_slice = safe_exp_transform(recon_dct_slice)
        recon_pixel = self.idct_layer(recon_dct_slice)
        recon_pixel = recon_pixel.clamp(0, 1)

        recon_pixel_loss = F.mse_loss(
            recon_pixel,
            imgs
        )

        loss += self.recon_lambda*recon_loss
        loss += 0.5*recon_pixel_loss

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders_zigzag(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_mae_decoder(x, mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        # tokens = self.unpatchify(tokens)

        # tokens = dct.idct_2d(tokens)

        # linear embed & unpatchify
        tokens = self.final_layer(tokens)
        freqs = self.unpatchify(tokens)

        freqs = safe_exp_transform(freqs)
        # Inverse dct
        imgs = self.idct_layer(freqs)

        return imgs


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
