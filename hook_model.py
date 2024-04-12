from collections import OrderedDict
import re
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from fancy_einsum import einsum
from tqdm.notebook import tqdm
import transformer_lens
from transformer_lens.HookedTransformer import HookedTransformer, HookedTransformerConfig
from jaxtyping import Float
import gc
from transformer_lens.hook_points import (
    HookPoint,
) 
import transformer_lens.utils as utils

from eval import get_run_metrics, read_run_dir, get_model_from_run
from munch import Munch
import yaml
from samplers import get_data_sampler
from tasks import get_task_sampler


sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')


class PassThroughEmbed(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        # No parameters needed, but constructor accepts cfg for compatibility

    def forward(self, tokens):
        # Directly return the input without any modifications
        return tokens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, device=DEVICE):

    model = torch.load(path, map_location=DEVICE)
    tl_model = model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

def get_model(path, device=DEVICE):

    model = load_model(path + 'last_final_toy_model.pt', device)
    read_in_weight = torch.load(path + 'last_final_toy_read_in_weight.pt', map_location=device)
    read_in_bias = torch.load(path + 'last_final_toy_read_in_bias.pt', map_location=device) 

    return model, read_in_weight, read_in_bias

def validation_metric(predictions, labels, return_one_element, device):
    predictions = predictions.to(device)

    # loss is mse on last 5 predictions (full context)
    loss = (labels.to(DEVICE) - predictions.to(DEVICE)).square().cpu().numpy().mean(axis=0)[-5:].mean()
    return loss

def get_acts(cache):
    mlp1_out = cache['blocks.0.hook_mlp_out']
    mlp2_out = cache['blocks.1.hook_mlp_out']
    mlp3_out = cache['blocks.2.hook_mlp_out']
    
    # Flatten activations [batch, position, d_model] -> [batch, position * d_model]
    mlp1_flat = mlp1_out.view(mlp1_out.shape[0], -1)
    mlp2_flat = mlp2_out.view(mlp2_out.shape[0], -1)
    mlp3_flat = mlp3_out.view(mlp3_out.shape[0], -1)
    
    # Concatenate flattened activations along the feature dimension
    concatenated_activations = torch.cat([mlp1_flat, mlp2_flat, mlp3_flat], dim=1)
    
    return concatenated_activations
    

def get_data(conf, read_in_weight, read_in_bias):
    # generate random data (20d points on a gaussian)

    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size

    data_sampler = get_data_sampler(conf.training.data, n_dims)
    task_sampler = get_task_sampler(
        conf.training.task,
        n_dims,
        batch_size,
        **conf.training.task_kwargs
    )
    task = task_sampler()
    
    xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end) # should be n_points=conf.training.curriculum.points.end, but has been hacked to work for the max_len of 101 (202)
    ys = task.evaluate(xs)
    weights = task.w_b  # b_size, n_dims, 1

    batch, n_ctx, d_xs = xs.shape

    ys_wide = torch.cat(
        (
            ys.view(batch, n_ctx, 1),
            torch.zeros(batch, n_ctx, d_xs - 1, device=ys.device),
        ),
        axis=2,
    )
    my_zs = torch.stack((xs, ys_wide), dim=2)
    my_zs = my_zs.view(batch, 2 * n_ctx, d_xs).to(DEVICE)

    # apply the read_in transformation
    transformed_zs = einsum("batch n_ctx d_xs, d_model d_xs -> batch n_ctx d_model", my_zs, read_in_weight) + read_in_bias

    # apply padding

    current_len = transformed_zs.shape[1]
    max_len = min(conf.model.n_embd, conf.model.n_positions * 2) # HORRID tech debt I don't get
    pad_len = max(max_len - current_len, 0)

    

    # Apply padding to the right of the second dimension
    # The padding order in F.pad is (left, right, top, bottom) for 4D input, but here it's the equivalent for 3D
    return F.pad(transformed_zs, (0, 0, 0, pad_len), "constant", 0), xs, ys, weights

def project_eigenvector_torch(eig_input, eig_vector):
    eig_vector_norm = eig_vector / eig_vector.norm()
    projections = torch.matmul(eig_input, eig_vector_norm)
    return projections
    
def search_heads_get_loss(model, conf, read_in_weight, read_in_bias, significant_eigenvalue_threshold=0.1):

    model.set_use_split_qkv_input(True)

    # initiate scores, attn position eigenvalues projections
    head_scores = torch.zeros((conf.model.n_layer, conf.model.n_head, 4)).to(DEVICE)
    head_scores.to(DEVICE)

    input, xs, ys, _ = get_data(conf, read_in_weight, read_in_bias)

    with torch.no_grad():
        cache = model.run_with_cache(input)

    corrupt_input, _, _, _ = get_data(conf, read_in_weight, read_in_bias)
    with torch.no_grad():
        corrupt_cache = model.run_with_cache(corrupt_input)

    # compute how much eigenvalues are negative (n_layer n_head n_eigenvalues), multiply by -1 since we want negative values (low eigs = higer score)
    eigenvalue_scores = model.OV.eigenvalues.sum(dim=2) * -1

    for block in range(conf.model.n_layer): # conf.model.n_layer

        head_scores[block, :, 3] = 1 - (block / conf.model.n_layer)
        

        # compute how much this head has ys attend to the preciding xs
        attention_pattern = cache[1][f'blocks.{block}.attn.hook_attn_scores']
        avg_attention_pattern = attention_pattern.mean(dim=0)
        odd_indices = torch.arange(1, avg_attention_pattern.shape[-1], step=2)
        attn_scores = avg_attention_pattern[:, odd_indices, odd_indices - 1].mean(dim=1)

        for head in range(conf.model.n_head): # conf.model.n_head
            # compute how much more xs project on the OV's most important eigenvalues than ys
            AB_matrix = model.OV.AB[block, head]
            eig_results = torch.linalg.eig(AB_matrix)
            eigenvalues, eigenvectors = eig_results.eigenvalues, eig_results.eigenvectors.to(torch.float32)
            significant_idxs = torch.where(torch.abs(eigenvalues) > significant_eigenvalue_threshold)[0]
            significant_eigenvectors = eigenvectors[:, significant_idxs]
            for i, eig_vector in enumerate(significant_eigenvectors.T):
                # Compute projections
                projections = torch.stack([project_eigenvector_torch(input[seq], eig_vector) for seq in range(input.shape[0])])
                even_indices = torch.arange(0, conf.training.curriculum.points.end - 1, step=2) # -1 because the last x does not have a y value
                odd_indices = torch.arange(1, conf.training.curriculum.points.end, step=2)
                even_projections = torch.abs(projections[:, even_indices]).sum()
                odd_projections = torch.abs(projections[:, odd_indices]).sum()
                head_scores[block, head, 2] += (even_projections.to(DEVICE) / odd_projections.to(DEVICE)).to(DEVICE)
            head_scores[block, head, 2] /= i
            # record scores for this head
            head_scores[block, head, 0] = attn_scores[head]
            head_scores[block, head, 1] = eigenvalue_scores[block, head]
            head_scores[block, head, 2] /= max(significant_eigenvectors.shape[1], significant_eigenvalue_threshold) # avoid division by 0

    # normalize and select best head according to these criteria
    for i in range(head_scores.shape[2]):
        criterion_scores = head_scores[:, :, i]
        min_score = criterion_scores.min()
        max_score = criterion_scores.max()
        
        # Avoid division by zero in case all scores for this criterion are the same
        if max_score != min_score:
            head_scores[:, :, i] = (criterion_scores - min_score) / (max_score - min_score)
        else:
            head_scores[:, :, i] = 0
    
    # Select the head with the highest score after normalization
    total_scores = head_scores.mean(dim=2)  # Mean scores across all criteria
    best_head_flat_idx = torch.argmax(total_scores.flatten())  # Flatten and get the index of the best head across all layers & heads
    best_layer_idx = best_head_flat_idx // conf.model.n_head
    best_head_idx = best_head_flat_idx % conf.model.n_head
    selected_head = (best_layer_idx.item(), best_head_idx.item())

    activation = f"blocks.{selected_head[0]}.hook_v_input"

    # by now we have the head we're looking for. compute restricted and excluded loss
    activation = f"blocks.{selected_head[0]}.hook_v_input"

    # by now we have the head we're looking for. compute restricted and excluded loss
    model.reset_hooks()

    

   
    def excluded_patching_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Each HookPoint has a name attribute giving the name of the hook.
        
        corrupt_embedding = corrupt_cache[1]['blocks.0.hook_resid_pre']
        clean_embedding = cache[1]['blocks.0.hook_resid_pre'] # this is actually just embed + pos embed
  
        resid_pre[:, ::2, selected_head[1], :].sub_(clean_embedding[:, ::2, :]).add_(corrupt_embedding[:, ::2, :])
        return resid_pre

    def restricted_patching_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Each HookPoint has a name attribute giving the name of the hook.
        
        corrupt_embedding = corrupt_cache[1]['blocks.0.hook_resid_pre']
        clean_embedding = cache[1]['blocks.0.hook_resid_pre'] # this is actually just embed + pos embed

        resid_pre[:, 1::2, selected_head[1], :].sub_(clean_embedding[:, 1::2, :]).add_(corrupt_embedding[:, 1::2, :])
        return resid_pre


    with torch.no_grad():
        excluded_preds = model.run_with_hooks(input, fwd_hooks=[
            (activation, excluded_patching_hook)
        ])

        restricted_preds = model.run_with_hooks(input, fwd_hooks=[
            (activation, restricted_patching_hook)
        ])
    excluded_loss = (ys - excluded_preds[:, ::2, 0][:, torch.arange(ys.shape[1])]).square().mean()
    restricted_loss = (ys - restricted_preds[:, ::2, 0][:, torch.arange(ys.shape[1])]).square().mean()
    
    #excluded_loss = loss_function(excluded_preds[:, ::2, 0][:, torch.arange(ys.shape[1])], ys)
    #restricted_loss = loss_function(restricted_preds[:, ::2, 0][:, torch.arange(ys.shape[1])], ys)

    model.reset_hooks()
    gc.collect()

    # return losses
    return excluded_loss, restricted_loss, selected_head, total_scores


def convert_gpt2_weights(gpt2, cfg: HookedTransformerConfig, DEVICE='cpu'):
    state_dict = {}

    state_dict["embed.W_E"] = gpt2._backbone.wte.weight
    state_dict["pos_embed.W_pos"] = gpt2._backbone.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = gpt2._backbone.h[l].ln_1.weight.to(torch.device(DEVICE))
        state_dict[f"blocks.{l}.ln1.b"] = gpt2._backbone.h[l].ln_1.bias.to(torch.device(DEVICE))
        
        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = gpt2._backbone.h[l].attn.c_attn.weight
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q.to(torch.device(DEVICE))
        state_dict[f"blocks.{l}.attn.W_K"] = W_K.to(torch.device(DEVICE))
        state_dict[f"blocks.{l}.attn.W_V"] = W_V.to(torch.device(DEVICE))

        qkv_bias = gpt2._backbone.h[l].attn.c_attn.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0].to(torch.device(DEVICE))
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1].to(torch.device(DEVICE))
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2].to(torch.device(DEVICE))

        W_O = gpt2._backbone.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O.to(torch.device(DEVICE))
        state_dict[f"blocks.{l}.attn.b_O"] = gpt2._backbone.h[l].attn.c_proj.bias.to(torch.device(DEVICE))

        
        state_dict[f"blocks.{l}.ln2.w"] = gpt2._backbone.h[l].ln_2.weight.to(torch.device(DEVICE))
        state_dict[f"blocks.{l}.ln2.b"] = gpt2._backbone.h[l].ln_2.bias.to(torch.device(DEVICE))
        
        W_in = gpt2._backbone.h[l].mlp.c_fc.weight
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in.to(torch.device(DEVICE))
        state_dict[f"blocks.{l}.mlp.b_in"] = gpt2._backbone.h[l].mlp.c_fc.bias.to(torch.device(DEVICE))
        
        W_out = gpt2._backbone.h[l].mlp.c_proj.weight
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out.to(torch.device(DEVICE))
        state_dict[f"blocks.{l}.mlp.b_out"] = gpt2._backbone.h[l].mlp.c_proj.bias.to(torch.device(DEVICE))


    state_dict["ln_final.w"] = gpt2._backbone.ln_f.weight.to(torch.device(DEVICE))
    state_dict["ln_final.b"] = gpt2._backbone.ln_f.bias.to(torch.device(DEVICE))

    
    return state_dict


def hook_model(model, conf):
    # model config
    hooked_config = HookedTransformerConfig(
        d_model = conf.model.n_embd,
        d_head = int(conf.model.n_embd / conf.model.n_head),
        d_vocab = 50257,
        n_layers = conf.model.n_layer,
        n_heads = conf.model.n_head,
        n_ctx = 2 * conf.model.n_positions,
        act_fn = 'gelu_new',
        init_weights = False,
        device = DEVICE
    )
    # convert model weights to hooked format
    converted_state_dict = convert_gpt2_weights(model, hooked_config)
    

    # load current parameters into a hooked model
    hooked_model = HookedTransformer(cfg=hooked_config, move_to_device=True)
    hooked_model.load_and_process_state_dict(converted_state_dict)

    # insert custom unembedder
    new_unembed = nn.Linear(hooked_config.d_model, 1)
    new_unembed.weight = torch.nn.Parameter(model._read_out.weight)
    new_unembed.bias = torch.nn.Parameter(model._read_out.bias)

    hooked_model.unembed = new_unembed

    # remove the original model embedder (we are not using tokens) and replace it with a pass trough embedder
    
    pass_through_embed = PassThroughEmbed(cfg=hooked_config)
    hooked_model.embed = pass_through_embed

    # load the parameters of the model's real embedder
    read_in_weight = model._read_in.weight
    read_in_bias = model._read_in.bias

    return hooked_model, read_in_weight, read_in_bias