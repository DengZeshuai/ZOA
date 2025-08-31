import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Mlp
# from timm.models.helpers import checkpoint_seq
import math
from functools import reduce
from operator import mul
import numpy as np

class PromptViT(nn.Module):
    '''
    Vision Transformer with added prompts at the input layer
    '''
    def __init__(self,
                model:VisionTransformer,
                num_prompts = 1):
        super().__init__()
        self.model = model
        self.num_prompts = num_prompts
        self.prompt_dim = model.embed_dim
        self.num_features = model.num_features

        if num_prompts > 0:
            self.prompts = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim))
            # initialization adopted from vpt, https://arxiv.org/abs/2203.12119
            val = math.sqrt(6. / float(3 * reduce(mul, model.patch_embed.patch_size, 1) + self.prompt_dim)) # noqa
            nn.init.uniform_(self.prompts.data, -val, val) # xavier_uniform initialization
    
    def reset(self):
        val = math.sqrt(6. / float(3 * reduce(mul, self.model.patch_embed.patch_size, 1) + self.prompt_dim)) # noqa
        nn.init.uniform_(self.prompts.data, -val, val) # xavier_uniform initialization

    def prompt_injection(self, x):
        if self.num_prompts > 0:
            x = torch.cat((
                x[:,:1,:],
                self.prompts.expand(x.shape[0],-1,-1),
                x[:,1:,:]
            ), dim=1)
        return x
    
    def _collect_layers_features(self, x):
        # collecting features for each layer
        cls_features = []
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)
            if i < len(self.model.blocks) - 1:
                cls_features.append(self.model.blocks[i+1].norm1(x[:, 0]))
            else:
                cls_features.append(self.model.norm(x[:, 0]))
        cls_features = torch.cat(cls_features, dim=1)
        return cls_features

    def forward_features(self, x):
        '''
        Forwarding a batch of samples with prompts' embeddings inserted
        We added only the highlighted line of code based on `timm` library
        '''
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x)
        # !!end
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.model.forward_head(x)
        return x
    
    def layers_cls_features(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)
        return self._collect_layers_features(x)
    
    def layers_cls_features_with_prompts(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x)
        # !!end
        x = self.model.norm_pre(x)
        return self._collect_layers_features(x)