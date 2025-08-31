"""
Copyright to FOA Authors ICML 2024
"""

import torch
import torch.nn as nn

import cma
import numpy as np
import os

from quant_library.quant_layers.matmul import *

RUNNING_IMAGNET_R = False

class FOA(nn.Module):
    """test-time Forward Optimization Adaptation
    FOA devises both input level and output level adaptation.
    It avoids modification to model weights and adapts in a backpropogation-free manner.
    """
    def __init__(self, args, model, fitness_lambda=0.4):
        super().__init__()
        self.fitness_lambda = fitness_lambda
        self.args = args
        self.model = model
        self.num_features = model.num_features
        self.popsize = args.popsize
        self.es = self._init_cma() # initialization for CMA-ES

        self.best_prompts = model.prompts
        self.best_loss = np.inf
        self.hist_stat = None # which is used for calculating the shift direction in Eqn. (8)

    def _init_cma(self):
        """CMA-ES initialization"""
        dim = self.model.prompts.numel()
        popsize = self.popsize if self.popsize > 1 else self.popsize + 1
        cma_opts = {
            'seed': 2020,
            'popsize': popsize,
            'maxiter': -1,
            'verbose': -1,
        }
        es = cma.CMAEvolutionStrategy(dim * [0], 1, inopts=cma_opts)
        return es

    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean
            
    def _get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-self.num_features:] - self.hist_stat

    def forward(self, x):
        """calculating shift direction, Eqn. (8)"""
        shift_vector = self._get_shift_vector()

        self.best_loss, self.best_outputs, batch_means = np.inf, None, []

        """Sampling from CMA-ES and evaluate the new solutions.
        Note that we also compare the current solutions with the previous best one"""
        if self.popsize > 1:
            prompts, losses = self.es.ask() + [self.best_prompts.flatten().cpu()], []
        else:
            prompts, losses = self.es.ask(), []
        
        for j, prompt in enumerate(prompts):
            self.model.prompts = torch.nn.Parameter(torch.tensor(prompt, dtype=torch.float).
                                                        reshape_as(self.model.prompts).cuda())
            self.model.prompts.requires_grad_(False)

            outputs, loss, batch_mean = forward_and_get_loss(x, self.model, self.fitness_lambda, self.train_info, shift_vector, self.imagenet_mask)
            batch_means.append(batch_mean[-self.num_features:].unsqueeze(0))
            del batch_mean

            if self.best_loss > loss.item():
                self.best_prompts = self.model.prompts
                self.best_loss = loss.item()
                self.best_outputs = outputs
                outputs = None
            losses.append(loss.item())
            del outputs

            print(f'Solution:[{j+1}/{len(prompts)}], Loss: {loss.item()}')

        """CMA-ES updates, Eqn. (6)"""
        self.es.tell(prompts, losses)
        
        """Update overall test statistics, Eqn. (9)"""
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_means)
        return self.best_outputs
    
    def obtain_origin_stat(self, train_loader):
        print('===> begin calculating mean and variance')
        
        self.model.eval()
        # std_path = f'train_{self.args.arch}_info_std.pt'
        # mean_path = f'train_{self.args.arch}_info_mean.pt'
        std_path = f'train_info_std.pt'
        mean_path = f'train_info_mean.pt'

        if not self.args.compute_train_info and os.path.exists(std_path) and os.path.exists(mean_path):
            std = None
            mean = None    

            if os.path.exists(std_path):
                std = torch.load(std_path).cuda()
            if os.path.exists(mean_path):
                mean = torch.load(mean_path).cuda()
            
            if std is not None and mean is not None:
                print('===> loaded mean and variance end')
                self.train_info = (std, mean)
            else:
                raise ValueError('std and mean files can not be found')
        else:
            features = []
            with torch.no_grad():
                for _, dl in enumerate(train_loader):
                    images = dl[0].cuda()
                    feature = self.model.layers_cls_features(images)
                    features.append(feature)
                    # break
                features = torch.cat(features, dim=0)
                self.train_info = torch.std_mean(features, dim=0)
            del features
        
        if self.args.quant:
            # preparing quantized model for prompt adaptation
            for _, m in self.model.model.named_modules():
                if type(m) == PTQSLBatchingQuantMatMul:
                    m._get_padding_parameters(torch.zeros((1,self.num_features//64,197+self.model.num_prompts,64)).cuda(), torch.zeros((1,self.num_features//64,64,197+self.model.num_prompts)).cuda())
                elif type(m) == SoSPTQSLBatchingQuantMatMul:
                    m._get_padding_parameters(torch.zeros((1,self.num_features//64,197+self.model.num_prompts,197+self.model.num_prompts)).cuda(), torch.zeros((1,self.num_features//64,197+self.model.num_prompts,64)).cuda())
        print('===> calculating mean and variance end')

        ### save train_info
        if not os.path.exists(std_path):
            torch.save(self.train_info[0], std_path)
        if not os.path.exists(mean_path):
            torch.save(self.train_info[1], mean_path)

    def reset(self):
        self.es = self._init_cma()
        self.hist_stat = None

        self.model.reset()
        self.best_prompts = self.model.prompts
        

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='none').cuda()

def forward_and_get_loss(images, model, fitness_lambda, train_info, shift_vector, imagenet_mask):
    features = model.layers_cls_features_with_prompts(images)

    """discrepancy loss for Eqn. (5)"""
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])
    # NOTE: $lambda$ should be 0.2 for ImageNet-R!!
    discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) * images.shape[0] / 64
    
    cls_features = features[:, -model.num_features:] # the feature of classification token
    output = model.model.head(cls_features)

    """entropy loss for Eqn. (5)"""
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]
    entropy_loss = softmax_entropy(output).sum()
    loss = discrepancy_loss + entropy_loss
    
    """activation shifting, Eqn. (7)"""
    if shift_vector is not None:
        output = model.model.head(cls_features + 1. * shift_vector)
        if imagenet_mask is not None:
            output = output[:, imagenet_mask]

    return output, loss, batch_mean