"""
Copyright to FOA Authors ICML 2024
"""

import torch
import torch.nn as nn
import torch.jit
import cma
import os
import numpy as np
from models.resnet import Bottleneck


class ZOA_ResNet(nn.Module):
    """test-time Forward Only Adaptation
    FOA devises both input level and output level adaptation.
    It avoids modification to model weights and adapts in a backpropogation-free manner.
    """
    def __init__(self, args, model, fitness_lambda=40):
        super().__init__()
        self.args = args
        self.fitness_lambda = fitness_lambda

        self.model = model
        self.steps = args.steps

        self.hist_stat = None # which is used for calculating the shift direction in Eqn. (8)

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
            return self.train_info[1][-1] - self.hist_stat

    def forward(self, x):
        shift_vector = self._get_shift_vector()

        outputs, batch_mean, loss, domain_change = forward_and_get_loss(x, self.model, self.fitness_lambda, 
                                                                        self.train_info, shift_vector)
        
        if domain_change:
            self.model.logger.info(f'Reset shift vector')
            self.hist_stat = None
            shift_vector = None

        # obtain the params that requires computing gradient
        alphas, params, embed_dims = self.model.parameters_to_vector()
       
        self.params_ori = params
        
        if self.steps > 0:
            for step in range(self.steps):

                # obtain zo grad
                ghat_alpha, ghat, loss = spsa_grad_estimate_bi(x, self.model, alphas, params, 
                                                self.fitness_lambda, self.train_info, 
                                                ck_alpha=self.args.spsa_c_alpha, 
                                                ck=self.args.spsa_c, 
                                                sp_avg=self.args.sp_avg, 
                                                loss=loss if step==0 else None)
                
                ##### update model
                self.model.vector_to_parameters(alphas, self.params_ori)
                self.model.record_gradient(ghat_alpha, ghat)
                self.model.optimizes()

                # update params for next step
                alphas, params, embed_dims = self.model.parameters_to_vector()
                self.params_ori = params

                print(f'Solution:[{step+1}/{self.steps}], Loss: {loss.item()}')

        self._update_hist(batch_mean)
        return outputs
    
    def obtain_origin_stat(self, train_loader):
        print('===> begin calculating mean and variance')
        self.model.eval()


        features, layer_stds, layer_means = [], [], []
        with torch.no_grad():
            for i, dl in enumerate(train_loader):
                images = dl[0].cuda()
                feature, _ = self.model.forward_features(images)
                features.append(feature)
                if i == 24: break

        for i in range(len(features[0])):
            layer_features = [feature[i] for feature in features]
            layer_features = torch.cat(layer_features, dim=0).cuda()

            assert len(layer_features.shape) == 2
            if len(layer_features.shape) == 4: dim = (0,2,3)
            else: dim = (0)

            layer_stds.append(layer_features.std(dim=dim))
            layer_means.append(layer_features.mean(dim=dim))

        layer_stds = torch.cat(layer_stds, dim=0)
        layer_means = torch.cat(layer_means, dim=0)
        self.train_info = (layer_stds, layer_means)
        print('===> calculating mean and variance end')

    def reset(self):
        """Reset model and optimizer states to initial values."""
        self.hist_stat = None
        if hasattr(self.model, 'reset'):
            self.model.reset()

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='mean').cuda()

def forward_and_get_loss(images, model, fitness_lambda, train_info, shift_vector):
    features, domain_change = model.forward_features(images)
    
    discrepancy_loss = 0
    loss_std, loss_mean = 0, 0
    si, ei = 0, 0
    for i in range(len(features)):
        layer_features = features[i]
        si = ei
        ei = si + features[i].shape[1]
        if len(layer_features.shape) == 4:
            dim = (0,2,3)
        else:
            dim = (0)

        batch_std, batch_mean = layer_features.std(dim=dim), layer_features.mean(dim=dim)
        loss_std += criterion_mse(batch_std, train_info[0][si:ei])
        loss_mean += criterion_mse(batch_mean, train_info[1][si:ei])

    loss_std = loss_std / len(features)
    loss_mean = loss_mean / len(features)
    discrepancy_loss += loss_std + loss_mean

    cls_features = features[-1] # the output feature of average pooling layer
    output = model.model.fc(cls_features)

    entropy_loss = softmax_entropy(output).mean()

    loss = fitness_lambda * discrepancy_loss + entropy_loss

    """activation shifting, Eqn. (7) of FOA"""
    if shift_vector is not None:
        output = model.model.fc(cls_features + 1. * shift_vector)

    return output, batch_mean, loss, domain_change

def spsa_grad_estimate_bi(inputs, model, alphas, params, fitness_lambda, train_info, shift_vector=None, ck_alpha=0.01, ck=0.01, sp_avg=5, loss=None):
    #* repeat k times and average them for stabilizing
    ghats_alpha, ghats = [], []
    N_alphas = len(alphas)
    N_params = len(params)

    for _ in range(sp_avg):
        #! Segmented Uniform [-1, 0.5] U [0.5, 1]
        p_side = (torch.rand(N_alphas).reshape(-1,1) + 1)/2
        samples = torch.cat([p_side,-p_side], dim=1)
        perturb_a = torch.gather(samples, 1, 
            torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)
            ).reshape(-1).cuda()
        
        p_side = (torch.rand(N_params).reshape(-1,1) + 1)/2
        samples = torch.cat([p_side,-p_side], dim=1)
        perturb = torch.gather(samples, 1, 
            torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)
            ).reshape(-1).cuda()
        del samples; del p_side

        #* one-side Approximated Numerical Gradient by default
        alphas_r = alphas + ck_alpha * perturb_a
        param_r = params + ck * perturb
        model.vector_to_parameters(alphas_r, param_r)
        *_, loss1, _ = forward_and_get_loss(inputs, model, fitness_lambda, train_info, shift_vector)
        del param_r, alphas_r

        #* one-side Approximated Numerical Gradient
        if loss is not None:
            loss2 = loss
        else:
            model.vector_to_parameters(alphas, params)
            *_, loss2, _ = forward_and_get_loss(inputs, model, fitness_lambda, train_info, shift_vector)
            loss = loss2

        #* parameter update via estimated gradient
        ghat_alpha = (loss1 - loss2)/(ck_alpha*perturb_a)
        ghat = (loss1 - loss2)/(ck*perturb)
        
        # record gradients at each step
        ghats_alpha.append(ghat_alpha.reshape(1, -1))
        ghats.append(ghat.reshape(1, -1))
    
    ghats_alpha = torch.cat(ghats_alpha, dim=0).mean(dim=0)
    ghat = torch.cat(ghats, dim=0).mean(dim=0) 
    loss = ((loss1+loss2)/2)

    return ghats_alpha, ghat, loss

def load_model_and_optimizer(model, alpha_optimizer, eps_optimizer, 
                            model_state, alpha_optimizer_state, eps_optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    alpha_optimizer.load_state_dict(alpha_optimizer_state)
    eps_optimizer.load_state_dict(eps_optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    
    fix_layers = ['model.bn1', 'model.layer1.0.']
    fix_layers += ['model.layer4']
    for name, param in model.named_parameters():
        for fix_name in fix_layers:
            # if fix_name in name:
            if fix_name in name or name.find('bn3') >= 0 or name.find('downsample.1') >= 0:
                param.requires_grad_(False)

    return model