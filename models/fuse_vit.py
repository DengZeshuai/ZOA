from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FuseViT(nn.Module):
    def __init__(self,
                args,
                model,
                logger):
        super().__init__()
        self.args = args
        self.model = model
        self.embed_dim = model.embed_dim
        self.num_features = model.num_features # dim of the last norm layer
        
        self.weight_margin = 0.01
        self.logger = logger

        self.domain_var = None
        self.domain_mean = None

        self.similarities = torch.Tensor([[0]]).cuda()

        self.corruption = 'original'
        self.max_weight_nums = args.max_weight_nums


    def estimate_scale_factor(self):
        max_weight = max([_.abs().mean().item() for _ in self.params['epsilon_weight']])
        max_bias = max([_.abs().mean().item() for _ in self.params['epsilon_bias']])
        max_value = max(max_weight, max_bias)

        scale_factor = max_value / self.weight_margin
        return scale_factor
    
    @torch.no_grad()
    def save_current_weight(self):
        # the first param is the zero matrix, should be ignored
        if (self.weight_nums -1)>= self.args.max_weight_nums:
            # discard one of the most similar params
            _, k, p = self.select_similar_pair()
            idx = torch.min(k, p)
            # remove the redundant one
            self._remove(idx+1)

        self.weight_nums += 1

        scale_factor = self.estimate_scale_factor()
        assert torch.isnan(torch.tensor([scale_factor])) == False, 'wrong scale factor'
        self.logger.info(f'===> saving weight !!!, scale factor: {scale_factor}')
        new_alpha_value = -np.inf
        if scale_factor > 1: # 大于1的话进行缩放
            new_alpha_value = torch.log(torch.exp(self.alpha * self.alpha_scale).sum() * (scale_factor-1))
        new_alpha_value = torch.max(torch.max(self.alpha * self.alpha_scale), torch.Tensor([new_alpha_value]).cuda()) / self.alpha_scale
        new_alpha_value = torch.clip(new_alpha_value, -10, 10)

        self.alpha = nn.Parameter(torch.cat([self.alpha, new_alpha_value]))

        for m in self.model.modules():
            if isinstance(m, FuseLayerNorm):
                m.save_current_weight(self.alpha)
        
        self.update_similar_matrix(add_idx=-1) # -1 for the new params
        self.logger.info(f'====> remaining: {self.weight_nums}')
        self.collect_params()

        # reconfigure the optimizer for new alpha
        self.alpha_optimizer.param_groups = []
        self.alpha_optimizer.add_param_group(
            {'params': self.params['alpha'], 'lr': self.args.lr_alpha, 'weight_decay': self.args.weight_decay_alpha}
        )

    def select_similar_pair(self):
        if not hasattr(self, 'similar_matrix'):
            self.similar_matrix = self.update_similar_matrix()

        max_val, max_idx = torch.max(self.similar_matrix.view(-1), dim=0)
        k = max_idx // self.similar_matrix.size(1)
        p = max_idx % self.similar_matrix.size(1)

        return max_val, k, p
    
    def update_similar_matrix(self, add_idx=None):
        if not hasattr(self, "similar_matrix"):
            # weight_nums is N + 1, the first zero matrix do not require to calculate the similarity
            # self.similar_matrix = torch.zeros((self.weight_nums - 1, self.weight_nums - 1)).cuda()
            # max_weight_nums is fixed to N
            self.similar_matrix = torch.zeros((self.max_weight_nums, self.max_weight_nums)).cuda()
            
        delta_params_weight = [[] for _ in range(self.weight_nums - 1)]
        delta_params_bias = [[] for _ in range(self.weight_nums - 1)]
        for _, m in self.model.named_modules():
            if isinstance(m, FuseLayerNorm):
                # obtain the previous N weights and biases, 
                # idx 0 is zeros matrix that does not have any meaning
                for i in range(self.weight_nums - 1):
                    delta_params_weight[i].append(m.weights[i+1].flatten())
                    delta_params_bias[i].append(m.biases[i+1].flatten())
        
        if add_idx is None: # self.similar_matrix.sum() == 0:
            # calculate the cosine similarity between the N+1 parameters
            # only record the upper traingle of the matrix
            for i in range(self.weight_nums - 1):
                for j in range(i+1, self.weight_nums - 1):
                    if i == j:
                        # avoid compute the self-similarity
                        self.similar_matrix[i, j] = 0
                    else:
                        cos_sim = self.calculate_cosine_similarity(
                            delta_params_weight[i] + delta_params_bias[i], 
                            delta_params_weight[j] + delta_params_bias[j])
                        self.similar_matrix[i, j] = cos_sim

        else: 
            # only calculate the cosine similarity between 
            # the new parameters and the previous parameters
            if add_idx < 0: 
                add_idx += self.weight_nums - 1 # add_idx is negative
                assert 0 <= add_idx <= self.args.max_weight_nums, 'wrong add_idx'

            # update the add_idx column
            for j in range(add_idx):
                if j == add_idx:
                    # avoid compute the self-similarity
                    self.similar_matrix[j, add_idx] = 0
                else:
                    cos_sim = self.calculate_cosine_similarity(
                        delta_params_weight[j] + delta_params_bias[j], 
                        delta_params_weight[add_idx] + delta_params_bias[add_idx])
                self.similar_matrix[j, add_idx] = cos_sim
            
            # update the add_idx row
            for i in range(add_idx, self.weight_nums -1):
                if add_idx == i:
                    # avoid compute the self-similarity
                    self.similar_matrix[i, add_idx] = 0
                else:
                    assert i < len(delta_params_weight) and i < len(delta_params_bias), 'wrong index'
                    assert add_idx < len(delta_params_weight) and add_idx < len(delta_params_bias), 'wrong index'
                    cos_sim = self.calculate_cosine_similarity(
                        delta_params_weight[i] + delta_params_bias[i], 
                        delta_params_weight[add_idx] + delta_params_bias[add_idx])
                    self.similar_matrix[add_idx, i] = cos_sim
            
        return self.similar_matrix
        
    def calculate_cosine_similarity(self, params_x, parmas_y):  
        cos_sims = []
        for x, y in zip(params_x, parmas_y):
            sim = F.cosine_similarity(x, y, dim=0)
            cos_sims.append(sim)
        return torch.stack(cos_sims, dim=0).mean()

    def replace_fuse_ln(self):
        self.weight_nums = 1
        self.state_dicts = [self.model.state_dict()]
        self._replace_fuse_layer()
        del self.state_dicts
        self.collect_params()
    
    def _replace_fuse_layer(self):
        self.alpha = nn.Parameter(torch.randn(self.weight_nums) / self.weight_nums)
        self.alpha_scale = torch.ones(1).cuda() * self.args.alpha_scale
        ln_modules = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.LayerNorm) and module.weight.requires_grad == True]
        self.fuse_modules = []
        for index, (name, module) in enumerate(ln_modules):
            ln_weight_list = [sd[f'{name}.weight'].cuda() for sd in self.state_dicts]
            ln_bias_list = [sd[f'{name}.bias'].cuda() for sd in self.state_dicts]
            
            fuse_ln = FuseLayerNorm(
                    name, module.normalized_shape, module.eps,
                    ln_weight_list, ln_bias_list, self.alpha, self.alpha_scale
                ).cuda()
            self.fuse_modules.append(fuse_ln)
            
            modules_dict = dict(self.model.named_modules())
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = modules_dict[parent_name]
                setattr(parent_module, child_name, fuse_ln)
            else:
                setattr(self.model, name, fuse_ln)

    def configure_model(self):
        self.model.train()
        self.model.requires_grad_(False)
        self.alpha.requires_grad_(True)
        # self.alpha_scale.requires_grad_(True)
        for m in self.model.modules():
            if isinstance(m, FuseLayerNorm):
                m.epsilon_weight.requires_grad_(True)
                m.epsilon_bias.requires_grad_(True)
        
    def collect_params(self):
        update_names = ['epsilon_weight', 'epsilon_bias']
        params = {update_name:[] for update_name in update_names}
        params['alpha'] = [self.alpha]
        # params['alpha_scale'] = [self.alpha_scale]
        params['alpha_scale'] = [] # no need to update alpha_scale
        for nm, m in self.model.named_modules():
            if isinstance(m, FuseLayerNorm):
                for np, p in m.named_parameters():
                    if np in update_names:  # weight is scale, bias is shift
                        params[np].append(p)
        self.params = params
        return params

    def parameters_to_vector(self):
        params = self.collect_params()
        alphas = nn.utils.parameters_to_vector(params['alpha'])
        delta_param = [param for layer_params in zip(params['epsilon_weight'], params['epsilon_bias']) for param in layer_params]
        delta_param = nn.utils.parameters_to_vector(delta_param)
        embed_dims = self.embed_dim
        return alphas, delta_param, embed_dims

    def vector_to_parameters(self, alphas, delta_param):
        params = self.collect_params()
        alpha_ori = params['alpha']
        delta_param_ori = [param for layer_params in zip(params['epsilon_weight'], params['epsilon_bias']) for param in layer_params]
        params = alpha_ori + delta_param_ori
        vector = torch.cat([alphas, delta_param], dim=0)
        
        # Pointer for slicing the vector for each parameter
        pointer = 0
        for param in params:
            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param.data = vector[pointer:pointer + num_param].view_as(param).data
            # Increment the pointer
            pointer += num_param

    def record_gradient(self, alphas_grad, delta_param_grad):
        params = self.collect_params()
        alpha_ori = params['alpha']
        delta_param_ori = [param for layer_params in zip(params['epsilon_weight'], params['epsilon_bias']) for param in layer_params]
        params = alpha_ori + delta_param_ori
        
        grad_vector = torch.cat([alphas_grad, delta_param_grad], dim=0)

        # Pointer for slicing the vector for each parameter
        pointer = 0
        for param in params:
            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param.grad = grad_vector[pointer:pointer + num_param].view_as(param).data
            # Increment the pointer
            pointer += num_param
    
    @torch.no_grad()
    def forward(self, x):
        x = self.forward_features(x)
        x = self.model.forward_head(x)
        return x
    
    def forward(self, x):
        x = self.embed(x)
        if self.args.quant:
            features = self.get_stages_last_quant_features(x)
        else:
            features = self.get_stages_last_features(x)
        outputs = self.model.fc(features[-1])
        return outputs

    def forward_features(self, x):
        '''
        Forwarding a batch of samples with prompts' embeddings inserted
        We added only the highlighted line of code based on `timm` library
        '''
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x
    
    @torch.no_grad()
    def forward_embedding(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        return self.model.norm_pre(x)[:,1:]
    
    def forward_last_features(self, x):
        x = self.model.forward_features(x)
        return x[:,1:], self.model.forward_head(x)
    
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
    
    def layers_cls_features(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)
        return self._collect_layers_features(x)
    
    def layers_cls_features_with_prompts(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)

        x = self.model.norm_pre(x)

        # detect domain shift
        with torch.no_grad():
            embed_features = x[:,1:].clone().detach()
            domain_change = self._check_should_save(embed_features)
            if domain_change:
                self.logger.info('===> domain changes, saving models')
                self.save_current_weight()
                self.reset_domain_info()
            self._update_domain_info(embed_features)
        
        return self._collect_layers_features(x), domain_change


    def plot_alpha(self, num=100):
        with torch.no_grad():
            if self.weight_nums < num:
                info = f"=====> All alpha: {self.alpha.data}, scale: {self.alpha_scale.item():.4f}, dw: {self.params['epsilon_weight'][0].abs().mean(0).item():.4f}, db: {self.params['epsilon_bias'][0].abs().mean(0).item():.4f}"
            else:
                info = f"=====> All alpha: {self.alpha.data[-num:]}, scale: {self.alpha_scale.item():.4f}, dw: {self.params['epsilon_weight'][0].abs().mean(0).item():.4f}, db: {self.params['epsilon_bias'][0].abs().mean(0).item():.4f}"
        return info

    def set_optimizers(self, alpha_optimizer, eps_optimizer):
        self.alpha_optimizer = alpha_optimizer
        self.eps_optimizer = eps_optimizer

    def reset_domain_info(self):
        self.domain_var = None
        self.domain_mean = None
    
    @torch.no_grad()
    def _check_should_save(self, embed_features):
        if embed_features.shape[0] < 64: # small batch size leads to disturbulance
            return False
        
        if self.domain_var is None or self.domain_mean is None:
            return False
        
        emb_var, emb_mean = embed_features.var(dim=(0,1)), embed_features.mean(dim=(0,1))

        return self._calculate_domain_shift(
            self.domain_var,
            self.domain_mean,
            emb_var,
            emb_mean
        ) > self.args.domain_t # there is a shift, default 0.1 or 0.05
    
    @torch.no_grad()
    def _calculate_domain_shift(self, domain_var, domain_mean, cur_var, cur_mean, eps=1e-8):
        d1 = (domain_var + (domain_mean - cur_mean) ** 2) / 2. / (cur_var + eps) - 0.5
        d2 = (cur_var + (domain_mean - cur_mean) ** 2) / 2. / (domain_var + eps) - 0.5
        return torch.mean((d1+d2))

    @torch.no_grad()
    def _update_domain_info(self, embed_features):
        if embed_features.shape[0] < 64: # small batch size leads to disturbulance
            return False
        
        emb_var, emb_mean = embed_features.var(dim=(0,1)), embed_features.mean(dim=(0,1))
        if self.domain_var is None:
            self.domain_var, self.domain_mean = emb_var, emb_mean
        else:
            self.domain_var = 0.8 * self.domain_var + 0.2 * emb_var
            self.domain_mean = 0.8 * self.domain_mean + 0.2 * emb_mean


    @torch.no_grad()
    def reset(self):
        self.weight_nums = 1
        device = self.alpha.device
        self.alpha = nn.Parameter(torch.randn(self.weight_nums).to(device) / self.weight_nums)
        self.alpha_scale = torch.ones(1).to(device) * self.args.alpha_scale
        
        for m in self.fuse_modules:
            m.reset(self.weight_nums, self.alpha, self.alpha_scale)

        params = self.collect_params()
        # reset optimizes
        self.alpha_optimizer = torch.optim.AdamW([
            {'params': params['alpha'], 'lr': self.args.lr_alpha, 
                    'weight_decay': self.args.weight_decay_alpha}
        ])
        self.eps_optimizer = torch.optim.SGD(
            params['epsilon_weight'] + params['epsilon_bias'], 
            self.args.lr, momentum=self.args.spsa_momentum, 
            weight_decay=self.args.weight_decay)

    @torch.no_grad()
    def _remove(self, idx): # TODO
        assert 0 <= idx < self.alpha.size(0)
        assert self.weight_nums == self.alpha.size(0)
        self.weight_nums -= 1
        self.alpha = nn.Parameter(torch.cat([self.alpha.data[:idx], self.alpha.data[idx+1:]]))
        self.logger.info(f'removing idx: {idx-1}, remaining num: {self.weight_nums-1}' )
        for i, m in enumerate(self.fuse_modules):
            m.remove_idx(self.alpha, idx)
        assert self.alpha.size(0) == self.weight_nums

    def optimizes(self):
        if self.weight_nums > 1:
            self.alpha_optimizer.step()

        self.eps_optimizer.step()

        self.alpha_optimizer.zero_grad()
        self.eps_optimizer.zero_grad()
        self.alpha_optimizer.state = defaultdict(dict)


class FuseLayerNorm(nn.Module):
    def __init__(self, name, normalized_shape, eps, weights, biases, alpha, alpha_scale):
        super(FuseLayerNorm, self).__init__()
        # default LayerNorm parameter
        self.name = name
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # fuse parameter
        self.weight_nums = len(weights)
        self.origin_weight = weights[0]
        self.origin_bias = biases[0]

        self.weights = torch.stack(weights, dim=0) - self.origin_weight
        self.biases = torch.stack(biases, dim=0) - self.origin_bias

        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self.epsilon_weight = nn.Parameter(torch.zeros_like(self.weights[0]))
        self.epsilon_bias = nn.Parameter(torch.zeros_like(self.biases[0]))
        
    def get_fused_weights_and_biases(self):
        alphas_normalized = F.softmax(self.alpha *  self.alpha_scale, dim=0).view(-1, 1)
        fuse_weight = self.origin_weight + (self.weights * alphas_normalized).sum(0) + self.epsilon_weight
        fuse_bias = self.origin_bias + (self.biases * alphas_normalized).sum(0) + self.epsilon_bias
        return fuse_weight, fuse_bias
    
    @torch.no_grad()
    def get_current_delta_weights_and_biases(self):
        alphas_normalized = F.softmax(self.alpha *  self.alpha_scale, dim=0).view(-1, 1)
        delta_weight = (self.weights * alphas_normalized).sum(0) + self.epsilon_weight
        delta_bias = (self.biases * alphas_normalized).sum(0) + self.epsilon_bias
        return delta_weight, delta_bias

    @torch.no_grad()
    def get_aggregated_weights_and_biases(self):
        alphas_normalized = F.softmax(self.alpha *  self.alpha_scale, dim=0).view(-1, 1)
        aggregated_weight = (self.weights * alphas_normalized).sum(0)
        aggregated_bias = (self.biases * alphas_normalized).sum(0)
        return aggregated_weight, aggregated_bias
    
    @torch.no_grad()
    def get_idx(self, idx):
        if idx < 0: idx += self.weight_nums
        # do not operate the zero index
        assert 1 <= idx <= self.weight_nums - 1, 'wrong index'
        k_weight, k_bias = self.weights[idx], self.biases[idx]
        return k_weight, k_bias
    
    def forward(self, x):
        fuse_weight, fuse_bias = self.get_fused_weights_and_biases()
        return F.layer_norm(x, self.normalized_shape, fuse_weight, fuse_bias, self.eps)
    
    @torch.no_grad()
    def save_current_weight(self, cur_alpha):
        self.weight_nums += 1
        cur_weight, cur_bias = self.get_fused_weights_and_biases()
        self.weights = torch.cat([self.weights, (cur_weight - self.origin_weight).unsqueeze(0)], dim=0)
        self.biases = torch.cat([self.biases, (cur_bias - self.origin_bias).unsqueeze(0)], dim=0)
        self.alpha = cur_alpha

        new_fuse_weight, new_fuse_bias = self.get_fused_weights_and_biases()
        self.epsilon_weight += cur_weight - new_fuse_weight
        self.epsilon_bias += cur_bias - new_fuse_bias

    @torch.no_grad()
    def remove_idx(self, cur_alpha, remove_idx):
        self.weight_nums -= 1
        cur_weight, cur_bias = self.get_fused_weights_and_biases()
        self.weights = torch.cat([self.weights[:remove_idx], self.weights[remove_idx+1:]], dim=0)
        self.biases = torch.cat([self.biases[:remove_idx], self.biases[remove_idx+1:]], dim=0)
        self.alpha = cur_alpha

        new_fuse_weight, new_fuse_bias = self.get_fused_weights_and_biases()
        self.epsilon_weight += cur_weight - new_fuse_weight
        self.epsilon_bias += cur_bias - new_fuse_bias
    
    @torch.no_grad()
    def reset(self, weight_nums, alpha, alpha_scale):
        # # default LayerNorm parameter
        # self.normalized_shape = self.normalized_shape
        # self.eps = self.eps
        self.weight_nums = weight_nums
        self.origin_weight = self.origin_weight
        self.origin_bias = self.origin_bias

        self.weights = self.weights[0:1] # [1, weight_dim]
        self.biases = self.biases[0:1] # [1, bias_dim]

        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self.epsilon_weight = nn.Parameter(torch.zeros_like(self.weights[0]))
        self.epsilon_bias = nn.Parameter(torch.zeros_like(self.biases[0]))
