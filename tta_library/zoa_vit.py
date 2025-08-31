"""
Copyright to FOA Authors ICML 2024
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit



class ZOA_ViT(nn.Module):
    """variant of ZOA (Zeroth-Order Test-Time Adaptation)
    ZOA_Fuse uses the same loss as FOA, while it updates norm layers with zeroed-order optimization (SPSA-GC)
    """
    def __init__(self, args, model, fitness_lambda=30):
        super().__init__()

        self.args = args
        self.steps = args.steps
        self.model = model
        self.embed_dim = model.embed_dim
        self.num_features = model.num_features
        self.fitness_lambda = fitness_lambda
        
        self.hist_stat = None
        
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
        shift_vector = self._get_shift_vector()
        outputs, batch_mean, loss, domain_change = forward_and_get_loss(x, self.model, self.fitness_lambda, 
                                                                        self.train_info, shift_vector)
        if domain_change:
            self.model.logger.info(f'Reset shift vector')
            self.hist_stat = None
            shift_vector = None

        alphas, params, embed_dims = self.model.parameters_to_vector()

        self.params_num = len(params) // self.embed_dim
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

        self._update_hist(batch_mean[-self.model.num_features:])
        return outputs

    def obtain_origin_stat(self, train_loader):
        print('===> begin calculating mean and variance')
        self.model.eval()
        
        features = []
        with torch.no_grad():
            for _, dl in enumerate(train_loader):
                images = dl[0].cuda()
                feature = self.model.layers_cls_features(images)
                features.append(feature)
            features = torch.cat(features, dim=0)
            self.train_info = torch.std_mean(features, dim=0) # occupy 0.2MB 
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

def copy_model_only(model):
    source_model = deepcopy(model)
    for param in source_model.parameters():
        param.detach_()
    return source_model

criterion_mse = nn.MSELoss(reduction='mean').cuda()

@torch.no_grad()  # disable grads context for fast testing
def forward_and_get_loss(images, model, fitness_lambda, train_info, shift_vector):
    features, domain_change = model.layers_cls_features_with_prompts(images)
    
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])
    
    discrepancy_loss = fitness_lambda * (std_mse + mean_mse)

    cls_features = features[:, -model.num_features:]
    del features

    output = model.model.head(cls_features)
    entropy_loss = softmax_entropy(output).mean()

    loss = discrepancy_loss + entropy_loss

    if shift_vector is not None:
        output = model.model.head(cls_features + 1. * shift_vector)

    return output, batch_mean, loss, domain_change


def spsa_grad_estimate_bi(inputs, model, alphas, params, fitness_lambda, train_info, shift_vector=None, ck_alpha=0.001, ck=0.001, sp_avg=5, loss=None):
    #* repeat k times and average them for stabilizing
    ghats_alpha, ghats = [], []
    N_alphas = len(alphas)
    N_params = len(params)

    for si in range(sp_avg):
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
    
    # average gradients
    ghats_alpha = torch.cat(ghats_alpha, dim=0).mean(dim=0)
    ghat = torch.cat(ghats, dim=0).mean(dim=0) 
    loss = ((loss1+loss2)/2)

    return ghats_alpha, ghat, loss

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

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

    fix_layers = ['model.blocks.0.norm1', 'model.blocks.0.norm2']
    fix_layers += ['model.blocks.9', 'model.blocks.10', 'model.blocks.11', 'model.norm']
    for name, param in model.named_parameters():
        for fix_name in fix_layers:
            if fix_name in name:
                param.requires_grad_(False)

    return model

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names