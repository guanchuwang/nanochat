import torch
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _get_value,
                        _fused_doc, _maximize_doc, _default_to_fused_or_foreach)
from typing import List, Optional, Tuple, Union

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
        epoch=0, 
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        self.epoch = epoch
        super().__init__(params, defaults)

    def update_epoch(self, epoch):
        self.epoch = epoch

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if (g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p)):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

def exclude_bias_and_norm(p):
    return p.ndim == 1

class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warm_up_steps, warm_up_base_lr_divider, cosine_scheduler, warm_up_finished_func=None):
        self.warm_up_steps = warm_up_steps
        self.cosine_scheduler = cosine_scheduler
        self.last_step = 0
        self.highest_lr = [group['lr'] for group in optimizer.param_groups]
        self.finished_warming_up = False
        self.warm_up_finished_func = warm_up_finished_func

        self.base_lr_divider = warm_up_base_lr_divider
        if self.base_lr_divider == -1:
            self._last_lr = [0.0 for _ in self.highest_lr]
        else:
            self._last_lr = [lr / self.base_lr_divider for lr in self.highest_lr]

        super(WarmUpCosineAnnealingLR, self).__init__(optimizer)
        
    def step(self):
        self.last_step += 1
        super().step()

    def get_lr(self):
        if self.warm_up_steps != 0 and self.last_step <= self.warm_up_steps:
            if self.base_lr_divider == -1: # Warm-up from 0 to highest_lr
                warmup_lr = [
                    lr * (self.last_step / self.warm_up_steps)
                    for lr in self.highest_lr
                ]
            else: # Warm-up from lr / base_lr_divider to highest_lr
                warmup_lr = [
                    (lr / self.base_lr_divider) +
                    (lr - (lr / self.base_lr_divider)) * (self.last_step / self.warm_up_steps)
                    for lr in self.highest_lr
                ]
            self._last_lr = warmup_lr
            return warmup_lr
        else:
            if not self.finished_warming_up:
                self.finished_warming_up = True
                if self.warm_up_finished_func is not None:
                    self.warm_up_finished_func()
            # Proceed with the cosine scheduler after warm-up
            self.cosine_scheduler.step()
            self._last_lr = self.cosine_scheduler.get_last_lr()
            return self._last_lr
            
    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {
            'warm_up_steps': self.warm_up_steps,
            'last_lr': self._last_lr,
            'last_step': self.last_step,
            'finished_warming_up': self.finished_warming_up,
            'cosine_scheduler_state_dict': self.cosine_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.warm_up_steps = state_dict['warm_up_steps']
        self._last_lr = state_dict['last_lr']
        self.last_step = state_dict['last_step']
        self.finished_warming_up = state_dict['finished_warming_up']
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler_state_dict'])
            
class StableAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=None, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(StableAdamW, self).__init__(params, defaults)

    def update_epoch(self, epoch):
        self.epoch = epoch

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Correct bias for moving averages
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Apply weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute debiased moving averages
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                RMSt = corrected_exp_avg_sq.sqrt().add_(group['eps'])
                eta_t = group['lr'] / max(1, RMSt.mean())  # Adjust learning rate dynamically

                # Update parameters with the dynamically adjusted learning rate
                # Adjusting the formula to fit typical implementation patterns
                p.data.mul_(1 - eta_t * group['weight_decay'])
                p.data.addcdiv_(corrected_exp_avg, RMSt, value=-eta_t)

class StableAdamWUnfused(torch.optim.Optimizer): # gotten from https://gist.github.com/mitchellnw/d42e22a0b9ec02ceaf4f7b4457f51423

    def __init__(self, params, lr=0.002, weight_decay=0.2, betas=(0.9, 0.99), eps=1e-6, clip_thresh=1., custom_scalar=65536):
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(StableAdamWUnfused, self).__init__(params, defaults)

        self.eps=eps
        self.d = clip_thresh

        # Set precision to "custom_fp16" if you want to use a fixed loss scalar, custom_scalar, which is divided out in the update step.
        # If you do this, call (custom_scalar * loss).backward() instead of loss.backward().
        self.precision = None
        self.custom_scaler = custom_scalar

        for group in self.param_groups:
            group['step'] = 1.
        
        print('Using StableAdamWUnfused-v1')
    def update_epoch(self, epoch):
        # self.epoch = epoch
        pass

    def __setstate__(self, state):
        super(StableAdamWUnfused, self).__setstate__(state)
        
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:

            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            step = group['step']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                theta=p.data
                param_state = self.state[p]

                if self.precision == 'custom_fp16':
                    g = p.grad.data / self.custom_scaler
                    if torch.any(torch.isnan(g) | torch.isinf(g)):
                        continue
                else:
                    g = p.grad.data

                if 'exp_avg' not in param_state:
                    v = param_state['exp_avg'] = torch.zeros_like(theta)
                    u = param_state['exp_avg_sq'] = torch.zeros_like(theta)
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']

                beta1hat = beta1 * (1 - beta1**(step - 1)) / (1 - beta1**step)
                beta2hat = beta2 * (1 - beta2**(step - 1)) / (1 - beta2**step)
                    
                v = v.mul_(beta1hat).add_(g, alpha=1.0-beta1hat)
                u = u.mul_(beta2hat).addcmul_(g,g,value=1.0-beta2hat)

                denominator = u.sqrt().add_(self.eps)
                    
                # StableAdamW = AdamW + update clipping (https://arxiv.org/abs/1804.04235) applied tensor-wise.
                rms = torch.div(
                    g.pow(2), 
                    torch.maximum(u, (self.eps ** 2) * torch.ones_like(u))
                ).mean().sqrt().item()
                
                new_lr = lr * (1. / max(1., rms / self.d ))

                theta = theta.mul_(1.0-new_lr*weight_decay).addcdiv_(
                    v, 
                    denominator, 
                    value=-new_lr
                )

                # save current params
                param_state['exp_avg'] = v
                param_state['exp_avg_sq'] = u
            
            group['step'] = step + 1