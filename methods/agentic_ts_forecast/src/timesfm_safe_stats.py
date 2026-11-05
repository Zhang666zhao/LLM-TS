# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training-only adaptation of TimesFM3 update_running_stats.
Source: github.com/google-research/timesfm/blob/master/src/timesfm3/torch/util.py
Changes: safe masked zero-count denominators and zero sqrt subgradient at zero.
Forward values are unchanged; installed package files are NEVER modified.
"""
import torch
class _ZeroSafeSqrt(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x):
  root=torch.sqrt(x);ctx.save_for_backward(root);return root
 @staticmethod
 def backward(ctx,grad):
  (root,)=ctx.saved_tensors;positive=root>0;denominator=2*torch.where(positive,root,torch.ones_like(root));return torch.where(positive,grad/denominator,torch.zeros_like(grad))
def update_running_stats(n,mu,sigma,x,mask):
 is_legit=~mask;is_legit_f=is_legit.float();inc_n=is_legit_f.sum(dim=-1);safe_inc_n=torch.where(inc_n==0,torch.ones_like(inc_n),inc_n)
 x_masked=torch.where(is_legit,x,torch.zeros_like(x));inc_sum=x_masked.sum(dim=-1);inc_mu=torch.where(inc_n==0,torch.zeros_like(inc_sum),inc_sum/safe_inc_n)
 x_diff_sq=torch.where(is_legit,(x-inc_mu.unsqueeze(-1))**2,torch.zeros_like(x));inc_var=torch.where(inc_n==0,torch.zeros_like(inc_sum),x_diff_sq.sum(dim=-1)/safe_inc_n);inc_sigma=_ZeroSafeSqrt.apply(inc_var)
 new_n=n+inc_n;safe_new_n=torch.where(new_n==0,torch.ones_like(new_n),new_n);new_mu=torch.where(new_n==0,torch.zeros_like(mu),(n*mu+inc_mu*inc_n)/safe_new_n)
 new_sigma=_ZeroSafeSqrt.apply(torch.where(new_n==0,torch.zeros_like(sigma),(n*sigma*sigma+inc_n*inc_sigma*inc_sigma+n*(mu-new_mu)*(mu-new_mu)+inc_n*(inc_mu-new_mu)*(inc_mu-new_mu))/safe_new_n))
 return new_n,new_mu,new_sigma
def install():
 from timesfm3.torch import util
 original=util.update_running_stats;assert original is not update_running_stats,'Already installed in this process';util.update_running_stats=update_running_stats;return original
def restore(original):
 from timesfm3.torch import util
 util.update_running_stats=original
