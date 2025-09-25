import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from .utils import SelectiveScan,\
    flops_selective_scan_ref,print_jit_input_names, Mlp,\
    x_selective_scan, x_selective_scan_interleaved,DeformableLayer,SharedDeformableLayer, DeformableLayerReverse
import numbers
from .refine import Refine

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"



def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=True, with_Z=True, with_Group=True)
    return flops

# =====================================================
class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class DSSM(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=32,
        d_state=8,
        ssm_ratio=2.66,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        forward_type="v2",
        # ======================
        stage = 0,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.d_inner = d_inner
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.stage = stage

        self.K = 3
        self.K2 = self.K

        # in proj =======================================
        self.in_proj_ms = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.in_projp_pan = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            stride = 1
            self.conv2d_ms = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                stride=stride,
                **factory_kwargs,
            )

            self.conv2d_pan = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                stride=stride,
                **factory_kwargs,
            )


        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_proj_ms = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.out_proj_pan = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        #self.DS = DeformableLayer(index=stage, embed_dim=d_inner, debug=False)
        self.DS = SharedDeformableLayer(index=stage, embed_dim=d_inner)
        self.DR = DeformableLayerReverse()

        # other kwargs ====================================
        self.kwargs = kwargs
        if simple_init:
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        self.debug = False
        self.outnorm = None
        self.out_norm_vis = nn.LayerNorm(self.d_inner)
        self.out_norm_inf = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A) 
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D) 
        D._no_weight_decay = True
        return D

    # def forward_core(self, x: torch.Tensor, nrows=-1, channel_first=False):
    def forward_core(self, ms, pan): 
        nrows = 1
        if self.debug: debug_rec = []

        # B, D, H, W = ms.shape
        # N = H * W  # 序列长度

        # # (B, D, H, W) -> (B, D, N)
        # ms_flat = ms.flatten(2)
        # pan_flat = pan.flatten(2)

        # # 交替排列 ms 和 pan 的 token
        # # 沿着一个新的维度堆叠，形状变为 (B, D, N, 2)
        # stacked_features = torch.stack([ms_flat, pan_flat], dim=3)
        # # 展平最后两个维度，实现交替 [ms1, pan1, ms2, pan2, ...]
        # # 形状变化: (B, D, N, 2) -> (B, D, 2*N)
        # x = stacked_features.flatten(2)

        # if not channel_first:
        #     x = x.permute(0, 3, 1, 2).contiguous() # B D H W
       
        ms_out,pan_out = x_selective_scan_interleaved(
            ms, pan, self.x_proj_weight, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,DS=self.DS, DR=self.DR,
             delta_softplus=True, force_fp32=self.training, nrows=nrows,
            **self.kwargs,
        )
       
        return  ms_out,pan_out

    def forward(self, ms, pan):
        x_ms = rearrange(ms, 'b c h w -> b h w c')
        x_pan = rearrange(pan, 'b c h w -> b h w c')
        B, H, W, C = x_ms.shape

        xz_ms = self.in_proj_ms(x_ms) #(B, H, W, 2D )  D=d_expand*C
        x_ms, z_ms = xz_ms.chunk(2, dim=-1)
        xz_pan = self.in_projp_pan(x_pan) #(B, H, W, D)
        x_pan, z_pan = xz_pan.chunk(2, dim=-1)

        #b, h, w, d = x.shape
       
        x_ms = x_ms.permute(0, 3, 1, 2).contiguous() #(B, D, H, W)
        x_ms= self.act(self.conv2d_ms(x_ms))
        x_pan = x_pan.permute(0, 3, 1, 2).contiguous()
        x_pan= self.act(self.conv2d_pan(x_pan))

        x_ms,x_pan= self.forward_core(x_ms,x_pan)

        x_ms = torch.transpose(x_ms, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        x_pan = torch.transpose(x_pan, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        x_ms = self.out_norm_vis(x_ms)
        x_pan = self.out_norm_inf(x_pan)

        x_ms = x_ms * F.silu(z_ms)
        x_pan = x_pan * F.silu(z_pan)

        # out = self.dropout(self.out_proj_ms(x_ms))

        out_ms = self.out_proj_ms(x_ms)
        out_pan = self.out_proj_pan(x_pan)
        out_ms = rearrange(out_ms, 'b h w c -> b c h w')
        out_pan = rearrange(out_pan, 'b h w c -> b c h w')
        return out_ms, out_pan

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
        
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        # =============================
        stage = 0,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = DSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                simple_init=ssm_simple_init,
                # ==========================
                forward_type=forward_type,
                # =============================
                stage=stage,
                **kwargs,
            )
        self.drop_path = DropPath(drop_path)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)
        self.kwargs = kwargs

    def _forward(self, input: torch.Tensor, h_tokens=None, w_tokens=None):
        if self.ssm_branch:
            x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor,h_tokens=None, w_tokens=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input,h_tokens=None, w_tokens=None)
        else:
            return self._forward(input,h_tokens=None, w_tokens=None)

class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi

    
class MMMamba(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(MMMamba, self).__init__()
        self.norm_pan= LayerNorm(dim, LayerNorm_type)
        self.norm_ms = LayerNorm(dim, LayerNorm_type)
        self.norm_pan_2= LayerNorm(dim, LayerNorm_type)
        self.norm_ms_2 = LayerNorm(dim, LayerNorm_type)


        self.attn = DSSM(d_model=dim)
        self.ffn = FeedForward(dim,ffn_expansion_factor=2)
    def forward(self,x):
        ms,pan = x
        ms_f,pan_f = self.attn(self.norm_ms(ms),self.norm_pan(pan))
        ms = ms_f+ms
        pan = pan_f+pan
        ms = self.ffn(self.norm_ms_2(ms))+ms
        pan = self.ffn(self.norm_pan_2(pan))+pan
        return [ms,pan]
    
class Net(nn.Module):
    def __init__(self,num_channels=None,base_filter=None,args=None):
        super(Net, self).__init__()
        base_filter=32
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.pan_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(4,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.mm_mamba = nn.Sequential(MMMamba(base_filter,'BiasFree'),MMMamba(base_filter,'BiasFree'),MMMamba(base_filter,'BiasFree'),MMMamba(base_filter,'BiasFree'),MMMamba(base_filter,'BiasFree'))
        self.simple_fusion = nn.Conv2d(base_filter*2,base_filter,1,1,0)
        self.ms_head = Refine(base_filter,4)
    def forward(self, ms, _, pan):
        ms_bic = F.interpolate(ms,scale_factor=4)
        ms_f = self.ms_encoder(ms_bic)
        b,c,h,w = ms_f.shape
        # pan = torch.zeros((1,1,128,128)).to(ms_f.device)
        pan_f = self.pan_encoder(pan)
        # pan_f=torch.zeros_like(ms_f)
        # pan_f = ms_f
        x= [ms_f,pan_f]
        ms_f,pan_f = self.mm_mamba(x)
        ms_f = self.simple_fusion(torch.concat([ms_f,pan_f],dim=1))
        # ms_bic = torch.zeros_like(ms_bic)

        hrms = self.ms_head(ms_f)+ms_bic
        return hrms


    


