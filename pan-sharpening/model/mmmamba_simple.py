import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from mamba_ssm.modules.mamba_simple import Mamba
from .refine import Refine
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')



import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numbers
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat


class Attention(nn.Module):
    def __init__(
            self,
            d_model,
            window_size,
            d_state=8,
            d_conv=3,
            expand=2.66,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj_vis = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_inf = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d_vis = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act_vis = nn.SiLU()

        self.conv2d_inf = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act_inf = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm_vis = nn.LayerNorm(self.d_inner)
        self.out_norm_inf = nn.LayerNorm(self.d_inner)
        self.out_proj_vis = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_inf = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def create_patches(self, image_tensor, w, order='ltr_utd'):
        """
        按照指定顺序生成图像patch
        order参数:
        - 'ltr_utd': 从左到右，再从上到下
        - 'rtl_dtu': 从下到上，再从右到左
        - 'utd_ltr': 从上到下，再从左到右
        - 'dtu_rtl': 从右到左，再从下到上
        """
        B, C, H, W = image_tensor.shape
        Hg, Wg = math.ceil(H / w), math.ceil(W / w)

        # 确保图像尺寸能被w整除
        assert H % w == 0 and W % w == 0, f"图像尺寸({H}x{W})必须能被patch大小({w})整除"
        
        # 重塑为patch网格
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous() # 转为BHWC
        patches = image_tensor.view(B, Hg, w, Wg, w, C)  # [B, Hg, w, Wg, w, C]
        
        # 根据不同顺序调整patch排列
        if order == 'ltr_utd':  # 从左到右，再从上到下
            # 原始顺序：patch按行优先排列，每个patch内按行优先
            patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, Hg, Wg, w, w]
        
        elif order == 'rtl_dtu':  # 从下到上，再从右到左
            # 翻转patch顺序和每个patch内的像素顺序
            patches = patches.flip(1, 3).permute(0, 5, 1, 3, 2, 4).flip(4, 5).contiguous()
        
        elif order == 'utd_ltr':  # 从上到下，再从左到右
            # patch按列优先排列，每个patch内按列优先
            patches = patches.permute(0, 5, 3, 1, 4, 2).contiguous()  # [B, C, Wg, Hg, w, w]
        
        elif order == 'dtu_rtl':  # 从右到左，再从下到上
            # 翻转patch顺序和每个patch内的像素顺序
            patches = patches.flip(1, 3).permute(0, 5, 3, 1, 4, 2).flip(4, 5).contiguous()
        
        else:
            raise ValueError(f"Unsupported order: {order}")
        
        return patches
    

    def get_scan(self, x_vis, x_inf, orders):
        output = []
        for order in orders:
            x_vis_patches = self.create_patches(x_vis, self.window_size, order)
            x_inf_patches2 = self.create_patches(x_inf, self.window_size, order)

            # 展平每个patch为向量，但保留patch的网格结构
            B, C, Hg, Wg, w1, w2 = x_vis_patches.shape
            x_vis_patches_flat = x_vis_patches.reshape(B, C, Hg, Wg, w1*w2)  # [B, C, Hg, Wg, w*w]
            x_inf_patches2_flat = x_inf_patches2.reshape(B, C, Hg, Wg, w1*w2)  # [B, C, Hg, Wg, w*w]

            # 在每个patch位置交替合并两个图像的patch
            # 先将两个图像的patch在最后一维拼接，然后reshape成所需的交替形式
            merged_patches = torch.cat([x_vis_patches_flat.unsqueeze(-2), x_inf_patches2_flat.unsqueeze(-2)], dim=-2)
            merged_patches = merged_patches.reshape(B, C, Hg, Wg, -1)  # [B, C, Hg, Wg, w*w*2]
    
            # 展平所有patch
            merged_sequence = merged_patches.reshape(B, C, -1).unsqueeze(1)
            output.append(merged_sequence)
        output = torch.cat(output, dim=1)
        return output

    def reconstruct_images(self, final_result, w, x_vis, orders):
        B, C, H, W = x_vis.shape
        L = H * W
        Hg, Wg = math.ceil(H / w), math.ceil(W / w)
        patch_size = w * w
        sequences = [final_result[:, i] for i in range(4)]

        y_vis = None
        y_inf = None

        for i, order in enumerate(orders):
            seq = sequences[i].reshape(B, C, Hg, Wg, 2, patch_size)
            patches1 = seq[:, :, :, :, 0, :].reshape(B, C, Hg, Wg, w, w)
            patches2 = seq[:, :, :, :, 1, :].reshape(B, C, Hg, Wg, w, w)

            if order == 'ltr_utd':
                patches1 = patches1.permute(0, 1, 2, 4, 3, 5).contiguous()  # [B, Hg, Wg, w, w, C]
                patches2 = patches2.permute(0, 1, 2, 4, 3, 5).contiguous()
                img1 = patches1.view(B, C, Hg*w, Wg*w)
                img2 = patches2.view(B, C, Hg*w, Wg*w)
            elif order == 'rtl_dtu':
                patches1 = patches1.flip(2, 3)  # 翻转patch的行和列顺序（Hg和Wg维度）
                patches2 = patches2.flip(2, 3)
                
                patches1 = patches1.flip(4, 5)  # 翻转每个patch内的像素（w和w维度）
                patches2 = patches2.flip(4, 5)
                
                img1 = patches1.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hg*w, Wg*w)
                img2 = patches2.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hg*w, Wg*w)

            elif order == 'utd_ltr':
                patches1 = patches1.permute(0, 1, 2, 4, 3, 5).contiguous()
                patches2 = patches2.permute(0, 1, 2, 4, 3, 5).contiguous()
                img1 = patches1.view(B, C, Hg*w, Wg*w).transpose(-1, -2)
                img2 = patches2.view(B, C, Hg*w, Wg*w).transpose(-1, -2)
            
            elif order == 'dtu_rtl':
                patches1 = patches1.flip(2, 3)  # 翻转patch的行和列顺序（Hg和Wg维度）
                patches2 = patches2.flip(2, 3)
                
                patches1 = patches1.flip(4, 5)  # 翻转每个patch内的像素（w和w维度）
                patches2 = patches2.flip(4, 5)
                
                img1 = patches1.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hg*w, Wg*w).transpose(2, 3)
                img2 = patches2.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hg*w, Wg*w).transpose(2, 3)

            if y_vis == None:
                y_vis = img1.contiguous().view(B, -1, L).contiguous()
                y_inf = img2.contiguous().view(B, -1, L).contiguous()
            else:
                y_vis = y_vis + img1.contiguous().view(B, -1, L).contiguous()
                y_inf = y_inf + img2.contiguous().view(B, -1, L).contiguous()

        return y_vis, y_inf


    def forward_core(self, x_vis, x_inf):
        B, C, H, W = x_vis.shape
        L = H * W * 2
        K = 4

        orders = ['ltr_utd', 'rtl_dtu', 'utd_ltr', 'dtu_rtl']
        xs = self.get_scan(x_vis, x_inf, orders)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        
        return self.reconstruct_images(out_y, self.window_size, x_vis, orders)

    def forward(self, visible, infrared):
        x_vis = rearrange(visible, 'b c h w -> b h w c')
        x_inf = rearrange(infrared, 'b c h w -> b h w c')

        B, H, W, C = x_vis.shape

        xz_vis = self.in_proj_vis(x_vis) # b h w 2d 
        x_vis, z_vis = xz_vis.chunk(2, dim=-1) # b h w d

        xz_inf = self.in_proj_vis(x_inf)
        x_inf, z_inf = xz_inf.chunk(2, dim=-1)

        x_vis = x_vis.permute(0, 3, 1, 2).contiguous() #b d h w 
        x_vis = self.act_vis(self.conv2d_vis(x_vis))

        x_inf = x_inf.permute(0, 3, 1, 2).contiguous()
        x_inf = self.act_inf(self.conv2d_inf(x_inf))    

        y_vis, y_inf = self.forward_core(x_vis, x_inf)

        y_vis = torch.transpose(y_vis, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_inf = torch.transpose(y_inf, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y_vis = self.out_norm_vis(y_vis)
        y_inf = self.out_norm_inf(y_inf)

        y_vis = y_vis * F.silu(z_vis)
        y_inf = y_inf * F.silu(z_inf)

        out_vis = self.out_proj_vis(y_vis)
        out_inf = self.out_proj_inf(y_inf)
        out_vis = rearrange(out_vis, 'b h w c -> b c h w')
        out_inf = rearrange(out_inf, 'b h w c -> b c h w')
        return out_vis, out_inf

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
    
class MMMamba(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(MMMamba, self).__init__()
        self.norm_pan= LayerNorm(dim, LayerNorm_type)
        self.norm_ms = LayerNorm(dim, LayerNorm_type)
        self.norm_pan_2= LayerNorm(dim, LayerNorm_type)
        self.norm_ms_2 = LayerNorm(dim, LayerNorm_type)


        self.attn = Attention(dim,window_size=4)
        self.ffn =FeedForward(dim,ffn_expansion_factor=2)
    def forward(self,x):
        ms,pan = x
        ms_f,pan_f = self.attn(self.norm_ms(ms),self.norm_pan(pan))
        ms = ms_f+ms
        pan = pan_f+pan
        ms = self.ffn(self.norm_ms_2(ms))+ms
        pan = self.ffn(self.norm_pan_2(pan))+pan
        return [ms,pan]
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms, pan):
        b, c, h, w = ms.shape

        kv = self.kv_dwconv(self.kv(pan))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(ms))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm_cro1= LayerNorm(dim, LayerNorm_type)
        self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.cro = CrossAttention(dim,num_heads,bias)
        self.proj = nn.Conv2d(dim,dim,1,1,0)
    def forward(self, ms,pan):
        ms = ms+self.cro(self.norm_cro1(ms),self.norm_cro2(pan))
        ms = ms + self.ffn(self.norm2(ms))
        return ms


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
# ---------------------------------------------------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x
class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)
import random
class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.msencoder = Mamba(dim,bimamba_type=None)
        self.panencoder = Mamba(dim,bimamba_type=None)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
    def forward(self, ms,pan
                ,ms_residual,pan_residual):
        # ms (B,N,C)
        #pan (B,N,C)
        ms_residual = ms+ms_residual
        pan_residual = pan+pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B,N,C = ms.shape
        ms_first_half = ms[:, :, :C//2]
        pan_first_half = pan[:, :, :C//2]
        ms_swap= torch.cat([pan_first_half,ms[:,:,C//2:]],dim=2)
        pan_swap= torch.cat([ms_first_half,pan[:,:,C//2:]],dim=2)
        ms_swap = self.msencoder(ms_swap)
        pan_swap = self.panencoder(pan_swap)
        return ms_swap,pan_swap,ms_residual,pan_residual
class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,ms,ms_resi,pan):
        ms_resi = ms+ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms),extra_emb=self.norm2(pan))
        B,HW,C = global_f.shape
        ms = global_f.transpose(1, 2).view(B, C, 128*8, 128*8)
        ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        return ms,ms_resi
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
    #42.3 wv2
    # def forward(self,ms,_,pan):
    #     ms_bic = F.interpolate(ms,scale_factor=4)
    #     ms_f = self.ms_encoder(ms_bic)
    #     b,c,h,w = ms_f.shape
    #     pan_f = self.pan_encoder(pan)

    #     x= [ms_f,pan_f]
    #     ms_f,pan_f = self.mm_mamba(x)
    #     hrms = self.ms_head(ms_f)+ms_bic
    #     hrms = self.pan_head(pan_f)
    #     return hrms


