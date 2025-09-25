import torch
import torch.nn as nn
import math
from functools import partial
from einops import rearrange, repeat
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import einops


try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class DeformablePathTrans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, de_index):
        B, C, N = x.shape
        _, indices = torch.topk(de_index, k=N, dim=-1, largest=False)
        x_gathered = torch.gather(x, 2, indices.unsqueeze(1).expand(-1, C, -1)).contiguous()
        x_out = x_gathered.permute(0, 2, 1).contiguous()
        ctx.save_for_backward(x, de_index, indices)
        return x_out, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        x, de_index, indices = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_x.scatter_add_(2, indices.unsqueeze(1).expand(-1, x.shape[1], -1), grad_output.permute(0, 2, 1).contiguous()).contiguous()
        grad_de_index = (grad_output.permute(0, 2, 1).contiguous()-grad_x).mean(dim=1)
        grad_de_index = grad_de_index.view_as(de_index)
        return grad_x, grad_de_index


class ConvOffset(nn.Module):
    def __init__(self, embed_dim, kk, pad_size):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kk, 1, pad_size, groups=embed_dim)
        self.ca = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//16),
                nn.GELU(),
                nn.Linear(embed_dim//16, embed_dim),
                nn.Sigmoid()
                )
        self.ln = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim, 3, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x_c = F.adaptive_avg_pool2d(x, (1, 1))
        x_c = self.ca(x_c.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x1 * x_c.expand_as(x)
        x = self.gelu(self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = self.conv2(x)
        return x


class DeformableLayer(nn.Module):
    def __init__(
            self, index=0, embed_dim=192, debug=False, h=0, w=0):
        super().__init__()
        self.ksize = [5, 5, 5, 5]
        self.stride = 1
        kk = self.ksize[index]
        pad_size = kk // 2 if kk != 1 else 0
        self.debug = debug
        self.conv_offset = ConvOffset(embed_dim, kk, pad_size)
        self.rpe_table = nn.Parameter(
            torch.zeros(embed_dim, 7, 7)
        )
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1)  # B H W 2

        return ref

    @torch.no_grad()
    def _get_key_ref_points(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0, H, H, dtype=dtype, device=device),
            torch.linspace(0, W, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1)  # B H W 2

        return ref

    @torch.no_grad()
    def _get_path_ref_points(self, N, B, dtype, device):
        ref_path = torch.linspace(0.5, N - 0.5, N, dtype=dtype, device=device),
        ref_path[0].div_(N - 1.0).mul_(2.0).sub_(1.0)
        ref = ref_path[0][None, ...].expand(B, -1)  # B H W 1
        return ref

    @torch.no_grad()
    def _get_path_key_ref_points(self, N, B, dtype, device):
        ref_path = torch.linspace(0, N, N, dtype=dtype, device=device),
        ref_path[0].div_(N - 1.0).mul_(2.0).sub_(1.0)
        ref = ref_path[0][None, ...].expand(B, -1).flatten(1)  # B H W 1
        return ref

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        dtype, device = x.dtype, x.device
        B, C, H, W = x.size()
        N = H * W

        offset = self.conv_offset(x).contiguous()  # B 2 Hg Wg
        offset, de_index = torch.split(offset, [2, 1], dim=1)  # 将偏移量拆分为坐标偏移（前2通道）和路径索引偏移（后1通道)
        Hk, Wk = offset.size(2), offset.size(3)
        # 计算偏移量的归一化范围（将偏移量映射到[-1,1])
        offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range)
        # 获取参考点坐标（初始采样网格），形状为(B, Hk, Wk, 2)，坐标已归一化到[-1,1]
        offset = einops.rearrange(offset, 'b p h w -> b h w p').contiguous()
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        # 对路径索引偏移应用tanh激活，并展平为(B, Hg*Wg)
        de_index = de_index.tanh().flatten(1)
        path_reference = self._get_path_ref_points(N, B, dtype, device)# 获取路径参考点（用于路径变形的初始序列位置），形状为(B, N)，N=H*W
        # 计算最终采样坐标：参考点 + 偏移量（形状为B, Hk, Wk, 2）
        pos = offset + reference
        # 计算最终路径位置：路径参考点 + 路径索引偏移（形状为B, N）
        path_pos = de_index + path_reference

        x_sampled = F.grid_sample(
            input=x,
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B, C, Hg, Wg
        # 处理相对位置偏差（RPE）
        rpe_table = self.rpe_table 
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        rpe_bias = F.interpolate(rpe_bias, size=(H, W), mode='bilinear', align_corners=False)
        key_grid = self._get_key_ref_points(H, W, B, dtype, device)
        displacement = (key_grid - pos) * 0.5
        pos_bias = F.grid_sample(
            input=rpe_bias,
            grid=displacement[..., (1, 0)],
            mode='bilinear', align_corners=True
        )
        x = x_sampled + pos_bias
        x = x.flatten(2)
         # 应用路径变形转换，根据path_pos调整特征序列顺序，返回变形后的特征和索引
        x, indices = DeformablePathTrans.apply(x, path_pos) #B N C
        return x, indices


# SharedDeformableLayer 
# ===============================================================
class SharedDeformableLayer(nn.Module):
    def __init__(self, index=0, embed_dim=32, h=0, w=0):
        super().__init__()
        self.ksize = [3, 3, 3, 3]
        kk = self.ksize[index]
        pad_size = kk // 2 if kk != 1 else 0
        self.conv_offset = ConvOffset(embed_dim, kk, pad_size)
        self.rpe_table = nn.Parameter(torch.zeros(embed_dim, 7, 7))
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device), torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device), indexing='ij')
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        return ref[None, ...].expand(B, -1, -1, -1)

    @torch.no_grad()
    def _get_path_ref_points(self, N, B, dtype, device):
        ref_path = torch.linspace(0.5, N - 0.5, N, dtype=dtype, device=device)
        ref_path = ref_path.div_(N - 1.0).mul_(2.0).sub_(1.0)
        return ref_path[None, ...].expand(B, -1)

    def forward(self, x_ms, x_pan):
        dtype, device = x_ms.dtype, x_ms.device
        B, C, H, W = x_ms.size()
        N = H * W

        # 1. 核心：只使用 x_ms 计算共享路径
        offset_full = self.conv_offset(x_ms).contiguous()
        offset, de_index = torch.split(offset_full, [2, 1], dim=1)
        
        Hk, Wk = offset.size(2), offset.size(3)
        offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range)
        pos = einops.rearrange(offset, 'b p h w -> b h w p') + self._get_ref_points(Hk, Wk, B, dtype, device)
        path_pos = de_index.tanh().flatten(1) + self._get_path_ref_points(N, B, dtype, device)

        # 2. 计算共享的相对位置偏置 (RPE)
        rpe_table = self.rpe_table 
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        rpe_bias = F.interpolate(rpe_bias, size=(H, W), mode='bilinear', align_corners=False)
        displacement = (self._get_ref_points(H, W, B, dtype, device) - pos) * 0.5
        pos_bias = F.grid_sample(rpe_bias, displacement[..., (1, 0)], mode='bilinear', align_corners=True)

        # 3. 将共享路径和偏置分别应用于 ms 和 pan
        ms_sampled = F.grid_sample(x_ms, pos[..., (1, 0)], mode='bilinear', align_corners=True) + pos_bias
        pan_sampled = F.grid_sample(x_pan, pos[..., (1, 0)], mode='bilinear', align_corners=True) + pos_bias
        
        # 4. 使用 DeformablePathTrans 对两者进行相同的排序
        ms_reordered, indices = DeformablePathTrans.apply(ms_sampled.flatten(2), path_pos)
        pan_reordered, _ = DeformablePathTrans.apply(pan_sampled.flatten(2), path_pos) #B, N, C

        return ms_reordered, pan_reordered, indices

class DeformableLayerReverse(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, indices=None):
        x = x.flatten(2)
        B, C, N = x.size()
        index_re = torch.zeros_like(indices, device=x.device)
        index_re.scatter_add_(1, indices, torch.arange(indices.size(-1), device=x.device).unsqueeze(0).expand(indices.size(0), -1))
        x = torch.gather(x, 2, index_re.unsqueeze(1).expand(-1, C, -1))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ===============================================================
# 3. 最终的跨模态扫描函数
# ===============================================================
def x_selective_scan_interleaved(
        x_ms: torch.Tensor, 
        x_pan: torch.Tensor,
        x_proj_weight: torch.Tensor, 
        dt_projs_weight: torch.Tensor,
        dt_projs_bias: torch.Tensor,
        A_logs: torch.Tensor, 
        Ds: torch.Tensor,
        DS: None, 
        DR: None,
        delta_softplus=True, 
        force_fp32=True,
        nrows=1,
):
    K, D, R = dt_projs_weight.shape #3 170 2
    B, D, H, W = x_ms.shape
    N = H * W
    _, d_state = A_logs.shape

    # 1. 特殊扫描：使用共享路径获取已排序的序列
    ms_deformed, pan_deformed, shared_indices = DS(x_ms, x_pan) # B N D
    
    # 2. 准备所有扫描序列
    xs_ms, xs_pan = torch.empty(B, 3, D, N, device=x_ms.device), torch.empty(B, 3, D, N, device=x_ms.device)
    # print("ms_deformed.shape",ms_deformed.shape)
    # print("x_ms.shape",x_ms.shape)
    
    xs_ms[:, 0], xs_pan[:, 0] = x_ms.flatten(2), x_pan.flatten(2)
    xs_ms[:, 1], xs_pan[:, 1] = torch.flip(xs_ms[:, 0], [-1]), torch.flip(xs_pan[:, 0], [-1])
    xs_ms[:, 2], xs_pan[:, 2] = ms_deformed.permute(0, 2, 1), pan_deformed.permute(0, 2, 1)

    # 3. 交错融合所有序列
    # print("xs_ms.shape",xs_ms.shape)
    # print("dt_projs_weight.shape",dt_projs_weight.shape)
    interleaved_xs = torch.stack([xs_ms, xs_pan], dim=4).flatten(3)
    #print("interleaved_xs.shape",interleaved_xs.shape)
    # 4. 参数投影
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", interleaved_xs, x_proj_weight)
    dts, Bs, Cs = torch.split(x_dbl, [R, d_state, d_state], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    u = interleaved_xs.contiguous().view(B, -1, 2 * N)
    delta = dts.contiguous().view(B, -1, 2 * N)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds_in = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        u, delta, Bs, Cs = [t.to(torch.float) for t in [u, delta, Bs, Cs]]

    def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys_interleaved = selective_scan_fn(
        u, delta, As, Bs, Cs, Ds_in, delta_bias, delta_softplus, nrows
    )
    
    separated_ys = ys_interleaved.view(B, K, D, N, 2)
    ys_ms, ys_pan = separated_ys[..., 0], separated_ys[..., 1]

    def aggregate_results(ys, indices):
        y_deformed_reversed = DR(ys[:, 2], indices)
        y = (ys[:, 0] + ys[:, 1].flip(dims=[-1]) + y_deformed_reversed) / 3.0
        return y.transpose(1, 2).contiguous().view(B, H, W, D)

    y_ms = aggregate_results(ys_ms, shared_indices)
    y_pan = aggregate_results(ys_pan, shared_indices)
    
    return y_ms.to(x_ms.dtype), y_pan.to(x_ms.dtype)



    
def x_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        force_fp32=True,
        stage=0,
        DS=None,
        DR=None,
        **kwargs,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
    K, D, R = dt_projs_weight.shape
    B, D, H, W = x.shape
    L = H * W
    _, N = A_logs.shape

    xs = x.new_empty((B, 3, D, H * W))
    xs[:, 0] = x.flatten(2, 3)
    xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
    temp, indices = DS(x)
    xs[:, 2] = temp.permute(0, 2, 1)  

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    )

    ys = ys.view(B, K, -1, H, W)
    ys = ys.view(B, K, D, -1) 
    y = (ys[:, 0] + ys[:, 1].flip(dims=[-1]) + DR(ys[:, 2], indices)) / 3.

    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    if K!=1: y = y.view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y), {'ys':ys, 'xs':xs, 'dts':dts, 'As':A_logs, 'Bs':Bs, 'Cs':Cs, 'Ds':Ds, 'delta_bias':delta_bias}


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)




