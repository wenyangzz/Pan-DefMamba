import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from mamba_ssm.modules.mamba_simple import Mamba
from .refine import Refine
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
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

class OffsetNetwork(nn.Module):
    """Offset网络,生成point偏移和token index偏移"""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                                padding=kernel_size//2, groups=channels)
        self.ca = ChannelAttention(channels)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(channels, eps=1e-6)
        self.conv1x1 = nn.Conv2d(channels, 3, kernel_size=1, bias=False)  # 输出3个通道，前两个是Point的x,y偏移量，第三个是token 偏移量
    
    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        
        # 深度卷积
        x = self.dw_conv(x)
        
        # 通道注意力
        x = self.ca(x)
        
        x = self.gelu(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 生成3个通道的偏移量
        offsets = self.conv1x1(x)  # [B, 3, H, W]
        
        # 使用tanh抑制极端值
        offsets = torch.tanh(offsets)
        
        # 拆分点偏移和令牌索引偏移
        delta_p = offsets[:, :2, :, :]  # [B, 2, H, W] - 点偏移(水平和垂直)
        delta_t = offsets[:, 2:, :, :]  # [B, 1, H, W] - 令牌索引偏移
        
        # 约束点偏移在单个令牌范围内
        # delta_p[:, 0, :, :] /= w  # 水平方向除以宽度
        # delta_p[:, 1, :, :] /= h  # 垂直方向除以高度
        delta_p = torch.clone(delta_p)
        delta_p[:, 0, :, :] = delta_p[:, 0, :, :] / w  # 水平方向除以宽度
        delta_p[:, 1, :, :] = delta_p[:, 1, :, :] / h  # 垂直方向除以高度
        
        return delta_p, delta_t

class CrossDeformableScanning(nn.Module):
    """可变形扫描模块，实现跨模态交替扫描的特征点偏移和令牌索引偏移"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, pan, delta_p, delta_t):
        # x: [B, C, H, W] - ms模态特征
        # pan: [B, C, H, W] - pan模态特征
        # delta_p: [B, 2, H, W] - 点偏移（来自ms模态）
        # delta_t: [B, 1, H, W] - 令牌索引偏移（来自ms模态）
        
        b, c, h, w = x.shape
        l = h * w  # 单张图像的空间位置数量
        
        # 生成参考点并归一化到[-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        grid = grid.repeat(b, 1, 1, 1)  # [B, 2, H, W]
        
        # 计算可变形点（使用ms的偏移量同时变形两个模态）
        deformed_grid = grid + delta_p  # [B, 2, H, W]
        
        # 调整网格格式以适应F.grid_sample [B, H, W, 2]
        deformed_grid_reshaped = deformed_grid.permute(0, 2, 3, 1)
        
        # 双线性插值提取两个模态的偏移后特征
        x_deformed = F.grid_sample(
            x, 
            deformed_grid_reshaped, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )  # [B, C, H, W] - ms模态变形特征
        
        pan_deformed = F.grid_sample(
            pan, 
            deformed_grid_reshaped,  # 使用相同的变形网格
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )  # [B, C, H, W] - pan模态变形特征
        
        # 处理令牌索引偏移（仅基于ms的偏移量）
        ref_indices = torch.arange(l, device=x.device).view(1, 1, h, w).repeat(b, 1, 1, 1)  # [B, 1, H, W]
        ref_indices = ref_indices.float() / (l - 1) * 2 - 1  # 归一化到[-1, 1]
        deformed_indices = ref_indices + delta_t  # [B, 1, H, W]
        
        # 转换为排序索引
        deformed_indices_flat = deformed_indices.view(b, -1)  # [B, L]
        sorted_indices = torch.argsort(deformed_indices_flat, dim=1)  # [B, L] - 排序后的索引
        
        # 展平变形后的特征
        x_flat = x_deformed.view(b, c, -1)  # [B, C, L]
        pan_flat = pan_deformed.view(b, c, -1)  # [B, C, L]
        
        # 应用排序（两个模态使用相同的排序索引）
        x_sorted = self._gather_with_gradient_approx(x_flat, sorted_indices, dim=2, c=c)  # [B, C, L]
        pan_sorted = self._gather_with_gradient_approx(pan_flat, sorted_indices, dim=2, c=c)  # [B, C, L]
        
        # 交替排列两个模态的特征：[x0, pan0, x1, pan1, ..., xL-1, panL-1]
        combined = torch.empty(b, c, 2*l, device=x.device, dtype=x.dtype)
        combined[:, :, ::2] = x_sorted  # 偶数位置放ms特征
        combined[:, :, 1::2] = pan_sorted  # 奇数位置放pan特征
        
        # 转换为序列格式 [B, 2L, C]
        return combined.permute(0, 2, 1)


    def _gather_with_gradient_approx(self, input: torch.Tensor, index: torch.Tensor, dim: int, c) -> torch.Tensor:
        """自定义gather操作,实现梯度近似"""
        def forward_hook(input, index, dim):
            return torch.gather(input, dim, index.unsqueeze(1).repeat(1, c, 1))

        def backward_hook(grad_output, input, index, dim):
            grad_avg = grad_output.mean(dim=dim, keepdim=True)
            grad_input = grad_avg.repeat_interleave(input.shape[dim], dim=dim)
            return grad_input

        class GradApproxGather(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, index, dim):
                ctx.save_for_backward(input, index)
                ctx.dim = dim
                return forward_hook(input, index, dim)

            @staticmethod
            def backward(ctx, grad_output):
                input, index = ctx.saved_tensors
                dim = ctx.dim
                return backward_hook(grad_output, input, index, dim), None, None

        return GradApproxGather.apply(input, index, dim)
    
class DeformableScanning(nn.Module):
    """可变形扫描模块，实现特征点偏移和令牌索引偏移"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, delta_p, delta_t):
        # x: [B, C, H, W]
        # delta_p: [B, 2, H, W] - 点偏移
        # delta_t: [B, 1, H, W] - 令牌索引偏移
        
        b, c, h, w = x.shape
        
        # 生成参考点并归一化到[-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        grid = grid.repeat(b, 1, 1, 1)  # [B, 2, H, W]
       
        # 计算可变形点
        deformed_grid = grid + delta_p  # [B, 2, H, W]
        
        # 双线性插值提取偏移后的特征
        # 调整网格格式以适应F.grid_sample
        deformed_grid = deformed_grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
        deformed_features = F.grid_sample(
            x, 
            deformed_grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )  # [B, C, H, W]
      
        # 处理令牌索引偏移
        # 生成参考令牌索引
        ref_indices = torch.arange(h * w, device=x.device).view(1, 1, h, w).repeat(b, 1, 1, 1)  # [B, 1, H, W]
        # print("ref_indices shape:",ref_indices.shape)
        # print("ref_indices:",ref_indices)
        ref_indices = ref_indices.float() / (h * w - 1) * 2 - 1  # 归一化到[-1, 1]
        
        # 计算可变形令牌索引
        deformed_indices = ref_indices + delta_t  # [B, 1, H, W]
        
        # 转换为排序索引
        deformed_indices = deformed_indices.view(b, -1)  # [B, H*W]
        #print("deformed_indices:",deformed_indices)
        sorted_indices = torch.argsort(deformed_indices, dim=1)  # [B, H*W]
        #print("sorted_indices:",sorted_indices)
        # 按新顺序重排特征为序列
        flattened_features = deformed_features.view(b, c, -1)  # [B, C, H*W]
        #print("flattened_features:",flattened_features)
        # 使用gather简化索引操作
        # sequence_features = torch.gather(flattened_features, dim=2, index=sorted_indices.unsqueeze(1).repeat(1, c, 1))
        # sequence_features = sequence_features.permute(0, 2, 1)  # [B, L, C]
        
        #return sequence_features  # [B, L, C] 其中L=H*W

        sorted_seq_feat = self._gather_with_gradient_approx(
            flattened_features, sorted_indices, dim=2,c=c
        )  # [B, C, L]，带梯度近似的索引 gather

        return sorted_seq_feat.permute(0, 2, 1)  # [B, L, C]


    def _gather_with_gradient_approx(self, input: torch.Tensor, index: torch.Tensor, dim: int,c) -> torch.Tensor:
        """
        自定义gather操作,实现梯度近似
        正向传播使用正常的gather,反向传播时将输出梯度平均后复制到输入梯度
        """
        def forward_hook(input, index, dim):
            return torch.gather(input, dim, index.unsqueeze(1).repeat(1, c, 1))

        def backward_hook(grad_output, input, index, dim):
            # 对输出梯度在"索引维度"上求平均（近似梯度）
            grad_avg = grad_output.mean(dim=dim, keepdim=True)
            # 将平均值复制到所有位置，作为输入的近似梯度
            grad_input = grad_avg.repeat_interleave(input.shape[dim], dim=dim)
            return grad_input  # index无梯度

        # 使用自定义函数包装，实现正向和反向传播逻辑分离
        class GradApproxGather(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, index, dim):
                ctx.save_for_backward(input, index)
                ctx.dim = dim
                return forward_hook(input, index, dim)

            @staticmethod
            def backward(ctx, grad_output):
                input, index = ctx.saved_tensors
                dim = ctx.dim
                return backward_hook(grad_output, input, index, dim), None, None

        return GradApproxGather.apply(input, index, dim)   


class CrossModalCompressor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 简单线性融合
        self.fuse_linear = nn.Linear(2 * d_model, d_model)
        
        # 注意力融合
        # self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        # self.fuse_proj = nn.Linear(d_model, d_model)
        
        self.silu = nn.SiLU()

    def forward(self, x):
        # x: [B, 2L, C] 其中2L是交替序列长度
        b, two_l, c = x.shape
        l = two_l // 2  # 恢复原始长度
        
        # 分离两个模态的特征：[B, L, C]
        x_ms = x[:, ::2, :]  # 取偶数位置（ms模态）
        x_pan = x[:, 1::2, :]  # 取奇数位置（pan模态）
        
        # 拼接后线性压缩
        combined = torch.cat([x_ms, x_pan], dim=-1)  # [B, L, 2C]
        compressed = self.fuse_linear(combined)  # [B, L, C]
        
        # 注意力融合
        # attn_output, _ = self.attn(x_ms, x_pan, x_pan)  # ms关注pan
        # compressed = self.fuse_proj(x_ms + attn_output)  # 残差融合
        
        return self.silu(compressed)

class CrossDeformableStateSpaceModel(nn.Module):

    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        
  
        self.pre_linear = nn.Linear(128, 128)
        self.pre_linear_pan = nn.Linear(128, 128)
      
        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=kernel_size,
                                padding=kernel_size//2, groups=d_model)

        self.dw_conv2 = nn.Conv2d(d_model, d_model, kernel_size=kernel_size,
                                padding=kernel_size//2, groups=d_model)                        
        
        # 偏移网络 仅基于ms模态生成偏移
        self.offset_network = OffsetNetwork(d_model, kernel_size=kernel_size)
        
        # 跨模态可变形交替扫描
        self.deformable_scanning = CrossDeformableScanning()
        
        # 前向和后向扫描
        self.forward_scanning = self._cross_modality_scanning
        self.backward_scanning = self._cross_modality_backward_scanning
        
         # 三个分支的SSM + 压缩模块
        self.forward_ssm = Mamba(d_model, bimamba_type=None)
        self.backward_ssm = Mamba(d_model, bimamba_type=None)
        self.deformable_ssm = Mamba(d_model, bimamba_type=None)

        self.forward_compressor = CrossModalCompressor(d_model)
        self.backward_compressor = CrossModalCompressor(d_model)
        self.deformable_compressor = CrossModalCompressor(d_model)
        
        self.silu = nn.SiLU()
        self.ln = nn.LayerNorm(d_model)
        
        # 融合
        self.gate = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU()
        )

    def _cross_modality_scanning(self, x_dw, pan_dw):
        
        b, c, h, w = x_dw.shape
        l = h * w
        
        x_flat = x_dw.flatten(2)  # [B, C, L]
        pan_flat = pan_dw.flatten(2)  # [B, C, L]
        
        combined = torch.empty(b, c, 2*l, device=x_dw.device, dtype=x_dw.dtype)
        combined[:, :, ::2] = x_flat
        combined[:, :, 1::2] = pan_flat
        
        return combined.permute(0, 2, 1)  # [B, 2L, C]
    
    def _cross_modality_backward_scanning(self, x_dw, pan_dw):
        """跨模态反向扫描：先反转空间顺序，再交替取特征"""
        x_flipped = x_dw.flip(2)
        pan_flipped = pan_dw.flip(2)
        return self._cross_modality_scanning(x_flipped, pan_flipped)

    def forward(self, ms, pan):
        # 输入形状: [B, C, H, W]
        b, c, h, w = ms.shape
        l = h * w  
        
        # 线性层预处理ms模态
        # ms_flat = ms.flatten(2).permute(0, 2, 1)  # [B, L, C]
        # ms_linear = self.pre_linear(ms_flat)  # [B, L, C]
        # ms_linear = ms_linear.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]
        ms_linear = self.pre_linear(ms)
        pan_linear = self.pre_linear_pan(pan)
        # 深度卷积处理
        x_dw = self.dw_conv(ms_linear)
        x_dw = self.silu(x_dw)

        pan_dw = self.dw_conv(pan_linear)
        pan_dw = self.silu(pan_dw)
        
        # 仅基于ms模态生成偏移量
        delta_p, delta_t = self.offset_network(x_dw)
        
        # 可变形扫描分支（跨模态交替扫描）
        deformable_seq = self.deformable_scanning(x_dw, pan_dw, delta_p, delta_t)  # [B, 2L, C]
        deformable_out = self.deformable_ssm(deformable_seq)  # [B, 2L, C]
        
        # 前向扫描分支
        forward_seq = self.forward_scanning(x_dw, pan_dw)  # [B, 2L, C]
        forward_out = self.forward_ssm(forward_seq)  # [B, 2L, C]
        
        # 后向扫描分支
        backward_seq = self.backward_scanning(x_dw, pan_dw)  # [B, 2L, C]
        backward_out = self.backward_ssm(backward_seq)  # [B, 2L, C]
        
        # 压缩回原始长度L
        deformable_out = self.deformable_compressor(deformable_seq)
        forward_out = self.forward_compressor(forward_out)
        backward_out = self.backward_compressor(backward_out)

        merged = (forward_out + backward_out + deformable_out) / 3 
      
        # 恢复空间形状
        merged = merged.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]
        merged = merged * self.gate(ms) 

        merged = merged.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.ln(merged)  # 通道维度归一化
        out = out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return out

class DeformableStateSpaceModel(nn.Module):
    """完整的可变形状态空间模型"""
    def __init__(self, d_model, kernel_size=3, expand=2):
        super().__init__()
        self.d_model = d_model
        
        #深度卷积代替1D卷积
        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=kernel_size,
                                padding=kernel_size//2, groups=d_model)

        self.dw_conv2 = nn.Conv2d(d_model, d_model, kernel_size=kernel_size,
                                padding=kernel_size//2, groups=d_model)                        
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1)
        #偏移网络
        self.offset_network = OffsetNetwork(d_model, kernel_size=kernel_size)
        
        #可变形扫描
        self.deformable_scanning = DeformableScanning()
        
        #前向和后向扫描
        self.forward_scanning = lambda x: x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        self.backward_scanning = lambda x: x.flip(2).flatten(2).permute(0, 2, 1)  # 反向
        
        # 三个分支的SSM
        self.forward_ssm = Mamba(d_model, bimamba_type=None)
        self.backward_ssm = Mamba(d_model, bimamba_type=None)
        self.deformable_ssm = Mamba(d_model, bimamba_type=None)
        
        # 激活和归一化 - 修复LayerNorm维度问题
        self.silu = nn.SiLU()
        self.ln = nn.LayerNorm(d_model)  # 仅指定通道维度
        
        # 融合层
        #self.fusion = nn.Linear(3 * d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU()
        )
        # self.linearms1 = nn.Linear(d_model, d_model)
        # self.linearms2 = nn.Linear(d_model, d_model)
        # self.linearpan1 = nn.Linear(d_model, d_model)
        # self.linearpan2 = nn.Linear(d_model, d_model)

    def forward(self, ms):
        # x: [B, C, H, W] 其中C = d_model
        b, c, h, w = ms.shape
        
        # 深度卷积处理
         
        x_dw = self.dw_conv(ms)
        x_dw = self.silu(x_dw)

        # pan_dw = self.dw_conv(pan)
        # pan_dw = self.silu(pan_dw)
        #ms = self.proj(ms)

        # 生成偏移
        delta_p, delta_t = self.offset_network(x_dw)
        
        # 可变形扫描分支
        deformable_seq = self.deformable_scanning(x_dw, delta_p, delta_t)  # [B, L, C]
        deformable_out = self.deformable_ssm(deformable_seq)  # [B, L, C]
        
        # 前向扫描分支
        forward_seq = self.forward_scanning(x_dw)  # [B, L, C]
        forward_out = self.forward_ssm(forward_seq)  # [B, L, C]
        
        # 后向扫描分支
        backward_seq = self.backward_scanning(x_dw)  # [B, L, C]
        backward_out = self.backward_ssm(backward_seq)  # [B, L, C]
        #out = forward_out.permute(0, 2, 1).view(b, c, h, w)
        
        #out = self.ln(merged)
        # 分支融合
        gate = self.gate(ms)
        merged =  (forward_out + backward_out + deformable_out) / 3 
        #merged =  (forward_out + backward_out ) / 2
        # # 恢复空间形状
        merged = merged.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]
        merged =  merged * gate 
        out = merged 
        # merged = merged.permute(0, 2, 3, 1)  # [B, H, W, C]
        # out = self.ln(merged)  
        # out = out.permute(0, 3, 1, 2)  # 恢复为[B, C, H, W]
        #out = linearms2(out)
        
        return out

class SingleMambaBlock1(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock1, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)
class TokenSwapMamba1(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba1, self).__init__()
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
class CrossMamba1(nn.Module):
    def __init__(self, dim):
        super(CrossMamba1, self).__init__()
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
        ms = global_f.transpose(1, 2).view(B, C, 128, 128) 
        ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        return ms,ms_resi
class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        
        #self.encoder = Mamba(dim,bimamba_type=None) 
        self.encoder = DeformableStateSpaceModel(dim) 
        self.norm = LayerNorm(dim,'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        #print("x.shape",x.shape)
        # if isinstance(residual, torch.Tensor):
        #     print("residual.shape:", residual.shape)
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)

class CrossMambaBlock(nn.Module):
    def __init__(self, dim): 
        super(CrossMambaBlock, self).__init__()
        self.msencoder = CrossDeformableStateSpaceModel(dim)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        
    def forward(self, ms,pan
                ,ms_residual):
        ms_residual = ms+ms_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan)
        ms = self.msencoder(ms,pan)
        return ms,ms_residual

class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.msencoder = DeformableStateSpaceModel(dim)
        self.panencoder = DeformableStateSpaceModel(dim)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
    def forward(self, ms,pan
                ,ms_residual,pan_residual):
        # ms (B,C,H,W)
        #pan (B,C,H,W)
        ms_residual = ms+ms_residual
        pan_residual = pan+pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B,H,W,C = ms.shape
        # print("swapms.shape",ms.shape)
        # print("swappan.shape",pan.shape)
        ms_first_half = ms[:, :C//2, :, :]
        pan_first_half = pan[:, :C//2, :, :]
        ms_swap= torch.cat([pan_first_half,ms[:,C//2:,:,:]],dim=1)
        pan_swap= torch.cat([ms_first_half,pan[:,C//2:,:,:]],dim=1)
        # print("swapms2.shape",ms.shape)
        # print("swappan2.shape",pan.shape)
        ms_swap = self.msencoder(ms_swap)
        pan_swap = self.panencoder(pan_swap)
        return ms_swap,pan_swap,ms_residual,pan_residual

class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = CrossDeformableStateSpaceModel(dim)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,ms,ms_resi,pan):
        ms_resi = ms+ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(ms,pan)
        # B,HW,C = global_f.shape
        # ms = global_f.transpose(1, 2).view(B, C, 128, 128) 
        # ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        ms =  (self.dwconv(global_f)+global_f)
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

class ChannelAttention(nn.Module):
    """通道注意力机制，用于减轻通道冗余"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Net(nn.Module):
    def __init__(self,num_channels=None,base_filter=None,args=None):
        super(Net, self).__init__()
        base_filter=32
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.pan_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(4,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.embed_dim = base_filter*self.stride*self.patch_size
        self.shallow_fusion1 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.shallow_fusion2 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.ms_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.deep_fusion1 = CrossMamba1(self.embed_dim)
        self.deep_fusion2 = CrossMamba1(self.embed_dim)
        self.deep_fusion3 = CrossMamba1(self.embed_dim)
        self.deep_fusion4 = CrossMamba1(self.embed_dim)
        self.deep_fusion5 = CrossMamba1(self.embed_dim)
        # self.deep_fusion6 = CrossMamba(self.embed_dim)
        # self.deep_fusion7 = CrossMamba(self.embed_dim)
        # self.deep_fusion8 = CrossMamba(self.embed_dim)
        # self.deep_fusion9 = CrossMamba(self.embed_dim)
        # self.deep_fusion10 = CrossMamba(self.embed_dim)
        
        # self.pan_feature_extraction = nn.Sequential(*[SingleMambaBlock1(self.embed_dim) for i in range(4)]) 
        # self.ms_feature_extraction = nn.Sequential(*[SingleMambaBlock1(self.embed_dim) for i in range(4)])

        self.pan_feature_extraction2 = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        self.ms_feature_extraction2 = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        
        # self.pan_feature_extraction2 = nn.Sequential(*[CrossMambaBlock(self.embed_dim) for i in range(4)])
        #self.ms_feature_extraction2 = [CrossMambaBlock(self.embed_dim) for _ in range(4)]
        # self.ms_feature_extraction2 = SingleMambaBlock(self.embed_dim)
        # self.ms_feature_extraction3 = SingleMambaBlock(self.embed_dim)
        # self.ms_feature_extraction4 = SingleMambaBlock(self.embed_dim)
        # self.ms_feature_extraction5 = SingleMambaBlock(self.embed_dim)

        # self.pan_feature_extraction2 = SingleMambaBlock(self.embed_dim)
        # self.pan_feature_extraction3 = SingleMambaBlock(self.embed_dim)
        # self.pan_feature_extraction4 = SingleMambaBlock(self.embed_dim)
        # self.pan_feature_extraction5 = SingleMambaBlock(self.embed_dim)
        self.swap_mamba1 = TokenSwapMamba1(self.embed_dim)
        self.swap_mamba2 = TokenSwapMamba1(self.embed_dim)
        self.patchunembe = PatchUnEmbed(base_filter)
        self.output = Refine(base_filter,4)
    def forward(self,ms,_,pan):
        #torch.autograd.set_detect_anomaly(True)
        ms_bic = F.interpolate(ms,scale_factor=4) #upsampling 
        ms_f = self.ms_encoder(ms_bic)#([4, 32, 128, 128])
        # ms_f = ms_bic
        # pan_f = pan
        b,c,h,w = ms_f.shape
        pan_f = self.pan_encoder(pan) #([4, 32, 128, 128])
        # ms_f = self.ms_to_token(ms_f) # feature flatten ([4, 16384, 32])
        # pan_f = self.pan_to_token(pan_f) 
        
        residual_ms_f = 0
        residual_pan_f = 0 

        ms_f, residual_ms_f = self.ms_feature_extraction2([ms_f, residual_ms_f])
        pan_f, residual_pan_f = self.pan_feature_extraction2([pan_f, residual_pan_f])
      
        ms_f = ms_f.flatten(2).transpose(1, 2)
        pan_f = pan_f.flatten(2).transpose(1, 2)
        residual_ms_f = residual_ms_f.flatten(2).transpose(1, 2)
        residual_pan_f = residual_pan_f.flatten(2).transpose(1, 2)
        # ms_f = self.ms_to_token(ms_f) # feature flatten ([4, 16384, 32])
        # pan_f = self.pan_to_token(pan_f) 
        # residual_ms_f = self.pan_to_token(residual_ms_f) 
        # residual_pan_f = self.pan_to_token(residual_pan_f) 
        # residual_ms_f = 0
        # residual_pan_f = 0 
        # ms_f,residual_ms_f = self.ms_feature_extraction([ms_f,residual_ms_f]) #mamba feature_extraction ([4, 16384, 32])
        # pan_f,residual_pan_f = self.pan_feature_extraction([pan_f,residual_pan_f])
        #
        # print("ms_f1.shape",ms_f.shape)
        # print("residual_ms_f.shape",residual_ms_f.shape)
        #ms_f,residual_ms_f = self.ms_feature_extraction1([ms_f,residual_ms_f]) #mamba feature_extraction ([4, 16384, 32])
        
        #pan_f,residual_pan_f = self.pan_feature_extraction([pan_f,residual_pan_f])
    
        
    
        # print("ms_f1.shape",ms_f.shape)
        # print("pan_f2.shape",pan_f.shape)
        # print("residual_ms_f1.shape",residual_ms_f.shape)
        ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba1(ms_f,pan_f,residual_ms_f,residual_pan_f)
        ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba2(ms_f,pan_f,residual_ms_f,residual_pan_f)
        
        ms_f = self.patchunembe(ms_f,(h,w)) 
        pan_f = self.patchunembe(pan_f,(h,w)) #([4, 32, 128, 128])
       
        # print("residual_ms_f2.shape",residual_ms_f.shape)
        ms_f = self.shallow_fusion1(torch.concat([ms_f,pan_f],dim=1))+ms_f
        pan_f = self.shallow_fusion2(torch.concat([pan_f,ms_f],dim=1))+pan_f #([4, 32, 128, 128])
        
        residual_ms_f = 0
        
        ms_f = self.ms_to_token(ms_f) # feature flatten ([4, 16384, 32])
        pan_f = self.pan_to_token(pan_f)

        ms_f,residual_ms_f = self.deep_fusion1(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion2(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion3(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion4(ms_f,residual_ms_f,pan_f)
        ms_f,residual_ms_f = self.deep_fusion5(ms_f,residual_ms_f,pan_f)
        
         
        ms_f = self.patchunembe(ms_f,(h,w))
        # residual_ms_f = 0
        # pan_f = self.patchunembe(pan_f,(h,w))
        # ms_f,residual_ms_f = self.deep_fusion6(ms_f,residual_ms_f,pan_f)
        # ms_f,residual_ms_f = self.deep_fusion7(ms_f,residual_ms_f,pan_f)
        # ms_f,residual_ms_f = self.deep_fusion8(ms_f,residual_ms_f,pan_f)
        # ms_f,residual_ms_f = self.deep_fusion9(ms_f,residual_ms_f,pan_f)
        # ms_f,residual_ms_f = self.deep_fusion10(ms_f,residual_ms_f,pan_f)
        hrms = self.output(ms_f)+ms_bic
        return hrms


