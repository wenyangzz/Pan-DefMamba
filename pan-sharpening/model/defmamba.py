import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
class ChannelAttention(nn.Module):
    """通道注意力机制，用于减轻通道冗余"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class OffsetNetwork(nn.Module):
    """偏移网络，生成点偏移和令牌索引偏移"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                                padding=kernel_size//2, groups=channels)
        self.ca = ChannelAttention(channels)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(channels, eps=1e-6)
        self.conv1x1 = nn.Conv2d(channels, 3, kernel_size=1, bias=False)  # 输出3个通道
    
    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        
        # 深度卷积提取局部特征
        x = self.dw_conv(x)
        
        # 通道注意力
        x = self.ca(x)
        
        # 激活和归一化
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
        delta_p[:, 0, :, :] /= w  # 水平方向除以宽度
        delta_p[:, 1, :, :] /= h  # 垂直方向除以高度
        
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
            return grad_input, None

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
                return backward_hook(grad_output, input, index, dim)

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
        print("ref_indices shape:",ref_indices.shape)
        print("ref_indices:",ref_indices)
        ref_indices = ref_indices.float() / (h * w - 1) * 2 - 1  # 归一化到[-1, 1]
        
        # 计算可变形令牌索引
        deformed_indices = ref_indices + delta_t  # [B, 1, H, W]
        
        # 转换为排序索引
        deformed_indices = deformed_indices.view(b, -1)  # [B, H*W]
        print("deformed_indices:",deformed_indices)
        sorted_indices = torch.argsort(deformed_indices, dim=1)  # [B, H*W]
        print("sorted_indices:",sorted_indices)
        # 按新顺序重排特征为序列
        flattened_features = deformed_features.view(b, c, -1)  # [B, C, H*W]
        print("flattened_features:",flattened_features)
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
            return grad_input, None  # index无梯度

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
                return backward_hook(grad_output, input, index, dim)

        return GradApproxGather.apply(input, index, dim)   

class SSM(nn.Module):
    """状态空间模型,基于Mamba架构"""
    def __init__(self, d_model, dt_rank="auto", dim_inner=None, expand=2):
        super().__init__()
        self.d_model = d_model
        self.dim_inner = dim_inner if dim_inner is not None else d_model * expand
        self.dt_rank = dt_rank if dt_rank != "auto" else max(d_model // 16, 1)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.dim_inner * 2)
        
        # SSM参数
        self.dt_proj = nn.Linear(self.dim_inner, self.dt_rank)
        self.A_log = nn.Parameter(torch.log(torch.ones(self.dt_rank, self.dim_inner // 2)))
        self.D = nn.Parameter(torch.ones(self.dim_inner // 2))
        self.out_proj = nn.Linear(self.dim_inner // 2, d_model)
        
    def forward(self, x):
        # x: [B, L, D]
        b, l, d = x.shape
        
        # 输入投影
        x_and_res = self.in_proj(x)  # [B, L, 2*dim_inner]
        x, res = x_and_res.split([self.dim_inner, self.dim_inner], dim=-1)
        x = F.silu(x)
        
        # 拆分门控
        x1, x2 = x.split(self.dim_inner // 2, dim=-1)  # [B, L, dim_inner//2]
        
        # 计算delta
        dt = self.dt_proj(x)  # [B, L, dt_rank]
        dt = F.softplus(dt)  # 确保dt为正
        
        # 计算A
        A = -torch.exp(self.A_log)  # [dt_rank, dim_inner//2]
        
        # 状态空间计算
        y = self.ssm(x1, dt, A, self.D)
        
        # 门控和输出投影
        y = y * F.silu(res[..., :self.dim_inner // 2])
        y = self.out_proj(y)
        
        return y
    
    def ssm(self, x, dt, A, D):
        # x: [B, L, N] where N = dim_inner//2
        # dt: [B, L, R] where R = dt_rank
        # A: [R, N]
        # D: [N]
        
        b, l, n = x.shape
        r = dt.shape[-1]
        
        # 计算状态更新
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # [B, L, R, N]
        dt_A_cumsum = torch.cumsum(dt_A, dim=1)  # [B, L, R, N]
        alpha = torch.exp(dt_A_cumsum)  # [B, L, R, N]
        
        x_input = x.unsqueeze(2) * dt.unsqueeze(-1)  # [B, L, R, N]
        x_input = x_input * torch.exp(-dt_A_cumsum + dt_A)  # [B, L, R, N]
        
        # 累积和
        beta = torch.cumsum(x_input, dim=1)  # [B, L, R, N]
        
        # 加权组合
        y = (alpha * beta).mean(dim=2)  # [B, L, N]
        
        # 残差连接
        y = y + x * D.unsqueeze(0).unsqueeze(0)
        
        return y
class CrossModalCompressor(nn.Module):
    """跨模态特征压缩模块：将2L长度序列压缩回L长度"""
    def __init__(self, d_model):
        super().__init__()
        # 选项1：简单线性融合（适合轻量化场景）
        self.fuse_linear = nn.Linear(2 * d_model, d_model)
        
        # 选项2：注意力融合（适合需要建模模态依赖的场景）
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
        
        # 选项1：拼接后线性压缩
        combined = torch.cat([x_ms, x_pan], dim=-1)  # [B, L, 2C]
        compressed = self.fuse_linear(combined)  # [B, L, C]
        
        # 选项2：注意力融合（如需使用请注释选项1并取消注释此处）
        # attn_output, _ = self.attn(x_ms, x_pan, x_pan)  # ms关注pan
        # compressed = self.fuse_proj(x_ms + attn_output)  # 残差融合
        
        return self.silu(compressed)

class CrossDeformableStateSpaceModel(nn.Module):
    """完整的可变形状态空间模型"""
    def __init__(self, d_model, kernel_size=3, expand=2):
        super().__init__()
        self.d_model = d_model
        
        # 在深度卷积前添加线性层
        self.pre_linear = nn.Linear(d_model, d_model)
        
        # 深度卷积
        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=kernel_size,
                                padding=kernel_size//2, groups=d_model)

        self.dw_conv2 = nn.Conv2d(d_model, d_model, kernel_size=kernel_size,
                                padding=kernel_size//2, groups=d_model)                        
        
        # 偏移网络（仅基于ms模态生成偏移）
        self.offset_network = OffsetNetwork(d_model, kernel_size=kernel_size)
        
        # 可变形扫描（修改为支持跨模态交替扫描）
        self.deformable_scanning = DeformableScanning()
        
        # 前向和后向扫描（跨模态交替扫描）
        self.forward_scanning = self._cross_modality_scanning
        self.backward_scanning = self._cross_modality_backward_scanning
        
         # 三个分支的SSM + 压缩模块（关键：确保序列长度不变）
        self.forward_ssm = Mamba(d_model, bimamba_type=None)
        self.backward_ssm = Mamba(d_model, bimamba_type=None)
        self.deformable_ssm = Mamba(d_model, bimamba_type=None)

        self.forward_compressor = CrossModalCompressor(d_model)
        self.backward_compressor = CrossModalCompressor(d_model)
        self.deformable_compressor = CrossModalCompressor(d_model)
        
        # 激活和归一化
        self.silu = nn.SiLU()
        self.ln = nn.LayerNorm(d_model)
        
        # 融合层
        self.gate = nn.Sequential(
            nn.Linear(32, 32),
            nn.SiLU()
        )

    def _cross_modality_scanning(self, x_dw, pan_dw):
        """跨模态前向扫描:交替取x_dw和pan_dw的对应位置特征"""
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
        l = h * w  # 单模态空间位置数量
        
        # 线性层预处理ms模态
        ms_flat = ms.flatten(2).permute(0, 2, 1)  # [B, L, C]
        ms_linear = self.pre_linear(ms_flat)  # [B, L, C]
        ms_linear = ms_linear.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]
        
        # 深度卷积处理
        x_dw = self.dw_conv(ms_linear)
        x_dw = self.silu(x_dw)

        pan_dw = self.dw_conv(pan)
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
        
        #偏移网络
        self.offset_network = OffsetNetwork(d_model, kernel_size=kernel_size)
        
        #可变形扫描
        self.deformable_scanning = DeformableScanning()
        
        #前向和后向扫描
        self.forward_scanning = lambda x: x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        self.backward_scanning = lambda x: x.flip(2).flatten(2).permute(0, 2, 1)  # 反向
        
        # 三个分支的SSM
        self.forward_ssm = SSM(d_model, expand=expand)
        self.backward_ssm = SSM(d_model, expand=expand)
        self.deformable_ssm = SSM(d_model, expand=expand)
        
        # 激活和归一化 - 修复LayerNorm维度问题
        self.silu = nn.SiLU()
        self.ln = nn.LayerNorm(d_model)  # 仅指定通道维度
        
        # 融合层
        #self.fusion = nn.Linear(3 * d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(32, 32),
            nn.SiLU()
        )
        # self.linearms1 = nn.Linear(d_model, d_model)
        # self.linearms2 = nn.Linear(d_model, d_model)
        # self.linearpan1 = nn.Linear(d_model, d_model)
        # self.linearpan2 = nn.Linear(d_model, d_model)

    def forward(self, ms,pan):
        # x: [B, C, H, W] 其中C = d_model
        b, c, h, w = ms.shape
        
        # 深度卷积处理
         
        x_dw = self.dw_conv(ms)
        x_dw = self.silu(x_dw)

        pan_dw = self.dw_conv(pan)
        pan_dw = self.silu(pan_dw)
        

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
        
        # 分支融合
        gate = self.gate(ms)
        merged =  (forward_out + backward_out + deformable_out) / 3 
        
        # 恢复空间形状
        merged = merged.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]
        merged =  merged * gate 

        # 修复LayerNorm输入维度：将通道维度放到最后
        #merged_plus_x = merged + x  # [B, C, H, W]
        #merged_plus_x = merged_plus_x.permute(0, 2, 3, 1)  # [B, H, W, C]
        merged = merged.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.ln(merged)  # 对通道维度归一化
        out = out.permute(0, 3, 1, 2)  # 恢复为[B, C, H, W]
        #out = linearms2(out)
        
        return out
# 测试模型
if __name__ == "__main__":
    d_model = 32
    height, width = 32, 32
    batch_size = 2
    

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型迁移到GPU
    model = DeformableStateSpaceModel(d_model=d_model).to(device)
    # 输入张量迁移到GPU
    ms = torch.randn(batch_size, d_model, height, width).to(device)
    pan = torch.randn(batch_size, d_model, height, width).to(device)


    output = model(ms, pan)
    
    print(f"输入形状: {ms.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == ms.shape, "输出形状与输入形状不匹配"
    
    # # 测试序列长度压缩
    # compressor = CrossModalCompressor(d_model)
    # doubled_seq = torch.randn(batch_size, 2*height*width, d_model)
    # compressed_seq = compressor(doubled_seq)
    # print(f"压缩前长度: {doubled_seq.shape[1]}, 压缩后长度: {compressed_seq.shape[1]}")
    # assert compressed_seq.shape[1] == height*width, "压缩后长度不正确"
    
    # print("模型测试成功!")


    print("模型测试成功!")