import torch
import torch.nn as nn	
import torch.nn.functional as F
import math

class SimpleEmbedding(nn.Module):
	def __init__(self, in_dim=200):
		super().__init__()
		
		self.extract_features = nn.Sequential(
					nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
					nn.GELU(),
					nn.GroupNorm(5, 25),
					
					nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
					nn.GELU(),
					nn.GroupNorm(5, 25),
					
					nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
					nn.GELU(),
					nn.GroupNorm(5, 25)		
				)
				
		self.register_buffer('mask_encoding', torch.zeros(in_dim, dtype=torch.float32))
		
	def forward(self, x, mask=None):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		if mask is None:
			mask_x = x
		else:
			mask_x = x.clone()
			mask_x[mask == 1] = self.mask_encoding

		mask_x = mask_x.reshape(Bz, 1, num_chans * num_bands * num_patch, patch_size)	
		out = self.extract_features(mask_x)

		out = out.permute(0, 2, 1, 3).contiguous().view(Bz, num_chans * num_bands * num_patch, 1, -1).contiguous()

		out = out.reshape(Bz, num_chans, num_bands, num_patch, patch_size)	
		
		return out

class InterBandConv(nn.Module):
	def __init__(self, d_model=200, kernel_size=3):
		super().__init__()
		
		self.conv = nn.Sequential(nn.Conv2d(in_channels=d_model, out_channels=d_model, stride=(1, 1), kernel_size=(kernel_size, 1), padding=((kernel_size - 1) // 2, 0), groups=d_model))
	
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		
		x = x.permute(0, 1, 3, 2, 4).contiguous()
		x = x.reshape(Bz * num_chans, 1, num_patch, num_bands, patch_size)
		
		x = x.reshape(Bz * num_chans * num_patch, 1, num_bands, patch_size)
		x = x.permute(0, 3, 2, 1).contiguous()
		
		out = self.conv(x)
		out = out.permute(0, 3, 1, 2).contiguous()
		
		out = out.reshape(Bz, num_chans, num_patch, patch_size, num_bands)
		
		out = out.permute(0, 1, 4, 2, 3).contiguous()
		
		return out
		
class InterChannelConvACPE(nn.Module):
	def __init__(self, d_model=200):
		super().__init__()
		self.conv = nn.Sequential(nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), padding=(9, 3), stride=(1, 1), groups=d_model))	
		
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		x = x.permute(0, 2, 1, 3, 4).contiguous()
		
		x = x.reshape(Bz * num_bands, num_chans, num_patch, patch_size)
		x = x.permute(0, 3, 1, 2).contiguous()	
		
		out = self.conv(x)
		out = out.reshape(Bz, num_bands, patch_size, num_chans, num_patch)
		
		out = out.permute(0, 3, 1, 4, 2).contiguous()
		
		return out

class TemEmbedSpcLayer(nn.Module):
	def __init__(self, convolution_set=[(1,), (3,), (5,)], d_model=200, stride=1):
		super().__init__()
		
		self.convolution_set = convolution_set
		
		dims = [d_model // 2 ** i for i in range(1, len(self.convolution_set))]	
		self.dim_scales = [*dims, d_model - sum(dims)]
		
		self.embed_layers = nn.ModuleList([nn.Conv2d(in_channels=d_model, out_channels=dim_scale, stride=(stride, 1), kernel_size=(kernel_size, 1), padding=((kernel_size - 1) // 2, 0))
					for (kernel_size, ), dim_scale in zip(self.convolution_set, self.dim_scales)
				])
		
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape	
		x = x.reshape(Bz * num_chans, num_bands, num_patch, patch_size)	
		
		x = x.reshape(Bz * num_chans * num_bands, 1, num_patch, patch_size)
		x = x.permute(0, 3, 2, 1).contiguous()
		
		f_maps = [conv(x) for conv in self.embed_layers]
		out = torch.cat(f_maps, dim=1)
		
		out = out.permute(0, 3, 2, 1).contiguous()
		out = out.reshape(Bz * num_chans, num_bands, num_patch, patch_size)
		
		out = out.reshape(Bz, num_chans, num_bands, num_patch, patch_size)
		
		return out		
		
class TemporalAttention(nn.Module):
	def __init__(self, convolution_set=[(1,), (3,), (5,)], d_model=200, num_heads=8, dropout=0.1, batch_first=True):
		super().__init__()
		
		self.convolution_set = convolution_set
		self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
	
	def forward(self, x, is_causal=False, need_key_padding=False):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		
		window_size = min(num_patch, self.convolution_set[-1][0])
		original_num_patch = num_patch	
		
		padded_x = x
		padding_size = 0
			
		if num_patch % window_size != 0:
			padding_size = window_size - num_patch % window_size	
			
			padded_x = F.pad(x, (0, 0, 0, padding_size))	
			num_patch = num_patch + padding_size	
			
		num_windows = num_patch // window_size
		
		key_padding_mask = None
		attn_mask = None
		
		if need_key_padding:
			mask_list = [False] * original_num_patch + [True] * padding_size
			key_padding_mask = torch.tensor(mask_list, dtype=torch.bool, device=x.device)
			
			key_padding_mask = key_padding_mask.view(1, 1, 1, -1).expand(Bz, num_chans, num_bands, num_patch)
			
			key_padding_mask = key_padding_mask.reshape(Bz, num_chans, num_bands, num_windows, window_size)	
			key_padding_mask = key_padding_mask.permute(0, 1, 2, 4, 3).contiguous()	
			
			key_padding_mask = key_padding_mask.reshape(Bz * num_chans * num_bands * window_size, num_windows)	
			
		if is_causal:
			attn_mask = torch.triu(torch.ones(num_windows, num_windows) * float('-inf'), diagonal=1)
			
			attn_mask = attn_mask.to(x.dtype)
			attn_mask = attn_mask.to(x.device)
			
		padded_x = padded_x.reshape(Bz, num_chans, num_bands, num_windows, window_size, patch_size)
		padded_x = padded_x.permute(0, 1, 2, 4, 3, 5).contiguous()
		
		padded_x = padded_x.reshape(Bz * num_chans * num_bands * window_size, num_windows, patch_size)
		attn_out = self.attn(padded_x, padded_x, padded_x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]
		
		attn_out = attn_out.reshape(Bz, num_chans, num_bands, window_size, num_windows, patch_size)
		attn_out = attn_out.permute(0, 1, 2, 4, 3, 5).contiguous()
		
		attn_out = attn_out.reshape(Bz, num_chans, num_bands, num_windows * window_size, patch_size)
		out = attn_out[:, :, :, :original_num_patch, :]	
		
		return out	

class CrossInterChannelAttention(nn.Module):
	def __init__(self, d_model=200, dropout=0.1, num_heads=8, batch_first=True):
		super().__init__()
		
		self.attn = nn.MultiheadAttention(embed_dim=d_model, dropout=dropout, num_heads=num_heads, batch_first=batch_first)
			
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		
		x = x.permute(0, 2, 3, 1, 4).contiguous()
		x = x.reshape(Bz * num_bands * num_patch, num_chans, patch_size)
			
		out = self.attn(x, x, x)[0]	
		out = out.reshape(Bz, num_bands, num_patch, num_chans, patch_size)
		out = out.permute(0, 3, 1, 2, 4).contiguous()
		return out

class PhaseEmbedding(nn.Module):
	def __init__(self, in_dim=200, d_model=200):
		super().__init__()
		self.extract_phase_features = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=d_model, kernel_size=(1, 1), stride=(1, 1)))
		
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		x = x.reshape(Bz, num_chans * num_bands, num_patch, patch_size)
		x = x.permute(0, 3, 1, 2).contiguous()
		out = self.extract_phase_features(x)
		out = out.permute(0, 2, 3, 1).contiguous()
		out = out.reshape(Bz, num_chans, num_bands, num_patch, patch_size)
		return out
		
class CrossAttention(nn.Module):
	def __init__(self, d_model=200, num_heads=8, dropout=0.1, batch_first=True, d_ffn=800):
		super().__init__()
		
		self.CustomAttention = CustomAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, d_ffn=d_ffn)
		
	def forward(self, x, phase_cos, phase_sin, need_cross_attn_mask=True):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		
		x = x.permute(0, 3, 1, 2, 4).contiguous()
		x = x.reshape(Bz * num_patch, num_chans * num_bands, patch_size)
		
		phase_cos = phase_cos.permute(0, 3, 1, 2, 4).contiguous()
		phase_cos = phase_cos.reshape(Bz * num_patch, num_chans * num_bands, patch_size)
		
		phase_sin = phase_sin.permute(0, 3, 1, 2, 4).contiguous()
		phase_sin = phase_sin.reshape(Bz * num_patch, num_chans * num_bands, patch_size)
		
		attn_mask = None
		
		if need_cross_attn_mask:
			attn_mask = torch.triu(torch.ones(num_bands, num_bands) * float('-inf'), diagonal=1).to(x.device)
			attn_mask = attn_mask.to(x.dtype)
			
			attn_mask = attn_mask.repeat(num_chans, num_chans)
			
		out = self.CustomAttention(x, phase_cos, phase_sin, attn_mask=attn_mask)
		
		out = out.reshape(Bz, num_patch, num_chans, num_bands, patch_size)
		out = out.permute(0, 2, 3, 1, 4).contiguous()
		
		return out
class CrissAttention(nn.Module):
	def __init__(self, d_model=200, num_heads=8, dropout=0.1, d_ffn=800):
		super().__init__()
		self.CustomAttention = CustomAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, d_ffn=d_ffn)
	def forward(self, x, phase_cos, phase_sin):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		x = x.reshape(Bz * num_chans * num_bands, num_patch, patch_size)
		phase_cos = phase_cos.reshape(Bz * num_chans * num_bands, num_patch, patch_size)
		phase_sin = phase_sin.reshape(Bz * num_chans * num_bands, num_patch, patch_size)
		attn_mask = None
		#if need_criss_attn_mask:
		out = self.CustomAttention(x, phase_cos, phase_sin, attn_mask=attn_mask)
		out = out.reshape(Bz, num_chans, num_bands, num_patch, patch_size)	
		return out
	
class CustomAttention(nn.Module):
	def __init__(self, d_model=200, num_heads=8, dropout=0.1, d_ffn=800):
		super().__init__()
		
		self.d_k = d_model // num_heads
		self.Wq = nn.Linear(d_model, d_model)
		
		self.Wk_cos = nn.Linear(d_model, d_model)
		self.Wk_sin = nn.Linear(d_model, d_model)
		
		self.Wv_cos = nn.Linear(d_model, d_model)
		self.Wv_sin = nn.Linear(d_model, d_model)
		
		self.dropout = nn.Dropout(dropout)
		self.num_heads = num_heads
	
	def forward(self, x, phase_cos, phase_sin, attn_mask=None):
		Batch, Seq_Len, Features = x.shape
		
		def shape_for_heads(t):
			return t.reshape(Batch, Seq_Len, self.num_heads, self.d_k).transpose(1, 2)
			
		Q = shape_for_heads(self.Wq(x))	
		
		K_cos = shape_for_heads(self.Wk_cos(phase_cos))	
		K_sin = shape_for_heads(self.Wk_sin(phase_sin))
		
		K = K_cos + K_sin
		
		V_cos = shape_for_heads(self.Wv_cos(phase_cos))	
		V_sin = shape_for_heads(self.Wv_sin(phase_sin))	
		
		V = V_cos + V_sin
		
		scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)	
		
		if attn_mask is not None:
			scores = scores + attn_mask
			
		attention = torch.nn.functional.softmax(scores, dim=-1)	
		attention = self.dropout(attention)	
		
		ctx = torch.matmul(attention, V)
		ctx = ctx.transpose(1, 2)
		
		ctx = ctx.reshape(Batch, Seq_Len, Features)	
		
		return ctx

class _ff_block(nn.Module):
	def __init__(self, in_dim=200, d_ffn=800, out_dim=200):
		super().__init__()
		
		self.FFN = nn.Sequential(
						nn.Linear(in_dim, d_ffn),
						nn.GELU(),
						nn.Dropout(0.1),
						nn.Linear(d_ffn, out_dim),
					)
	def forward(self, x):
		out = self.FFN(x)
		
		return out
		
class SpcEncoderLayer(nn.Module):
	def __init__(self, dropout=0.1, d_model=200, convolution_set=[(1, ), (3, ), (5, )], stride=1, num_heads=8, d_ffn=800):
		super().__init__()
		
		self.norm1 = nn.LayerNorm(d_model)	
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.norm4 = nn.LayerNorm(d_model)
		self.norm5 = nn.LayerNorm(d_model)
		
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)
		self.dropout4 = nn.Dropout(dropout)
		self.dropout5 = nn.Dropout(dropout)
		#python3 -m torch.distributed.run --nproc_per_node=2 pretrain_spc.py --batch_size 32

		self.transformer_FFN = _ff_block(in_dim=d_model, d_ffn=d_ffn, out_dim=d_model)
		self.TemEmbedSpcLayer = TemEmbedSpcLayer(d_model=d_model, convolution_set=convolution_set, stride=stride)
		
		self.TemporalAttention = TemporalAttention(d_model=d_model, convolution_set=convolution_set, dropout=dropout, num_heads=num_heads)
		self.CrossInterChannelAttention = CrossInterChannelAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
		
		self.CrissAttention = CrissAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, d_ffn=d_ffn)
		self.CrossAttention = CrossAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, d_ffn=d_ffn)
		
	def forward(self, x, phase_cos, phase_sin, is_causal=False, need_key_padding=False, need_cross_attn_mask=True):	
		out_conv = x + self.TemEmbedSpcLayer(x)
		
		out_tmp_attn = out_conv + self.dropout1(self.TemporalAttention(self.norm1(out_conv), is_causal=is_causal, need_key_padding=need_key_padding))
		
		out_chn_attn = out_tmp_attn + self.dropout2(self.CrossInterChannelAttention(self.norm2(out_tmp_attn)))
		
		out_crs_attn = out_chn_attn + self.dropout4(self.CrossAttention(self.norm4(out_chn_attn), phase_cos, phase_sin, need_cross_attn_mask=False))
		
		out_cris_attn = out_crs_attn + self.dropout3(self.CrissAttention(self.norm3(out_crs_attn), phase_cos, phase_sin))
		
		out = out_cris_attn + self.dropout5(self.transformer_FFN(self.norm5(out_cris_attn)))
		
		return out
		
class Final(nn.Module):
	def __init__(self, num_layers=6, d_model=200, kernel_size=3, in_dim=200, out_dim=200, convolution_set=[(1,), (3,), (5,)], stride=1, num_heads=8, d_ffn=800, dropout=0.1, num_chans=29):	
		super().__init__()
		self.num_layers = num_layers
		self.SimpleEmbedding = SimpleEmbedding(in_dim=in_dim)
		
		self.InterBandConv = InterBandConv(d_model=d_model, kernel_size=kernel_size)
		self.InterChannelConvACPE = InterChannelConvACPE(d_model=d_model)
		self.norm_final = nn.LayerNorm(d_model)
		
		self.PhaseEmbedding = PhaseEmbedding(in_dim=in_dim, d_model=d_model)
		
		self.SpcEncoderLayers = nn.ModuleList([SpcEncoderLayer(dropout=dropout, d_model=d_model, convolution_set=convolution_set, stride=stride, num_heads=num_heads, d_ffn=d_ffn) for _ in range(num_layers)])	
		
		#self.proj_out_spc = _ff_block(in_dim=d_model * 3, d_ffn=d_ffn, out_dim=d_model)
		
		self.proj_out_spc2 = _ff_block(in_dim=num_chans * 5, d_ffn=d_ffn, out_dim=num_chans)
		
	def forward(self, amp, phase_cos, phase_sin, spc_mask=None, is_causal=False, need_key_padding=False, need_cross_attn_mask=True):
		Bz, num_chans, num_bands, num_patch, patch_size = amp.shape
		amp = self.SimpleEmbedding(amp, spc_mask)
		
		phase_cos = self.PhaseEmbedding(phase_cos)
		phase_sin = self.PhaseEmbedding(phase_sin)
		amp = self.InterBandConv(amp)
		
		amp = self.InterChannelConvACPE(amp)
		
		for i in range(self.num_layers):
			amp = self.SpcEncoderLayers[i](amp, phase_cos, phase_sin, is_causal=is_causal, need_key_padding=need_key_padding, need_cross_attn_mask=need_cross_attn_mask)
			
		amp = self.norm_final(amp)
		#amp_phase = torch.cat([amp, phase_cos, phase_sin], dim=-1)
		
		#out = self.proj_out_spc(amp_phase)
		amp = amp.reshape(Bz, num_chans * num_bands, num_patch, patch_size)
		
		amp = amp.permute(0, 3, 2, 1).contiguous()
		out = self.proj_out_spc2(amp)
		
		out = out.permute(0, 3, 2, 1).contiguous()
		
		
		return out

