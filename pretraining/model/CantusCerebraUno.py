import pywt
import ptwt
import torch
import torch.nn as nn
import numpy as np
import mne
import torch.nn.functional as F
from model.TmpEncoderUno import *
from model.SpcEncoderUno import *

class _ff_block(nn.Module):
	def __init__(self, d_model=200, d_ffn=800):
	
		super().__init__()
		
		self.FFN = nn.Sequential(
						nn.Linear(d_model, d_ffn),
						nn.GELU(),
						nn.Dropout(0.1),
						nn.Linear(d_ffn, d_model),
					)
	def forward(self, x):
	
		out = self.FFN(x)
		return out
	
class CrossAttentionTemporal(nn.Module):	# It is going to be strided cross attention both for temporal and functional. So, you'd have two CrossAttentions. One for temporal and one for functional.
	def __init__(self, convolution_set=[(1,), (3,), (5,)], d_model=200, num_heads=8, dropout=0.1):
	
		super().__init__()
		self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
		
		self.convolution_set = convolution_set
		#self.hook_handle = self.attn.register_full_backward_hook(self.backward_hook)
	
#	def backward_hook(self, module, grad_input, grad_output):
#		print(f"\n--- Backward Hook Triggered on {module.__class__.__name__} ---")
#		if grad_input[0] is not None:
#			print(f"Gradient of Loss w.r.t Module Input: \n{grad_input[0]}")
#		if grad_output[0] is not None:
#			print(f"Gradient of Loss w.r.t Module Output: \n{grad_output[0]}")
		
	def forward(self, t1, t2):
		if t1.ndim < t2.ndim:
		
			Bz, num_chans, num_bands, num_patch, patch_size = t2.shape
			window_size = self.convolution_set[-1][0]
			
			original_num_patch = num_patch
			
			if num_patch % window_size != 0:
				padding_size = window_size - num_patch % window_size
				
				t1 = F.pad(t1, (0, 0, 0, padding_size))
				t2 = F.pad(t2, (0, 0, 0, padding_size))
				
				num_patch = num_patch + padding_size
				
			num_windows = num_patch // window_size
			
			t1 = t1.reshape(Bz, num_chans, num_windows, window_size, patch_size)
			t2 = t2.reshape(Bz, num_chans, num_bands, num_windows, window_size, patch_size)
			
			t1 = t1.permute(0, 1, 3, 2, 4).contiguous()
			t2 = t2.permute(0, 1, 4, 2, 3, 5).contiguous()
			
			t1 = t1.reshape(Bz * num_chans * window_size, num_windows, patch_size)
			t2 = t2.reshape(Bz * num_chans * window_size, num_bands * num_windows, patch_size)
			
			out = self.attn(t1, t2 ,t2)[0]
			
			final = out.reshape(Bz, num_chans, window_size, num_windows, patch_size)
			final = final.permute(0, 1, 3, 2, 4).contiguous()
			
			final = final.reshape(Bz, num_chans, -1, patch_size)
			
			return final[:, :, :original_num_patch, :]
			
		else:
			Bz, num_chans, num_bands, num_patch, patch_size = t1.shape
			
			window_size = self.convolution_set[-1][0]
			original_num_patch = num_patch
			
			if num_patch % window_size != 0:
				padding_size = window_size - num_patch % window_size
				
				t1 = F.pad(t1, (0, 0, 0, padding_size))
				t2 = F.pad(t2, (0, 0, 0, padding_size))
				
				num_patch = num_patch + padding_size
				
			num_windows = num_patch // window_size
			
			t1 = t1.reshape(Bz, num_chans, num_bands, num_windows, window_size, patch_size)
			t2 = t2.reshape(Bz, num_chans, num_windows, window_size, patch_size)
			
			t1 = t1.permute(0, 1, 4, 2, 3, 5).contiguous()
			t2 = t2.permute(0, 1, 3, 2, 4).contiguous()
			
			t1 = t1.reshape(Bz * num_chans * window_size, num_bands * num_windows, patch_size)
			t2 = t2.reshape(Bz * num_chans * window_size, num_windows, patch_size)
			
			out = self.attn(t1, t2, t2)[0]
			
			final = out.reshape(Bz, num_chans, window_size, num_bands, num_windows, patch_size)
			final = final.permute(0, 1, 3, 4, 2, 5).contiguous()
			
			final = final.reshape(Bz, num_chans, num_bands, num_patch, patch_size)
			
			return final[:, :, :, :original_num_patch, :]

class CrossAttentionFunctional(nn.Module):
	def __init__(self ,sorted_map, d_model=200, dropout=0.1, num_heads=8):
	
		super().__init__()
		self.register_buffer('sorted_map', sorted_map)
		
		self.attn = nn.MultiheadAttention(embed_dim=d_model, dropout=dropout, batch_first=True, num_heads=num_heads)
		self.group_transform = nn.Linear(d_model, d_model)
		
		self._precompute_matrices()
		
		#self.hook_handle = self.attn.register_full_backward_hook(self.backward_hook)
	
	#def backward_hook(self, module, grad_input, grad_output):
		#print(f"\n--- Backward Hook Triggered on {module.__class__.__name__} ---")
		#if grad_input[0] is not None:
		#	print(f"Gradient of Loss w.r.t Module Input: \n{grad_input[0]}")
		#if grad_output[0] is not None:
		#	print(f"Gradient of Loss w.r.t Module Output: \n{grad_output[0]}")
	
	def _precompute_matrices(self):
		num_chans = self.sorted_map.shape[0]
		K = self.sorted_map.shape[1]
		num_groups = K // 4
		
		col_idx_mean = self.sorted_map[:, :num_groups * 4].reshape(-1, 4).long()	 
		col_idx_first = self.sorted_map[:, 0:num_groups * 4:4].reshape(-1, 1).long()
		
		W_mean = torch.zeros(num_chans * num_groups, num_chans, device=self.sorted_map.device)
		W_first = torch.zeros(num_chans * num_groups, num_chans, device=self.sorted_map.device)

		W_mean.scatter_(dim=1, index=col_idx_mean, value=0.25)

		W_first.scatter_(dim=1, index=col_idx_first, value=1.0)
		
		self.register_buffer('W_mean', W_mean.T)   
		self.register_buffer('W_first', W_first.T) 
		
		self.num_chans = num_chans
		self.num_groups = num_groups
	
	def forward(self, t1, t2):
		if t1.ndim < t2.ndim:
		
			Bz, num_chans, num_bands, num_patch, patch_size = t2.shape
			t2 = t2.permute(0, 2, 3, 4, 1)
			
			t2 = t2.reshape(Bz * num_bands * num_patch, patch_size, num_chans)
			mean_groups = torch.matmul(t2, self.W_mean).transpose(1, 2)
			
			first_tensors = torch.matmul(t2, self.W_first).transpose(1, 2)
			final_stacked = first_tensors + self.group_transform(mean_groups)
			
			final_stacked = final_stacked.reshape(Bz, num_bands, num_patch, num_chans, self.num_groups, patch_size)
			final_stacked = final_stacked.permute(0, 3, 2, 4, 1, 5)
			
			final_stacked = final_stacked.reshape(Bz * num_chans * num_patch, self.num_groups * num_bands, patch_size)
			t1 = t1.reshape(Bz * num_chans * num_patch, 1, patch_size)
			
			out = self.attn(t1, final_stacked, final_stacked)[0]
			out = out.reshape(Bz, num_chans, num_patch, patch_size)
			
			return out	
		else:
		
			Bz, num_chans, num_bands, num_patch, patch_size = t1.shape
			t2 = t2.permute(0, 2, 3, 1)
			
			t2 = t2.reshape(Bz * num_patch, patch_size, num_chans)
			mean_groups = torch.matmul(t2, self.W_mean).transpose(1, 2)
			
			first_tensors = torch.matmul(t2, self.W_first).transpose(1, 2)
			final_stacked = first_tensors + self.group_transform(mean_groups)
			
			final_stacked = final_stacked.reshape(Bz, num_patch, num_chans, self.num_groups, patch_size)
			final_stacked = final_stacked.permute(0, 2, 1, 3, 4)
			
			final_stacked = final_stacked.reshape(Bz * num_chans * num_patch, self.num_groups, patch_size)
			t1 = t1.permute(0, 1, 3, 2, 4)
			t1 = t1.reshape(Bz * num_chans * num_patch, num_bands, patch_size)
			
			out = self.attn(t1, final_stacked, final_stacked)[0]
			out = out.reshape(Bz, num_chans, num_patch, num_bands, patch_size)
			
			out = out.permute(0, 1, 3, 2, 4).contiguous()
			
			return out
			
class CrossAttention(nn.Module):
	def __init__(self, sorted_map, d_model=200, dropout=0.1, convolution_set=[(1,), (3,), (5,)], num_heads=8, d_ffn=800):
		super().__init__()
		
		self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
		self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])
		
		self.CrossAttentionTemporal = CrossAttentionTemporal(convolution_set=convolution_set, d_model=d_model, num_heads=num_heads, dropout=dropout)
		self.CrossAttentionFunctional = CrossAttentionFunctional(sorted_map, d_model=d_model, num_heads=num_heads, dropout=dropout)
		
		self._ff_block_x = _ff_block(d_model, d_ffn)
	
	def forward(self, x, y):
		norm_y = self.norm[0](y)
		x = x + self.dropout[0](self.CrossAttentionTemporal(self.norm[1](x), norm_y))
		
		x = x + self.dropout[1](self.CrossAttentionFunctional(self.norm[2](x), norm_y))
		x = x + self.dropout[2](self._ff_block_x(self.norm[3](x)))
		
		return x
		
class CantusCerebraUno(nn.Module):
	def __init__(self, sorted_map, num_layers=6, in_dim=200, d_model=200, num_heads=8, dropout=0.1, convolution_set=[(1,), (3,), (5,)], d_ffn=800, out_dim=200):
		super().__init__()
		
		self.register_buffer('sorted_map', sorted_map)
		self.num_layers = num_layers
		self.EmbeddingStepTmp = EmbeddingStepTmp(in_dim=in_dim, d_model=d_model)
		
		self.EmbeddingStepSpc = EmbeddingStepSpc(in_dim=in_dim, d_model=d_model)
		self.TemporalEncoderLayers = nn.ModuleList([TransformerTemporalEncoderLayer(self.sorted_map, in_dim=in_dim, out_dim=out_dim, d_model=d_model, nheads=num_heads, dropout=dropout, batch_first=True, convolution_set=convolution_set, d_ffn=d_ffn) for _ in range(self.num_layers + 1)])
		
		self.SpcEncoderLayers = nn.ModuleList([SpcEncoderLayer(self.sorted_map, in_dim=in_dim, d_model=d_model, convolution_set=convolution_set, dropout=dropout, num_heads=num_heads, d_ffn=d_ffn) for _ in range(self.num_layers)])
		self.CrossAttn_Tmp_Spc = nn.ModuleList([CrossAttention(self.sorted_map, d_model=d_model, dropout=dropout, convolution_set=convolution_set, num_heads=num_heads, d_ffn=d_ffn) for _ in range(self.num_layers // 2)])
		self.CrossAttn_Spc_Tmp = nn.ModuleList([CrossAttention(self.sorted_map, d_model=d_model, dropout=dropout, convolution_set=convolution_set, num_heads=num_heads, d_ffn=d_ffn) for _ in range(self.num_layers // 2)])
		self.proj_out = nn.Linear(d_model, out_dim)
		
	def forward(self, x, psd_x, mask=None, is_causal=False, need_key_padding=False):
		embed_x = self.EmbeddingStepTmp(x, mask)
		embed_psd_x = self.EmbeddingStepSpc(psd_x)
		
		mix_layers = [1, 3, 5]
		
		for i in range(self.num_layers + 1):
			embed_x = self.TemporalEncoderLayers[i](embed_x, is_causal=is_causal, need_key_padding=need_key_padding)
			
			if i < self.num_layers:
				embed_psd_x = self.SpcEncoderLayers[i](embed_psd_x)
				
				if i in mix_layers:
				
					new_embed_x = self.CrossAttn_Tmp_Spc[i // 2](embed_x, embed_psd_x)
					new_embed_psd_x = self.CrossAttn_Spc_Tmp[i // 2](embed_psd_x, embed_x)

					embed_x = new_embed_x
					embed_psd_x = new_embed_psd_x
				
		out = self.proj_out(embed_x)
		return out		
