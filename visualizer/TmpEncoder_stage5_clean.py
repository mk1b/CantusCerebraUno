import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mne
import os
import random
from tqdm import tqdm
from torch.nn import MSELoss

hcp_positions_path = os.path.expanduser('~/CantusCerebra/data/HCP/positions_100_7.txt')
connectivity_path = os.path.expanduser('~//CantusCerebra/processed_data/connectivity_matrix.txt')
	
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
		self.register_buffer("mask_encoding", torch.zeros(in_dim))	
		self.spectral_proj = nn.Sequential(
			nn.Linear(in_dim // 2 + 1, in_dim),
			nn.Dropout(0.1),
		)
		self.in_dim = in_dim
		
	def forward(self, x, mask=None):
		if mask is None:
			mask_x = x
		else:
			mask_x = x.clone()
			mask_x[mask == 1] = self.mask_encoding	
		Bz, num_chans, num_patch, patch_size = mask_x.shape
		mask_x = mask_x.reshape(Bz, 1, num_patch * num_chans, patch_size)
		
		patch_emb = self.extract_features(mask_x)
		
		patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(Bz, num_chans, num_patch, self.in_dim)

		mask_x = mask_x.contiguous().view(Bz * num_chans * num_patch, patch_size)
		spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
		spectral = torch.abs(spectral).contiguous().view(Bz, num_chans, num_patch, patch_size // 2 + 1)
		spectral_emb = self.spectral_proj(spectral)
		patch_emb = patch_emb + spectral_emb	
		
		return patch_emb
	
class ACPE(nn.Module):
	def __init__(self, d_model=200):
		super().__init__()
		self.mix_featurewise = nn.Sequential(
									nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3), groups=d_model)
								)
	def forward(self, x):
		embedding = self.mix_featurewise(x.permute(0, 3, 1, 2))
		embedding = embedding.permute(0, 2, 3, 1)
		
		return embedding

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

class TemEmbedEEGLayer(nn.Module):
	def __init__(self, d_model=200, convolution_set=[(1,), (3,), (5,)], stride=1):
		super().__init__()
		
		dim_scales = [d_model // 2**i for i in range(1, len(convolution_set))]
		dim_scales = [*dim_scales, d_model - sum(dim_scales)]
		
		self.mix_features = nn.ModuleList([
								nn.Conv2d(in_channels=d_model, out_channels=dim_scale, kernel_size=(size, 1), padding=((size-1)//2, 0), stride=(stride, 1))
								for (size,), dim_scale in zip(convolution_set, dim_scales)
							])
		
	def forward(self, x):
		
		Bz, num_chans, num_patches, patch_size = x.shape
		
		x = x.reshape(Bz * num_chans, 1, num_patches, patch_size)
		x = x.permute(0, 3, 2, 1)
		
		fmaps = [conv(x) for conv in self.mix_features]
		
		out = torch.cat(fmaps, dim=1)
		
		out = out.permute(0, 3, 2, 1)
		out = out.reshape(Bz, num_chans, num_patches, patch_size)
		
		return out
		


class BrainEmbedEEGLayer(nn.Module):
	def __init__(self, sorted_map, d_model, convolution_set=[(1,), (3,), (5,)]):
		super().__init__()
		
		self.convolution_set = convolution_set
		self.register_buffer("sorted_map", sorted_map.long())
		
		dim_scales = [d_model // 2**i for i in range(1, len(convolution_set))]
		dim_scales = [*dim_scales, d_model - sum(dim_scales)]
		
		mix_features = [
							nn.Conv2d(in_channels=d_model, out_channels=dim_scale, kernel_size=(size, 1), stride=(1, 1), padding=((size - 1)//2, 0), padding_mode='circular')
							for (size,), dim_scale in zip(convolution_set, dim_scales) if size > 1
						]
						
		mix_features = [nn.Conv2d(in_channels=d_model, out_channels=dim_scales[0], kernel_size=(1, 1), stride=(1, 1)), *mix_features]
		self.mix_features = nn.ModuleList(mix_features)
		
	def forward(self, x):
		Bz, num_chans, num_patch, patch_size = x.shape
		largest_group_size = self.convolution_set[-1][0]
		x_chan = x[:, self.sorted_map[:, :largest_group_size], :, :]
		
		x_chan = x_chan.permute(0, 1, 3, 4, 2)	
		x_chan = x_chan.contiguous().reshape(Bz * num_chans * num_patch, patch_size, largest_group_size, 1)
		
		# x_chan = x_chan.permute(0, 2, 3, 1)		
		fmaps = [conv(x_chan) for conv in self.mix_features]
		x_fused = torch.cat(fmaps, dim=1)	
		
		x_fused = x_fused.squeeze(-1)
		x_fused = x_fused.reshape(Bz, num_chans, num_patch, patch_size, largest_group_size)
		
		out = x_fused[:, :, :, :, 0]
		
		return out
		# Flawless!

class TemporalAttention(nn.Module):
	def __init__(self, d_model=200, nheads=8, dropout=0.1, batch_first=True, convolution_set=[(1,), (3,), (5,)]):
		super().__init__()
		
		self.convolution_set = convolution_set
		self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, dropout=dropout, batch_first=batch_first)	
	
	def forward(self, x, is_causal=False, need_key_padding=False):
		Bz, num_chans, num_patches, patch_size = x.shape
		
		window_size = min(num_patches, self.convolution_set[-1][0])
		num_windows = num_patches // window_size
		original_num_patches = num_patches
		
		if num_patches % window_size != 0:
			padding = window_size - (num_patches % window_size)
			
			x = F.pad(x, (0, 0, 0, padding))	
			num_patches = num_patches + padding
			num_windows = num_patches // window_size
			
		x = x.reshape(Bz, num_chans, num_windows, window_size, patch_size)
		x = x.permute(0, 1, 3, 2, 4)
		x = x.reshape(Bz * num_chans * window_size, num_windows, patch_size)

		temporal_attn_mask = None
		if is_causal:
			temporal_attn_mask = torch.triu(torch.ones(num_windows, num_windows, device=x.device) * float('-inf'), diagonal=1)
			temporal_attn_mask = temporal_attn_mask.to(x.dtype)	
				
		key_padding_mask = None
		if need_key_padding:
			key_padding_mask = torch.full((num_patches,), False, dtype=torch.bool, device=x.device)
				
			key_padding_mask[original_num_patches:] = True
			key_padding_mask = key_padding_mask.view(1, 1, num_patches).expand(Bz, num_chans, num_patches)
				
			key_padding_mask = key_padding_mask.reshape(Bz, num_chans, num_windows, window_size)	
				
			key_padding_mask = key_padding_mask.permute(0, 1, 3, 2)
			key_padding_mask = key_padding_mask.reshape(Bz * num_chans * window_size, num_windows)
		
		out, wts = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=temporal_attn_mask, need_weights=True)
		
		mp = wts[0]
		
		plt.figure(figsize=(6, 5))
		sns.heatmap(mp, cmap="viridis")

		plt.xlabel("Key")
		plt.ylabel("Query")
		plt.title("Attention")

		plt.tight_layout()
		plt.savefig("tmp_attention.png", dpi=300)
		plt.close()
		
		out = out.reshape(Bz, num_chans, window_size, num_windows, patch_size)
		out = out.permute(0, 1, 3, 2, 4)	
		out = out.reshape(Bz, num_chans, num_patches, patch_size)
		
		if num_patches != original_num_patches:
			final_out = out[:, :, :original_num_patches, :]
		else:
			final_out = out
		
		return final_out

class RegionAttention(nn.Module):
	def __init__(self, sorted_map, d_model=200, nheads=8, dropout=0.1, batch_first=True):
		super().__init__()
		self.register_buffer("sorted_map", sorted_map.long())
		self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, dropout=dropout, batch_first=batch_first)
		self.group_transform = nn.Linear(d_model, d_model)
		
	def forward(self, x):
		Bz, num_chans, num_patch, patch_size = x.shape

		x = x.permute(0, 2, 1, 3).contiguous()
		x = x.reshape(Bz * num_patch, num_chans, patch_size)
		
		num_complete_groups = num_chans // 4
		num_electrodes_complete_groups = num_complete_groups * 4
		
		sorted_x = x[:, self.sorted_map, :]
			
		main_x = sorted_x[:, :, :num_electrodes_complete_groups, :]
		rem_x = sorted_x[:, :, num_electrodes_complete_groups:, :]
		
		index = torch.arange(0, num_electrodes_complete_groups, 4, device=x.device)
			
		first_tensors = main_x[:, :, index, :]
		grouped_x = main_x.reshape(Bz * num_patch, num_chans, num_complete_groups, 4, patch_size)
		
		groups = grouped_x.mean(dim=3)
		rep = first_tensors + self.group_transform(groups)
		
		if num_chans != num_electrodes_complete_groups:
			rem_rep = (rem_x[:, :, 0, :] * 2).unsqueeze(2)	
			final_stacked = torch.cat([rep, rem_rep], dim=2)
		else:
			final_stacked = rep

		num_groups = final_stacked.shape[2]
		
		final_flat = final_stacked.reshape(Bz * num_patch * num_chans, num_groups, patch_size)
		attn_out, wts = self.attn(final_flat, final_flat, final_flat, need_weights=True)
		
		mp = wts[0]  # (g, g)

		plt.figure(figsize=(6, 5))
		sns.heatmap(mp, cmap="viridis")

		plt.xlabel("Key")
		plt.ylabel("Query")
		plt.title("Attention")

		plt.tight_layout()
		plt.savefig("rg_attention.png", dpi=300)
		plt.close()
		
		attn_out = attn_out.reshape(Bz, num_patch, num_chans, num_groups, patch_size)
		
		
		updated = attn_out[:, :, :, 0, :]
		updated = updated.permute(0, 2, 1, 3)
		
		return updated

def sorted_maps():
	ch_names = [
	'Fp1', 'F7', 'T7', 'P7', 'O1', 
	'Fp2', 'F8', 'T8', 'P8', 'O2', 
	'F3', 'C3', 'P3', 'F4', 'C4', 'P4'
]
	
	montage = mne.channels.make_standard_montage('standard_1005')
	all_pos = montage.get_positions()['ch_pos']
  
	# mapped_pos = {}
	pos_array = np.array([all_pos[ch] for ch in ch_names]) * 1000
	
	brain_regions = np.loadtxt(hcp_positions_path)
	used_pos = []
	
	for ch in ch_names:
		pos = all_pos[ch] * 1000
		idx = np.argmin(np.sum((brain_regions - pos)**2, axis=1))
		
		used_pos.append(idx)
	 
	correlations = np.loadtxt(connectivity_path)
	
	sub_corr = correlations[np.ix_(used_pos, used_pos)]
	sorted_map = np.argsort(-sub_corr, axis=1)
	
	return torch.tensor(sorted_map)

class TransformerTemporalEncoderLayer(nn.Module):
	def __init__(self, sorted_map, in_dim=200, out_dim=200, d_model=200, nheads=8, dropout=0.1, batch_first=True, convolution_set=[(1,), (3,), (5,)], d_ffn=800):
		super().__init__()
		
		self.register_buffer('sorted_map', sorted_map.long())
		self.TemporalAttention = TemporalAttention(d_model, nheads, dropout, batch_first, convolution_set)
		self.RegionAttention = RegionAttention(sorted_map, d_model=d_model, nheads=nheads, dropout=dropout, batch_first=batch_first)
		self._ff_block = _ff_block(d_model=d_model, d_ffn=d_ffn)
		
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)
		
	def forward(self, x, is_causal=False, need_key_padding=False):
		x = x + self.dropout1(self.TemporalAttention(self.norm1(x), is_causal=is_causal, need_key_padding=need_key_padding))	
		
		x = x + self.dropout2(self.RegionAttention(self.norm2(x)))
		
		x = x + self.dropout3(self._ff_block(self.norm3(x)))
		
		return x

class Final(nn.Module):
	def __init__(self, sorted_map, d_model=200, convolution_set=[(1,), (3,), (5,)], stride=1, in_dim=200, out_dim=200, dropout=0.1, batch_first=True, d_ffn=800, nheads=8, num_layers=6):
		super().__init__()
		self.register_buffer('sorted_map', sorted_map)
		
		self.num_layers = num_layers
		#self.norm_final = nn.LayerNorm(d_model)
		
		self.SimpleEmbedding = SimpleEmbedding(in_dim=in_dim)
		self.ACPE = ACPE(d_model=d_model)
		
		self.TemEmbedEEGLayers = nn.ModuleList([TemEmbedEEGLayer(d_model=d_model, convolution_set=convolution_set, stride=stride) for _ in range(self.num_layers)])
		
		self.BrainEmbedEEGLayers = nn.ModuleList([BrainEmbedEEGLayer(sorted_map=sorted_map, d_model=d_model, convolution_set=convolution_set) for _ in range(self.num_layers)])
		
		self.TransformerTemporalEncoderLayers = nn.ModuleList([TransformerTemporalEncoderLayer(sorted_map, in_dim=in_dim, out_dim=out_dim, d_model=d_model,
													nheads=nheads, dropout=dropout, batch_first=batch_first, convolution_set=convolution_set, d_ffn=d_ffn) for _ in range(num_layers)])
		self.proj_out = nn.Linear(d_model, out_dim)
		
	def forward(self, x, mask=None, is_causal=False, need_key_padding=False):
		patch_emb = self.SimpleEmbedding(x, mask=mask)
		patch_emb = self.ACPE(patch_emb)
		
		for i in range(self.num_layers):
			patch_emb = patch_emb + self.TemEmbedEEGLayers[i](patch_emb)
			
			patch_emb = patch_emb + self.BrainEmbedEEGLayers[i](patch_emb)
			patch_emb = self.TransformerTemporalEncoderLayers[i](patch_emb, is_causal=is_causal, need_key_padding=need_key_padding)
			
		#out = self.norm_final(patch_emb)
		final = self.proj_out(patch_emb)
		return final
