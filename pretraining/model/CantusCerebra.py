import ptwt
import pywt
import torch
import torch.nn as nn
from model.TmpEncoder_stage5_clean import Final as FinalTmp
from model.SpcEncoder_stage5_clean import Final as FinalSpc

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
	
class DecoderLayer(nn.Module):
	def __init__(self, dropout=0.1, d_model=200, num_heads=8, d_ffn=800):
		super().__init__()
		
		self.MergeAttention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
		
		self.norm1 = nn.LayerNorm(d_model)
		
		self.FFN = _ff_block(in_dim=d_model, d_ffn=d_ffn, out_dim=d_model)
	
	def forward(self, tmp, spc):
	
		Bz, num_chans, num_patch, patch_size = tmp.shape
		
		tmp = tmp.reshape(Bz * num_chans, num_patch, patch_size)
		spc = spc.reshape(Bz * num_chans, num_patch, patch_size)

		attn = self.MergeAttention(tmp, spc, spc)[0]
		
		tmp_upd = tmp + attn
		
		out = tmp_upd + self.FFN(self.norm1(tmp_upd))
		
		out = out.reshape(Bz, num_chans, num_patch, patch_size)
		return out

class Decoder(nn.Module):
	def __init__(self, d_model=200, num_decoder_layers=2, out_dim=200, dropout=0.1, num_heads=8, d_ffn=800):
	
		super().__init__()
		
		self.num_decoder_layers = num_decoder_layers
		
		self.DecoderLayers = nn.ModuleList([DecoderLayer(dropout=dropout, d_model=d_model, num_heads=num_heads, d_ffn=d_ffn) for _ in range(num_decoder_layers)])
		
		self.proj_out = nn.Linear(d_model, out_dim)
	
	def forward(self, tmp, spc):
	
		Bz, num_chans, num_patch, patch_size = tmp.shape
		
		for i in range(self.num_decoder_layers):
		
			tmp = self.DecoderLayers[i](tmp, spc)
			
		out = self.proj_out(tmp)
		
		return out
		
class CantusCerebra(nn.Module):
	def __init__(self, sorted_maps, d_model=200, convolution_set=[(1,), (3,), (5,)], stride=1, in_dim=200, out_dim=200, dropout=0.1, d_ffn=800, num_heads=8, num_layers=6, mother_wavelet='cmor1.5-1.0', bands=[(0.5, 4.25), (4, 8.25), (8, 13.25), (13, 30.25), (30, 75)], num_decoder_layers=2, kernel_size=3, num_chans=29):
	
		super().__init__()
		
		self.register_buffer('sorted_maps', sorted_maps)
		
		self.FinalTmp = FinalTmp(sorted_maps, d_model=d_model, convolution_set=convolution_set, stride=stride, in_dim=in_dim, out_dim=out_dim, dropout=dropout, d_ffn=d_ffn, nheads=num_heads, num_layers=num_layers)
		
		self.FinalSpc = FinalSpc(num_layers=num_layers, d_model=d_model, kernel_size=kernel_size, in_dim=in_dim, out_dim=out_dim, convolution_set=convolution_set, stride=stride, d_ffn=d_ffn, num_heads=num_heads, dropout=dropout, num_chans=num_chans)
		
		self.mother_wavelet = mother_wavelet
		
		self.bands = bands
		
		self.decoder = Decoder(d_model=d_model, num_decoder_layers=num_decoder_layers, out_dim=out_dim, dropout=dropout, num_heads=num_heads, d_ffn=d_ffn)
		#self.decoder = nn.Linear(d_model * 2, d_model)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		
		self.proj_out = nn.Linear(d_model, d_model)
		
	def forward(self, x, amp, phase_cos, phase_sin, is_causal=False, need_key_padding=False, need_cross_attn_mask=False, mask=None, spc_mask=None):
	
		Bz, num_chans, num_patches, patch_size = x.shape
		
		Tmp_x = self.FinalTmp(x, is_causal=is_causal, need_key_padding=need_key_padding, mask=mask)
		
		Spc_x = self.FinalSpc(amp, phase_cos, phase_sin, spc_mask, is_causal=is_causal, need_key_padding=need_key_padding, need_cross_attn_mask=need_cross_attn_mask)
		
		norm_Tmp_x = self.norm1(Tmp_x)
		norm_Spc_x = self.norm2(Spc_x)
		
		out = self.decoder(norm_Tmp_x, norm_Spc_x)
		
		return out
