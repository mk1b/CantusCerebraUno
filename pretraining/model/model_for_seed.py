import torch
import torch.distributed as dist
import os
import torch.nn
import numpy as np
import mne

from model.CantusCerebraUno import *
from torch.nn.parallel import DistributedDataParallel as DDP

hcp_positions_path = os.path.expanduser('~/CantusCerebra/data/HCP/positions_100_7.txt')
connectivity_path = os.path.expanduser('~/CantusCerebra/processed_data/connectivity_matrix.txt')

def sorted_maps():
	ch_names = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "PO9", "O1", "Oz", "O2", "PO10"
]
	
	montage = mne.channels.make_standard_montage('standard_1005')
	all_pos = montage.get_positions()['ch_pos']
  
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

class Model(nn.Module):
	def __init__(self, params):
		super().__init__()
		
		self.backbone = CantusCerebraUno(sorted_maps(), d_model=params.d_model, convolution_set=params.convolution_set, stride=params.stride, in_dim=params.in_dim, out_dim=params.out_dim, dropout=params.dropout, d_ffn=params.d_ffn,
	num_heads=params.num_heads, num_layers=params.num_layers, mother_wavelet=params.mother_wavelet, bands=params.bands, num_decoder_layers=params.num_decoder_layers, kernel_size=params.kernel_size, num_chans=params.num_chans)	
		self.params = params
		
		if self.params.use_pretrained_weights:
			map_location = self.device	
			
			state_dict = torch.load(self.params.state_dict_path, map_location=map_location)
			new_state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
				
			model_state_dict = self.backbone.state_dict()
			matching_state_dict = {k:v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}	
			
			model_state_dict.update(matching_state_dict)		
			self.backbone.load_state_dict(model_state_dict)	
			
		self.FFN = nn.Sequential(
						nn.Linear(200, 100),
						nn.ELU(),
						nn.Dropout(params.dropout),
						nn.Linear(100, self.params.num_of_classes),
					)
						
	def forward(self, x):
		Bz, num_chans, num_patches, patch_size = x.shape
		
		emb = self.backbone(x)
		emb = emb.mean(dim=(1, 2))
		
		out = self.FFN(emb)	
		out = out.reshape(Bz)
		
		return out
