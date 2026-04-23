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
    'Fp1', 'F7', 'T7', 'P7', 'O1', 
    'Fp2', 'F8', 'T8', 'P8', 'O2', 
    'F3', 'C3', 'P3', 'F4', 'C4', 'P4'
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
		sorted_map = torch.rand((116, 116)).float()
		self.register_buffer('sorted_maps', sorted_map)
		self.backbone = CantusCerebraUno(sorted_map, num_layers=params.num_layers, in_dim=params.in_dim, d_model=params.d_model, num_heads=params.num_heads, dropout=params.dropout, convolution_set=params.convolution_set, d_ffn=params.d_ffn, out_dim=params.out_dim)	
		self.params = params
		self.device = torch.device(f'cuda:{self.params.cuda}') if torch.cuda.is_available() else 'cpu'
		
		if self.params.use_pretrained_weights:
			map_location = self.device	
			#self.params.state_dict_path = os.path.join(self.params.state_dict_path, f'finetune_weights_{self.params.epch}.pth')
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
						nn.Linear(100, 1),
					)
						
	def forward(self, x, psd_x):
		Bz, num_chans, num_patches, patch_size = x.shape
		
		emb = self.backbone(x, psd_x)
		emb = emb.mean(dim=(1, 2))
		
		out = self.FFN(emb)	
		out = out.reshape(Bz)
		
		return out
