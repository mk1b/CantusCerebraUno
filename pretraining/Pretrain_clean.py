import argparse
from model.TmpEncoder_stage5_clean import Final
import torch

import numpy as np
import random
from torch.utils.data import DataLoader

import mne
from datasets.siena_dataset import LoadDataset as LoadDataset_siena
from datasets.tuab_dataset import LoadDataset as LoadDataset_tuab
import ast
	
import os
from tqdm import tqdm
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
'''
import torch.distributed as dist	# We tell the GPUs to talk to each other.
from torch.utils.data.distributed import DistributedSampler	# We will make the change later in our training loop.

def GPU_init():
	dist.init_process_group(backend='nccl')
	local_rank = int(os.environ['LOCAL_RANK'])
	torch.cuda.set_device(local_rank)
'''

def generate_mask(Bz, ch_num, patch_num, mask_ratio, device):
	mask = torch.zeros((Bz, ch_num, patch_num), dtype=torch.float32, device=device)
	
	mask = mask.bernoulli_(mask_ratio)
	return mask

# AI generated plotting util function. Things which are AI generated will be written as AI generated above the line of code.
def plot_eeg_reconstruction(x, y, epoch, mask, batch_idx=0, channel_idx=0):
	"""
	x: Original tensor (Bz, Chans, Patches, Patch_size)
	y: Model output (Bz, Chans, Patches, Patch_size)
	mask: Mask tensor (Bz, Chans, Patches) - 1 for masked, 0 for visible
	"""
	# 1. Extract a single sequence and move to CPU
	# We flatten (Patches, Patch_size) into a single (Time_Steps) array
	true_signal = x[batch_idx, channel_idx].detach().cpu().numpy().flatten()
	pred_signal = y[batch_idx, channel_idx].detach().cpu().numpy().flatten()
	
	# 2. Create the "Input" signal (what the model actually saw)
	# We expand the mask to match the signal length
	m = mask[batch_idx, channel_idx].detach().cpu().numpy()
	m_expanded = np.repeat(m, x.shape[-1]) # Repeat mask for each point in patch
	
	input_signal = true_signal.copy()
	input_signal[m_expanded == 1] = 0 # Zero out the parts the model didn't see

	# 3. Plotting
	plt.figure(figsize=(15, 6))
	
	# Ground Truth
	plt.plot(true_signal, label='Original (Ground Truth)', color='gray', alpha=0.5, linestyle='--')
	
	# What the model saw (Visible parts)
	plt.plot(np.where(m_expanded == 0, true_signal, np.nan), label='Visible Context', color='blue', linewidth=2)
	
	# The Model's Prediction in the masked areas
	plt.plot(np.where(m_expanded == 1, pred_signal, np.nan), label='Model Reconstruction', color='red', linewidth=2)

	plt.title(f"EEG Reconstruction - Batch {batch_idx}, Channel {channel_idx}")
	plt.xlabel("Time Samples")
	plt.ylabel("Normalized Amplitude")
	plt.legend()
	plt.grid(True, alpha=0.3)
	path = os.path.expanduser(f'~/CantusCerebra/reconstruction_plots/reconstruction_epoch_{epoch}.png')
	plt.savefig(path)
	plt.close('all')
	
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


class Trainer():
	def __init__(self, params, data_loader, model):
		self.params = params
		
		self.device = torch.device(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
		self.data_loader = data_loader
		
		self.data_length = len(self.data_loader)
		self.model = model
		
		self.model = self.model.to(self.device)		
		
		if self.params.parallel:	
			device_ids = [int(i) for i in self.params.avail_gpus.split(' ')]
			self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)	
		
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
		
		self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
				self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-5	
			)
			
		self.criterion = nn.MSELoss(reduction='mean').to(self.device)	
		self.scaler = torch.amp.GradScaler('cuda')
		
	def train(self):
		best_loss = float('inf')
		
		for epoch in range(self.params.epochs):
			print(f'Epoch {epoch} start')
			
			losses = []
			self.model.train()
			
			tot = len(self.data_loader)
			
			for i, (x, _) in enumerate(tqdm(self.data_loader, mininterval=10)):	
				self.optimizer.zero_grad()	
				
				x = x.to(self.device)
				
				with torch.amp.autocast('cuda'):
					if self.params.need_mask:
						Bz, num_chans, num_patch, patch_size = x.shape
						
						mask = generate_mask(Bz, num_chans, num_patch, self.params.mask_ratio, self.device)
						y = self.model(x, mask=mask, is_causal=self.params.is_causal, need_key_padding=self.params.need_key_padding)
						
						mask_x = x[mask == 1]
						mask_y = y[mask == 1]	
						
						loss = self.criterion(mask_y, mask_x)
						
						if i == len(self.data_loader) - 1:
							plot_eeg_reconstruction(x, y, epoch, mask)	
								
					else:
						y = self.model(x, is_causal=self.params.is_causal, need_key_padding=self.params.need_key_padding)
						loss = self.criterion(y, x)
				
				self.scaler.scale(loss).backward()
				
				if self.params.clip_value > 0:
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
					
				self.scaler.step(self.optimizer)
				self.scaler.update()
				self.optimizer_scheduler.step()
				
				losses.append(loss.item())
				
			mean_loss = np.mean(losses)
			
			if mean_loss < best_loss:
				best_model_path = os.path.join(self.params.model_dir, 'pretrain_weights.pth')
				
				torch.save(self.model.state_dict(), best_model_path)	
				print(f"Model saved at {best_model_path}")
				
				best_loss = mean_loss
				
				optimizer_path = os.path.join(self.params.model_dir, 'pretrain_optimizer_weights.pth')
				
				torch.save(self.optimizer.state_dict(), optimizer_path)
				print(f'Optimizer saved @ {optimizer_path}')
			
			with open(os.path.expanduser('~/CantusCerebra/logs/pretrain_loss.txt'), 'a') as log:
				log.write(f'{mean_loss}\n')
			print(f'Epoch: {epoch}, mean_loss : {mean_loss}')

def setup_seed(seed):
	torch.manual_seed(seed) 	
	torch.cuda.manual_seed_all(seed)	
	np.random.seed(seed)	
	random.seed(seed)
	torch.backends.cudnn.deterministic = True	
	torch.backends.cudnn.benchmark = False

def main():
	# 	GPU_init()
	parser = argparse.ArgumentParser(description='EEG Foundation Model')
	
	parser.add_argument('--dropout', type=float, default=0.1, help='dropout value')
	parser.add_argument('--in_dim', type=int, default=200, help='Number of samples in 1s raw')		
	parser.add_argument('--out_dim', type=int, default=200, help='Output dimension')
	parser.add_argument('--d_model', type=int, default=200, help='Model operating dimension')
	parser.add_argument('--d_ffn', type=int, default=800, help='Standard 2-layer FFN dimensions')
	parser.add_argument('--num_layers', type=int, default=6, help='Number of Transformer layers')
	parser.add_argument('--nheads', type=int, default=8, help='Number of Heads in MHSA')
	parser.add_argument('--convolution_set', type=str, default="[(1,), (3,), (5,)]", help='Concentrated convolution sizes. < num_chans')
	parser.add_argument('--seq_len', type=int, default=30, help='num_patches')
	parser.add_argument('--seed', type=int, default=42, help='seed for RNG')
	parser.add_argument('--is_causal', action='store_true', help='If you want causal Temporal Attention')
	parser.add_argument('--need_key_padding', action='store_true', help='if any padding that could be added is to be ignored')
	parser.add_argument('--stride', type=int, default=1, help='stride for temp convs')
	parser.add_argument('--batch_size', type=int, default=64, help='bz')
	
	parser.add_argument('--lr', type=float, default=5e-4, help='lr')
	parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')
	parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
	parser.add_argument('--parallel', action='store_false', help='use gpu?')
	parser.add_argument('--epochs', type=int, default=40, help='# of passes over data')
	parser.add_argument('--need_mask', action='store_false', help='u need mask or no')
	parser.add_argument('--clip_value', type=float, default=1, help='grad over clip=cut')
	parser.add_argument('--model_dir', type=str, default='~/CantusCerebra/saved_fm', help='address of model')
	parser.add_argument('--dataset_dir', type=str, default='~/CantusCerebra/processed_data/processed_siena/json_generate', help='root of dataset')
	parser.add_argument('--cuda', type=int, default=0, help='what is the primary gpu that is available?')
	parser.add_argument('--avail_gpus', type=str, default='0', help='Atleast 1 GPU is required to train this model. zero indexed.')
	
	print('Starting training...')
	params = parser.parse_args()
	setup_seed(params.seed)
	
	params.model_dir = os.path.expanduser(params.model_dir)
	params.dataset_dir = os.path.expanduser(params.dataset_dir)
	
	print('Getting dataloader...')
	tuab_dataset = LoadDataset_tuab(params)
	tuab_data_loader = tuab_dataset.get_data_loader()
	
	sorted_map = sorted_maps().to(f'cuda:{params.cuda}' if torch.cuda.is_available() else 'cpu')
	print('Loading model...')
	
	model = Final(sorted_map=sorted_map, d_model=params.d_model, convolution_set=ast.literal_eval(params.convolution_set), stride=params.stride, in_dim=params.in_dim, out_dim=params.out_dim, dropout=params.dropout, 
		batch_first=True, nheads=params.nheads, num_layers=params.num_layers)
		
	print('Goes to training !')
	
	trainer = Trainer(params, siena_data_loader['train'], model)
	trainer.train()
	
if __name__ == '__main__':
	main()
