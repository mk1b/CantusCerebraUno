import torch
from tqdm import tqdm
import argparse
import ast
import os
import random
import mne
import torch.nn as nn
import numpy as np
import pywt
import ptwt
from model.CantusCerebraUno import CantusCerebraUno
from datasets.pretraining_dataset import PretrainingDataset as LoadDataset
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch.backends.cudnn as cudnn
import torch.backends.cuda as cuda
cudnn.benchmark = True
from torch.utils.data import DataLoader

hcp_positions_path = os.path.expanduser('~/CantusCerebra/data/HCP/positions_100_7.txt')
connectivity_path = os.path.expanduser('~/CantusCerebra/processed_data/connectivity_matrix.txt')

def sorted_maps():
	ch_names = [
	'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
	'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
	'Fz', 'Cz', 'Pz'
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

def plot_eeg_reconstruction(x, y, psd_x, psd_y, epoch, step, mask, bands, batch_idx=0, channel_idx=0):
	# Setup a 3x2 grid for 6 plots
	fig, axes = plt.subplots(3, 2, figsize=(20, 15))
	axes = axes.flatten()

	# --- 1. Temporal Signal Plot (Top Left) ---
	true_signal = x[batch_idx, channel_idx].detach().cpu().numpy().flatten()
	pred_signal = y[batch_idx, channel_idx].detach().cpu().numpy().flatten()
	
	m = mask[batch_idx, channel_idx].detach().cpu().numpy()
	m_expanded = np.repeat(m, x.shape[-1]) 
	
	axes[0].plot(true_signal, label='Original (Ground Truth)', color='gray', alpha=0.5, linestyle='--')
	axes[0].plot(np.where(m_expanded == 0, true_signal, np.nan), label='Visible Context', color='blue', linewidth=2)
	axes[0].plot(np.where(m_expanded == 1, pred_signal, np.nan), label='Model Reconstruction', color='red', linewidth=2)
	
	axes[0].set_title(f"Temporal EEG - Batch {batch_idx}, Channel {channel_idx}")
	axes[0].set_xlabel("Time Samples")
	axes[0].set_ylabel("Amplitude")
	axes[0].legend(loc="upper right")
	axes[0].grid(True, alpha=0.3)

	# --- 2. PSD Bands Plots (Remaining 5 Subplots) ---
	# Extract the PSDs for this specific batch and channel
	# Shape becomes: (num_bands, num_patch, patch_size)
	true_psd = psd_x[batch_idx, channel_idx].detach().cpu().numpy() 
	pred_psd = psd_y[batch_idx, channel_idx].detach().cpu().numpy() 
	
	# Flatten spatial dimensions so we just have (num_bands, Time)
	true_psd_flat = true_psd.reshape(true_psd.shape[0], -1) 
	pred_psd_flat = pred_psd.reshape(pred_psd.shape[0], -1)

	# Loop through each band and assign it to a subplot
	for i in range(len(bands)):
		ax = axes[i + 1]
		
		t_psd = true_psd_flat[i]
		p_psd = pred_psd_flat[i]
		
		ax.plot(t_psd, label='Original PSD', color='gray', alpha=0.5, linestyle='--')
		ax.plot(np.where(m_expanded == 0, t_psd, np.nan), label='Visible Context', color='blue', linewidth=2)
		ax.plot(np.where(m_expanded == 1, p_psd, np.nan), label='Reconstructed PSD', color='red', linewidth=2)
		
		ax.set_title(f"Band {i+1}: {bands[i][0]}-{bands[i][1]} Hz")
		ax.set_xlabel("Time Samples")
		ax.set_ylabel("Normalized Power")
		ax.legend(loc="upper right")
		ax.grid(True, alpha=0.3)

	plt.tight_layout()
	
	# Ensure directory exists just in case
	save_dir = os.path.expanduser('~/CantusCerebra/reconstruction_plots')
	os.makedirs(save_dir, exist_ok=True)
	
	path = os.path.join(save_dir, f'reconstruction_epoch_{epoch}_{step}.png')
	plt.savefig(path)
	plt.close('all')
	
def smooth_psd_tensor(psd_tensor, window_size=10):
	"""
	Applies a moving average smoothing filter to the PSD tensor across the time dimension.
	Fully differentiable so it can be used inside the training loop.
	"""
	Bz, num_chans, num_bands, num_patch, patch_size = psd_tensor.shape
	
	# Flatten spatial/band dimensions to treat the time sequence as a 1D signal
	# Shape becomes: (Batch * Channels * Bands, 1, Time)
	x = psd_tensor.reshape(Bz * num_chans * num_bands, 1, num_patch * patch_size)
	
	# Create a moving average kernel (e.g., [0.1, 0.1, ..., 0.1] for window_size=10)
	weight = torch.ones(1, 1, window_size, device=psd_tensor.device, dtype=psd_tensor.dtype) / window_size
	
	# Pad the time series to keep the sequence length exactly the same
	pad_left = window_size // 2
	pad_right = window_size - 1 - pad_left
	x_padded = F.pad(x, (pad_left, pad_right), mode='replicate')
	
	# Apply the convolution
	x_smoothed = F.conv1d(x_padded, weight)
	
	# Reshape back to original 5D shape
	return x_smoothed.view(Bz, num_chans, num_bands, num_patch, patch_size).contiguous()


def STFTTransform(x, sampling_frequency, bands=[(1, 5), (4, 9), (8, 14), (13, 31), (30, 76)]):
	Bz, num_chans, num_patch, patch_size = x.shape
	Total_Length = num_patch * patch_size
	
	x_flat = x.reshape(Bz * num_chans, Total_Length)
	
	n_fft = 256 
	# FIX: Increase hop_length. 16 or 32 is standard and shrinks memory massively.
	hop_length = 16 
	window = torch.hann_window(n_fft, device=x.device)

	# Output: (Batch*Chans, Freq_Bins, Reduced_Time_Steps)
	stft_out = torch.stft(
		x_flat, 
		n_fft=n_fft, 
		hop_length=hop_length, 
		window=window, 
		center=True,	  
		pad_mode='reflect',
		return_complex=True
	)
	
	psd_full = stft_out.abs() ** 2

	freq_resolution = sampling_frequency / n_fft
	
	band_means = []
	for (low, high) in bands:
		idx_start = int(low / freq_resolution)
		idx_end = int(high / freq_resolution)
		
		b_mean = torch.mean(psd_full[:, idx_start : max(idx_start + 1, idx_end), :], dim=1)
		band_means.append(b_mean)

	# Shape: (num_bands, Bz*Chans, Reduced_Time_Steps)
	band_cat = torch.stack(band_means, dim=0) 
	
	# Transpose for interpolation: (Batch, Channels, Length) -> (Bz*Chans, num_bands, Reduced_Time)
	band_cat = band_cat.permute(1, 0, 2)
	
	# FIX: Stretch the time dimension back to your exact original length
	# This is incredibly cheap and standard practice in signal processing
	band_cat = F.interpolate(band_cat, size=Total_Length, mode='linear', align_corners=False)
	
	# Unroll dimensions directly into your target shape!
	# Because of how we permuted above, the memory layout now perfectly matches:
	# (Bz, num_chans, num_bands, num_patch, patch_size)
	psd = band_cat.view(Bz, num_chans, len(bands), num_patch, patch_size)
	
	return psd

	
def generate_mask(Bz, ch_num, patch_num, mask_ratio, device):
	mask = torch.zeros((Bz, ch_num, patch_num), dtype=torch.float32, device=device)
	
	mask = mask.bernoulli_(mask_ratio)
	return mask

class Train:
	def __init__(self, params, model, data_loader):
		
		self.params = params
		self.device = torch.device(f'cuda:{self.params.cuda}') if torch.cuda.is_available() else 'cpu'
		
		self.band_dict = {i : torch.arange((self.params.bands[i][0]), (self.params.bands[i][1]), 1, device=self.device).long() for i in range(len(self.params.bands))}
		self.frequencies = torch.arange(1, 101, 1, device=self.device, requires_grad=False)
		
		self.scales = (pywt.central_frequency(self.params.mother_wavelet) * self.params.d_model / self.frequencies).cpu()	
			
		self.data_loader = data_loader
		self.model = model.to(self.device)
		self.model = torch.compile(self.model, options={"triton.cudagraphs": False})
		
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay, fused=True)
			
		self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
		
		self.accumulation_steps = 4
		
		self.length = len(data_loader) // self.accumulation_steps
		
		warmup_steps = int(0.05 * self.length * self.params.epochs)
		total_steps = self.length * self.params.epochs
		rem_steps = total_steps - warmup_steps 
		
		self.optimizer_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters = warmup_steps, start_factor=0.1, end_factor=1)
		self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = rem_steps, eta_min=1e-7)
		
		self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.optimizer_scheduler_warmup, self.optimizer_scheduler], milestones=[warmup_steps])
		self.scaler = torch.amp.GradScaler('cuda')
		self.num_bands = len(self.params.bands)
		
	def Normalize(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		x = torch.log1p(x)		
		x_flat = x.view(Bz, num_chans, num_bands, num_patch * patch_size).contiguous()
		
		#if stats is None:
			#x_mean = torch.mean(x_flat, dim=-1, keepdim=True)
			#x_std = torch.std(x_flat, dim=-1, keepdim=True) + 1e-3
		x_min = x_flat.min(dim=-1, keepdim=True).values
		x_max = x_flat.max(dim=-1, keepdim=True).values
		#	stats = (x_mean, x_std, x_min, x_max)
		#else:
			#x_mean, x_std, x_min, x_max = stats
		
		#x_flat = (x_flat - x_mean) / x_std
		x_flat = (x_flat - x_min) / (x_max - x_min + 1e-4)
		x = x_flat.view(Bz, num_chans, num_bands, num_patch, patch_size).contiguous()
		return x#, stats
			
	def trainer(self):
		
		for epoch in range(self.params.epochs):
			self.optimizer.zero_grad()
			losses = []
			self.model.train()
			
			running_loss = torch.zeros(1, device=self.device)
			running_tmp = torch.zeros(1, device=self.device)
			running_bands = None
			
			iterator = tqdm(self.data_loader, mininterval=10)
			
			for i, x in enumerate(iterator):
				Bz, num_chans, num_patch, patch_size = x.shape
				x = x.to(self.device, non_blocking=True) / 10
				#print(x.shape)
				
				with torch.no_grad():
					psd_x = STFTTransform(x, self.params.d_model, self.params.bands)
					#psd_x = smooth_psd_tensor(psd_x, window_size=10)	# You might want to remove this and try running one time as well.
					psd_x = self.Normalize(psd_x)
					mask = generate_mask(Bz, num_chans, num_patch, self.params.mask_ratio, self.device)
				#with torch.amp.autocast('cuda'):
					#with cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
				with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
					y = self.model(x, psd_x, mask)
				
			#	if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
			#		unwrapped_model = self.model.module
				#else:
				#	unwrapped_model = self.model
	
				# Export the unwrapped version
				#unwrapped_model = unwrapped_model.float()
				#x_export = x.float()
				#psd_x_export = psd_x.float()
				#mask_export = mask.float()
				#is_causal_export = False
				#need_key_padding_export = False
				#export_args = (x_export, psd_x_export, mask_export, is_causal_export, need_key_padding_export)
				#torch.onnx.export(unwrapped_model, export_args, "model_graph.onnx")
				y = y.float()
				
				psd_y = STFTTransform(y.float(), self.params.d_model, self.params.bands)
#				----------------
				# There was extra here.
				#------------------------------------------------------
				#psd_y = psd_y.to(torch.float32) does not work
				if (~torch.isfinite(psd_y)).any().item():

					print(f'psd_y : {is_corrupted}') # True
				#--------------------------------------------------------
				psd_y = self.Normalize(psd_y)
				if (~torch.isfinite(psd_y)).any().item():
					print(f'0.5 : {is_corrupted}') # True
				#psd_y = psd_y.to(torch.half)
				#----------------------------------------------------
				if (~torch.isfinite(psd_y)).any().item():
				
					psd_y_f = psd_y.detach().float() # Quantile requires float32 or float64
					q = torch.tensor([0.10, 0.50, 0.90], device=psd_y.device) # 10th, 50th, 90th percentiles
					percentiles = torch.quantile(psd_y_f, q)

					print(f"(psd_y)10% of data is below: {percentiles[0]:.4f}")
					print(f"(psd_y)Median (50%) is:	  {percentiles[1]:.4f}")
					print(f"(psd_y)90% of data is below: {percentiles[2]:.4f}")

					print('psd_y_norm : {is_corrupted}') # True
				#----------------------------------------------------
				mask_y = y[mask == 1]
				mask_x = x[mask == 1]
					
				loss_tmp = self.criterion(mask_y, mask_x)
				bands_y = [b.squeeze(2) for b in torch.chunk(psd_y, chunks=self.num_bands, dim=2)]
					
				bands_x = [b.squeeze(2) for b in torch.chunk(psd_x, chunks=self.num_bands, dim=2)]
				loss_bands = [self.criterion(band_y[mask == 1], band_x[mask == 1]) * (1 + i * 0.25) for i, (band_y, band_x) in enumerate(zip(bands_y, bands_x))]
					
				loss = (loss_tmp * 14 + 1.75 * sum(loss_bands)) / self.accumulation_steps
				#----------------------------------------------------
				if (~torch.isfinite(loss)).any().item():
					print(f'loss : is_corrupted, skipping_batch, {loss}', flush=True) # True
				#----------------------------------------------------
					if (~torch.isfinite(psd_x)).any().item():
					
						psd_x_f = psd_x.detach().float() # Quantile requires float32 or float64
						q = torch.tensor([0.10, 0.50, 0.90], device=psd_x.device) # 10th, 50th, 90th percentiles
						percentiles = torch.quantile(psd_x_f, q)

						print(f"(psd_x)10% of data is below: {percentiles[0]:.4f}")
						print(f"(psd_x)Median (50%) is:	  {percentiles[1]:.4f}")
						print(f"(psd_x)90% of data is below: {percentiles[2]:.4f}")

						print(f'psd_x : is_corrupted, {psd_x}') # True
					self.optimizer.zero_grad(set_to_none=True)	
					print(self.scaler.get_scale())
					print(f'loss:{loss}, loss_tmp:{loss_tmp}, sum_lbands:{sum(loss_bands)}')
					continue
				#----------------------------------------------------
				loss = loss.float()
				loss.backward()
				
				running_loss += loss.detach() * self.accumulation_steps
				
				running_tmp += loss_tmp.detach()
				
				
				if running_bands is None:
					running_bands = [torch.zeros(1, device=self.device) for _ in range(len(loss_bands))]
				
				for j, l_b in enumerate(loss_bands):
					running_bands[j] += l_b.detach()

				if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(iterator):
				
					if self.params.clip_value is not None:
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
						
					self.optimizer.step()
					
					self.optimizer.zero_grad(set_to_none=True)
					self.scheduler.step()
					
				if i % 500 == 0:
					plot_eeg_reconstruction(
							   x=x, 
							   y=y, 
							   psd_x=psd_x, 
							   psd_y=psd_y, 
							   epoch=epoch, 
							   step=i,
							   mask=mask, 
							   bands=self.params.bands
						   )
					mean_loss = (running_loss / len(self.data_loader)).item()
					mean_tmp = (running_tmp / len(self.data_loader)).item()
					
					mean_bands = [(b / len(self.data_loader)).item() for b in running_bands]
	
					with open(os.path.expanduser('~/CantusCerebra/loss.txt'), 'a') as m:
						bands_log_str = ", ".join([f"{b:.6f}" for b in mean_bands])
						m.write(f'{mean_loss:.6f}, {mean_tmp:.6f}, {bands_log_str}\n')
					for name, p in self.model.named_parameters():
						if p.grad is not None:
							with open(os.path.expanduser('~/CantusCerebra/grad.txt'), 'a') as m:
								m.write(f"{name}: {p.grad.norm().item():.4f}")

			mean_loss = (running_loss / len(self.data_loader)).item()
			mean_tmp = (running_tmp / len(self.data_loader)).item()
			
			mean_bands = [(b / len(self.data_loader)).item() for b in running_bands]

			with open(os.path.expanduser('~/CantusCerebra/loss.txt'), 'a') as m:
				bands_log_str = ", ".join([f"{b:.6f}" for b in mean_bands])
				m.write(f'{mean_loss:.6f}, {mean_tmp:.6f}, {bands_log_str}\n')
					
			# We must save each and every model for aggressive checkpointing.
					
			model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
					
			state_dict_path = os.path.join(self	.params.model_dir, f'pretrain_weights_{epoch}.pth')
			torch.save(model_to_save.state_dict(), state_dict_path)
					
			optim_dict_path = os.path.join(self.params.model_dir, f'optim_weights_{epoch}.pth')
			torch.save(self.optimizer.state_dict(), optim_dict_path)
					
			bands_print_str = ", ".join([f"Band {idx}: {b:.4f}\n" for idx, b in enumerate(mean_bands)])
			print(f'Epoch: {epoch}, Mean loss: {mean_loss:.4f}, tmp:{mean_tmp:.4f}, {bands_print_str}')
			print(f'Model and optim saved.')
					
def setup_seed(seed):
	torch.manual_seed(seed) 	
	torch.cuda.manual_seed_all(seed)	
	np.random.seed(seed)	
	random.seed(seed)
	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = True
	
def main():	
	
	parser = argparse.ArgumentParser('Brainwave decoding')
	parser.add_argument('--d_model', type=int, default=200)
	parser.add_argument('--convolution_set', type=str, default='[(1,), (3,), (5,)]')
	parser.add_argument('--stride', type=int, default=1)
	
	parser.add_argument('--in_dim', type=int, default=200)
	parser.add_argument('--out_dim', type=int, default=200)
	parser.add_argument('--dropout', type=float, default=0.1)
	
	parser.add_argument('--d_ffn', type=int, default=800)
	parser.add_argument('--num_heads', type=int, default=5)
	parser.add_argument('--num_layers', type=int, default=6)
	
	parser.add_argument('--mother_wavelet', type=str, default='cmor1.5-1.0')
	parser.add_argument('--bands', type=str, default='[(0.5, 5), (4, 9), (8, 14), (13, 31), (30, 76)]')	
	parser.add_argument('--num_decoder_layers', type=int, default=2)
	
	parser.add_argument('--kernel_size', type=int, default=3)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--clip_value', type=float, default=1)
	
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--weight_decay', type=float, default=5e-2)
	parser.add_argument('--mask_ratio', type=float, default=0.5)
	
	parser.add_argument('--parallel', action='store_false')
	parser.add_argument('--cuda', type=int, default=0)
	parser.add_argument('--avail_gpus', type=str, default='0')
	
	parser.add_argument('--model_dir', type=str, default='~/CantusCerebra/saved_fm')
	parser.add_argument('--epochs', type=int, default=40)
	parser.add_argument('--need_key_padding', action='store_true')
	
	parser.add_argument('--need_cross_attn_mask', action='store_true')
	parser.add_argument('--is_causal', action='store_true')
	parser.add_argument('--dataset_dir', type=str, default='~/CantusCerebra/data/TUEG', help='root of dataset')
	
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--num_chans', type=int, default=16)
	

	params = parser.parse_args()
	
	params.model_dir = os.path.expanduser(params.model_dir)
	
	params.bands = ast.literal_eval(params.bands)
	
	params.convolution_set = ast.literal_eval(params.convolution_set)
	
	params.dataset_dir = os.path.expanduser(params.dataset_dir)
	
	setup_seed(params.seed)
	
	dataset = LoadDataset(params.dataset_dir)
	
	data_loader = DataLoader(
		dataset,
		batch_size=params.batch_size,
		num_workers=16,
		shuffle=True,
	)
	
	device = torch.device(f'cuda:{params.cuda}') if torch.cuda.is_available() else 'cpu'
	
	sorted_map = sorted_maps().to(device)
	
	model = CantusCerebraUno(sorted_map, num_layers=params.num_layers, in_dim=params.in_dim, d_model=params.d_model, num_heads=params.num_heads, dropout=params.dropout, convolution_set=params.convolution_set, d_ffn=params.d_ffn, out_dim=params.out_dim)
	
	train = Train(params, model, data_loader)	
	train.trainer()	
	
if __name__ == '__main__':
	main()
