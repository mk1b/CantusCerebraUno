import torch
import argparse
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import random

import os
import mne
import ast
import torch.distributed as dist

from datasets.tuab_dataset import LoadDataset as LoadDataset_tuab	# edit here
from model.CantusCerebraUno import CantusCerebraUno
from model.model_for_tuab_eval import Model	# edit here
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
	precision_recall_curve, auc, r2_score, mean_squared_error
import torch.nn.functional as F

hcp_positions_path = os.path.expanduser('~/CantusCerebra/data/HCP/positions_100_7.txt')
connectivity_path = os.path.expanduser('~/CantusCerebra/processed_data/connectivity_matrix.txt')

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

class Evaluator:
	def __init__(self, params, data_loader, model, epoch):
	
		self.params = params
		self.data_loader = data_loader
		self.device = torch.device(f'cuda:{self.params.cuda}') if torch.cuda.is_available() else 'cpu'
		self.model = model
		
		self.criterion_for_binary_class = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)	# BCEWithLogitsLoss because of the asymmetry in the data labels !
		self.criterion_for_multiclass = nn.CrossEntropyLoss().to(self.device)
		self.criterion_for_regression = nn.MSELoss().to(self.device)
		if self.params.use_pretrained_weights:
			map_location = self.device	
			self.params.state_dict_path = os.path.join(self.params.state_dict_path, f'finetune_weights_{epoch}.pth')
			state_dict = torch.load(self.params.state_dict_path, map_location=map_location)
			new_state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
				
			model_state_dict = self.model.state_dict()
			matching_state_dict = {k:v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}	
			
			model_state_dict.update(matching_state_dict)		
			self.model.load_state_dict(model_state_dict)	
		self.model = model.to(self.device)
		
		#weight = torch.tensor([10.0]).to(self.device)
		#self.criterion = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
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
			
	def get_metrics_for_multiclass(self):
		# 1. Safety first: Lock BatchNorm and turn off Dropout
		self.model.eval()

		truths = []
		preds = []
		losses = []

		# 2. Stop gradient tracking to prevent Out Of Memory (OOM) GPU crashes!
		with torch.no_grad():
			# Added a desc to tqdm so you know which phase is running
			for x, y in tqdm(self.data_loader, mininterval=1, desc="Evaluating Multiclass"):
				# Using self.device is generally safer than hardcoding .cuda()
				x = x.to(self.device)
				y = y.to(self.device).long() 
				with torch.no_grad():
					psd_x = STFTTransform(x, self.params.d_model, self.params.bands)
					psd_x = self.Normalize(psd_x)

				pred = self.model(x, psd_x)
				
				# Get predictions (argmax)
				pred_y = torch.max(pred, dim=-1)[1]
				
				# Calculate loss (y is already .long() from above)
				loss = self.criterion_for_multiclass(pred, y)
				losses.append(loss.item())

				# 3. The Squeeze Fix: view(-1) guarantees a 1D array even if batch size is 1
				# 4. The Speed Fix: .extend() is cleaner and faster than += with .tolist()
				truths.extend(y.cpu().view(-1).numpy())
				preds.extend(pred_y.cpu().view(-1).numpy())

		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)

		# Scikit-learn metrics
		acc = balanced_accuracy_score(truths, preds)
		f1 = f1_score(truths, preds, average='weighted')
		kappa = cohen_kappa_score(truths, preds)

		return acc, kappa, f1, mean_loss
	
	def get_metrics_for_binaryclass(self):
		self.model.eval() # 1. Set to evaluation mode
		
		truths = []
		preds = []
		scores = []
		losses = []
		
		# 2. Prevent memory leaks!
		with torch.no_grad():
			for i, (x, y) in enumerate(tqdm(self.data_loader, mininterval=1, desc="Evaluating Binary")):
				x = x.to(self.device)
				y = y.to(self.device).float()
				with torch.no_grad():
					psd_x = STFTTransform(x, self.params.d_model, self.params.bands)
					psd_x = self.Normalize(psd_x)
				# Forward pass
				logit = self.model(x, psd_x)
				
				# Flatten everything to 1D (BatchSize,) to avoid shape bugs
				logit_flat = logit.view(-1)
				y_flat = y.view(-1)
				
				# Calculate metrics
				score_y = torch.sigmoid(logit_flat)
				pred_y = torch.gt(score_y, 0.5).long()
				
				# Calculate loss safely
				loss = self.criterion_for_binary_class(logit_flat, y_flat)
				losses.append(loss.item())
				
				# 3. Use extend with flat numpy arrays (much cleaner/faster)
				truths.extend(y_flat.cpu().long().numpy())
				preds.extend(pred_y.cpu().numpy())
				scores.extend(score_y.cpu().numpy())
				if not logit.isfinite().any():
					print('LOGITTTTTTT')
				if not psd_x.isfinite().any():
					print('PSD_X!!!!')
				if not x.isfinite().any():
					print('X !!!!!!')
				if not score_y.isfinite().any():
					print('SCORE_YYYYYYYYYY')
				if not pred_y.isfinite().any():
					print('PREDDDD_YYYYYYYY')
				if not loss.isfinite().any():
					print('LOSSSSSSSSSSS')
				#if i == 160:
				#	break
				
		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)
		scores = np.array(scores)
		
		# Calculate Scikit-Learn Metrics
		acc = balanced_accuracy_score(truths, preds)
		cohen = cohen_kappa_score(truths, preds)
		
		# PR AUC
		precision, recall, _ = precision_recall_curve(truths, scores, pos_label=1)
		pr_auc = auc(recall, precision)
		
		# ROC AUC (Added a try-except block just in case a batch only has one class)
		try:
			roc_auc = roc_auc_score(truths, scores)
		except ValueError:
			roc_auc = 0.0 # Failsafe if the val set somehow lacks both positive and negative examples
			
		return acc, pr_auc, roc_auc, cohen, mean_loss
	
	def get_metrics_for_regression(self):
		# 1. Turn off Dropout and lock BatchNorm
		self.model.eval() 
		
		truths = []
		preds = []
		losses = []
		
		# 2. Stop saving gradients to prevent out-of-memory crashes!
		with torch.no_grad(): 
			for x, y in tqdm(self.data_loader, mininterval=1):
				x = x.to(self.device)
				y = y.to(self.device).float() # Added .float() just to be safe for regression
				with torch.no_grad():
					psd_x = STFTTransform(x, self.params.d_model, self.params.bands)
					psd_x = self.Normalize(psd_x)
				
				pred = self.model(x, psd_x)
				
				truths += y.cpu().squeeze(-1).numpy().tolist()
				preds += pred.cpu().squeeze(-1).numpy().tolist()
				
				# Squeeze the inputs to the loss function too, just in case!
				loss = self.criterion_for_regression(pred.squeeze(-1), y.squeeze(-1))
				losses.append(loss.item())
		
		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)
		
		# Adding a try-except block in case truths/preds are constant (prevents NaN crash)
		try:
			corrcoef = np.corrcoef(truths, preds)[0, 1]
		except Exception:
			corrcoef = 0.0
			
		r2 = r2_score(truths, preds)
		rmse = mean_squared_error(truths, preds) ** 0.5
		
		return corrcoef, r2, rmse, mean_loss

def seed_init(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True

def main():
	
	parser = argparse.ArgumentParser(description='argparser')
	parser.add_argument('--cuda', type=int, default=0)
	parser.add_argument('--model_dir', type=str, default='~/CantusCerebra/saved_fm')
	parser.add_argument('--use_pretrained_weights', action='store_false')
	parser.add_argument('--state_dict_path', type=str, default='~/CantusCerebra/saved_fm')
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--dataset_dir', type=str, default='~/CantusCerebra/processed_data/processed_siena/json_generate')
	parser.add_argument('--seed', type=int, default=42)
	
	parser.add_argument('--dropout', type=float, default=0.1, help='dropout value')
	parser.add_argument('--in_dim', type=int, default=200, help='Number of samples in 1s raw')		
	parser.add_argument('--out_dim', type=int, default=200, help='Output dimension')
	parser.add_argument('--d_model', type=int, default=200, help='Model operating dimension')
	parser.add_argument('--d_ffn', type=int, default=800, help='Standard 2-layer FFN dimensions')
	parser.add_argument('--num_layers', type=int, default=6, help='Number of Transformer layers')
	parser.add_argument('--num_heads', type=int, default=8, help='Number of Heads in MHSA')
	parser.add_argument('--convolution_set', type=str, default='[(1,), (3,), (5,)]', help='Concentrated convolution sizes. < num_chans')
	parser.add_argument('--seq_len', type=int, default=30, help='num_patches')
	parser.add_argument('--is_causal', action='store_true', help='If you want causal Temporal Attention')
	parser.add_argument('--need_key_padding', action='store_true', help='if any padding that could be added is to be ignored')
	parser.add_argument('--stride', type=int, default=1, help='stride for temp convs')
	parser.add_argument('--epch', type=int, default=0)
	parser.add_argument('--bands', type=str, default='[(0.5, 5), (4, 9), (8, 14), (13, 31), (30, 76)]')	
	parser.add_argument('--mode', type=str, default='bin')
	
	params = parser.parse_args()
	params.bands = ast.literal_eval(params.bands)
	params.state_dict_path = os.path.expanduser(params.state_dict_path)
	params.state_dict_path = os.path.join(params.state_dict_path)
	
	params.dataset_dir = os.path.expanduser(params.dataset_dir)
	#params.model_dir = os.path.expanduser(params.model_dir)
	device = torch.device(f'cuda:{params.cuda}')
	params.convolution_set = ast.literal_eval(params.convolution_set)
	
	seed_init(params.seed)
	
	tuab_dataset = LoadDataset_tuab(params)	# edit here
	data_loader = tuab_dataset.get_data_loader()	# edit here.
	
	configured_model = Model(params)
	configured_model.eval()
	evaluator = Evaluator(params, data_loader['test'], configured_model, params.epch)
	if mode == 'bin':
		b_acc, pr_auc, auroc, cohen, mean_loss = evaluator.get_metrics_for_binaryclass()
		print(f'b_acc: {b_acc}, pr_auc: {pr_auc}, auroc: {auroc}, cohen: {cohen}, mean_loss: {mean_loss}')
	if mode == 'reg':
		corrcoef, r2, rmse, mean_loss = evaluator.get_metrics_for_regression()
		ls = [corrcoef, r2, rmse, mean_loss]
		if np.isfinite(ls):
			print('ok')
		else:
			print('not ok')
		print(f'corrcoef: {corrcoef}, r2: {r2}, rmse: {rmse}, mean_loss: {mean_loss}')
	if mode == 'mul':
		b_acc, kappa, f1, mean_loss = evaluator.get_metrics_for_multiclass()
		print(f'b_acc: {b_acc}, kappa: {kappa}, f1: {f1}, mean_loss: {mean_loss}')
	
if __name__ == '__main__':
	main()
	
	
