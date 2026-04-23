import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random
import os
import mne
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

#from model import model_for_seedv,model_for_bciciv2a, model_for_tuab, model_for_tuev,model_for_faced,model_for_chb,model_for_speech,model_for_tusl,model_for_shu,model_for_seedvig,model_for_physio,model_for_isruc
#from model import model_for_siena, model_for_hmc,model_for_stress,model_for_mumtaz
from datasets.tuab_dataset import LoadDataset as LoadDataset_tuab
from model import model_for_tuab
from model.CantusCerebraUno import *
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
	precision_recall_curve, auc, r2_score, mean_squared_error
import ast
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.backends.cuda as cuda
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly=True

hcp_positions_path = os.path.expanduser('~/CantusCerebra/data/HCP/positions_100_7.txt')
connectivity_path = os.path.expanduser('~/CantusCerebra/processed_data/connectivity_matrix.txt')

def STFTTransform(x, sampling_frequency, bands=[(1, 5), (4, 9), (8, 14), (13, 31), (30, 76)]):
	Bz, num_chans, num_patch, patch_size = x.shape
	Total_Length = num_patch * patch_size
	
	x_flat = x.reshape(Bz * num_chans, Total_Length)
	
	n_fft = 256 
	hop_length = 16 
	window = torch.hann_window(n_fft, device=x.device)

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

	band_cat = torch.stack(band_means, dim=0) 
	band_cat = band_cat.permute(1, 0, 2)
	band_cat = F.interpolate(band_cat, size=Total_Length, mode='linear', align_corners=False)
	psd = band_cat.view(Bz, num_chans, len(bands), num_patch, patch_size)
	
	return psd


class Evaluator:
	def __init__(self, params, data_loader, model, epoch, local_rank):
		self.params = params
		self.data_loader = data_loader
		self.local_rank = local_rank
		self.device = torch.device(f'cuda:{self.local_rank}')
		
		# DDP Model evaluation doesn't strictly need DDP wrapper, but we keep it intact
		self.model = model 
		
		self.criterion_for_binary_class = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
		self.criterion_for_multiclass = nn.CrossEntropyLoss().to(self.device)
		self.criterion_for_regression = nn.MSELoss().to(self.device)
		
	def Normalize(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		x = torch.log1p(x)		
		x_flat = x.view(Bz, num_chans, num_bands, num_patch * patch_size).contiguous()
		x_min = x_flat.min(dim=-1, keepdim=True).values
		x_max = x_flat.max(dim=-1, keepdim=True).values
		x_flat = (x_flat - x_min) / (x_max - x_min + 1e-4)
		x = x_flat.view(Bz, num_chans, num_bands, num_patch, patch_size).contiguous()
		return x
			
	def get_metrics_for_multiclass(self):
		self.model.eval()
		truths = []
		preds = []
		losses = []

		with torch.no_grad():
			# Only rank 0 gets the progress bar
			loader = tqdm(self.data_loader, mininterval=1, desc="Evaluating Multiclass", disable=self.local_rank != 0)
			for x, y in loader:
				x = x.to(self.device)
				y = y.to(self.device).long() 
				with torch.no_grad():
					psd_x = STFTTransform(x, self.params.d_model, self.params.bands)
					psd_x = self.Normalize(psd_x)
				x = x.float()
				psd_x = psd_x.float()
				pred = self.model(x, psd_x)
				pred_y = torch.max(pred, dim=-1)[1]
				loss = self.criterion_for_multiclass(pred, y)
				losses.append(loss.item())

				truths.extend(y.cpu().view(-1).numpy())
				preds.extend(pred_y.cpu().view(-1).numpy())

		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)

		acc = balanced_accuracy_score(truths, preds)
		f1 = f1_score(truths, preds, average='weighted')
		kappa = cohen_kappa_score(truths, preds)

		return acc, kappa, f1, mean_loss
	
	import torch.distributed as dist

	def get_metrics_for_binaryclass(self):
		self.model.eval() 
		
		# We will store PyTorch tensors here, NOT numpy arrays yet
		truths = []
		preds = []
		scores = []
		losses = []
		
		with torch.no_grad():
			loader = tqdm(self.data_loader, mininterval=1, desc="Evaluating Binary", disable=self.local_rank != 0)
			for i, (x, y) in enumerate(loader):
				x = x.to(self.device)
				y = y.to(self.device).float()
				
				with torch.no_grad():
					psd_x = STFTTransform(x, self.params.d_model, self.params.bands)
					psd_x = self.Normalize(psd_x)
					
				x = x.float()
				psd_x = psd_x.float()

				logit = self.model(x, psd_x)
				
				logit_flat = logit.float().view(-1)
				y_flat = y.view(-1)
				
				score_y = torch.sigmoid(logit_flat)
				pred_y = torch.gt(score_y, 0.5).long()
				loss = self.criterion_for_binary_class(logit_flat, y_flat)
				
				# Append tensors directly (keeping them on the GPU)
				losses.append(loss)
				truths.append(y_flat)
				preds.append(pred_y)
				scores.append(score_y)
				if not logit.isfinite().any():
					print('FUCK LOGITTTTTTT')
				if not psd_x.isfinite().any():
					print('FUCK PSD_X!!!!')
				if not x.isfinite().any():
					print('FUCK X !!!!!!')
				if not score_y.isfinite().any():
					print('FUCK SCORE_YYYYYYYYYY')
				if not pred_y.isfinite().any():
					print('FUCK PREDDDD_YYYYYYYY')
				if not loss.isfinite().any():
					print('FUCK LOSSSSSSSSSSS')
				#if i == 160:
				#	break
						
		# --- DDP SYNCHRONIZATION PHASE ---
		
		# 1. Concatenate all local batches into single tensors
		local_truths = torch.cat(truths)
		local_preds = torch.cat(preds)
		local_scores = torch.cat(scores)
		local_loss = torch.stack(losses).mean()
		
		# 2. Sync the Loss (using the all_reduce trick)
		dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
		world_size = dist.get_world_size()
		global_mean_loss = (local_loss / world_size).item()

		# 3. Prepare empty lists to catch the arrays from other GPUs
		# Note: all_gather requires the receiving tensors to be the exact same shape
		gathered_truths = [torch.zeros_like(local_truths) for _ in range(world_size)]
		gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
		gathered_scores = [torch.zeros_like(local_scores) for _ in range(world_size)]
		
		# 4. Gather the arrays from all GPUs
		dist.all_gather(gathered_truths, local_truths)
		dist.all_gather(gathered_preds, local_preds)
		dist.all_gather(gathered_scores, local_scores)
		
		# 5. Mash them all together and move to CPU/Numpy for Sklearn!
		global_truths = torch.cat(gathered_truths).cpu().numpy()
		global_preds = torch.cat(gathered_preds).cpu().numpy()
		global_scores = torch.cat(gathered_scores).cpu().numpy()
		 
		# --- METRICS PHASE ---
		
		acc = balanced_accuracy_score(global_truths, global_preds)
		cohen = cohen_kappa_score(global_truths, global_preds)
		
		precision, recall, _ = precision_recall_curve(global_truths, global_scores, pos_label=1)
		pr_auc = auc(recall, precision)
		
		try:
			roc_auc = roc_auc_score(global_truths, global_scores)
		except ValueError:
			roc_auc = 0.0 
			
		return acc, pr_auc, roc_auc, cohen, global_mean_loss
	
	def get_metrics_for_regression(self):
		self.model.eval() 
		truths = []
		preds = []
		losses = []
		
		with torch.no_grad(): 
			loader = tqdm(self.data_loader, mininterval=1, disable=self.local_rank != 0)
			for x, y in loader:
				x = x.to(self.device)
				y = y.to(self.device).float() 
				with torch.no_grad():
					psd_x = STFTTransform(x, self.params.d_model, self.params.bands)
					psd_x = self.Normalize(psd_x)
				x = x.float()
				psd_x = psd_x.float()
				pred = self.model(x, psd_x)
				truths += y.cpu().squeeze(-1).numpy().tolist()
				preds += pred.cpu().squeeze(-1).numpy().tolist()
				
				loss = self.criterion_for_regression(pred.squeeze(-1), y.squeeze(-1))
				losses.append(loss.item())
		
		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)
		
		try:
			corrcoef = np.corrcoef(truths, preds)[0, 1]
		except Exception:
			corrcoef = 0.0
			
		r2 = r2_score(truths, preds)
		rmse = mean_squared_error(truths, preds) ** 0.5
		
		return corrcoef, r2, rmse, mean_loss


class FineTune_Trainer(object):	
	def __init__(self, params, data_loader, model, local_rank):
		super().__init__()
		self.params = params
		self.local_rank = local_rank
		self.device = torch.device(f'cuda:{self.local_rank}')
		
		self.model = model.to(self.device)
		
		# --- DDP INITIALIZATION ---
		# find_unused_parameters=True prevents crashes if some branches in your network are skipped
		self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True, gradient_as_bucket_view=True)
		#torch.compile(self.model, options={'triton.cudagraphs':False})
			
		backbone_parameters = []
		other_parameters = []
		
		for name, parameter in self.model.named_parameters():
			if 'backbone' in name:
				backbone_parameters.append(parameter)
				if self.params.frozen:
					parameter.requires_grad = False
				else:
					parameter.requires_grad = True
			else:
				other_parameters.append(parameter)
				
		if self.params.multi_lr:
			self.optimizer = torch.optim.AdamW([{'params':backbone_parameters, 'lr':self.params.lr}, {'params':other_parameters, 'lr':self.params.lr * 100}], weight_decay=self.params.weight_decay)
		else:
			self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
			
		self.data_loader = data_loader
		self.length = len(data_loader['train'])
		
		self.criterion_for_binary_class = nn.BCEWithLogitsLoss(reduction='mean').to(self.device) 
		self.criterion_for_multiclass = nn.CrossEntropyLoss().to(self.device)
		self.criterion_for_regression = nn.MSELoss().to(self.device)
		
		warmup_steps = int((self.params.epochs * 0.4) // 10 * self.length)
		total_steps = self.params.epochs * self.length
		main_steps = total_steps - warmup_steps
		
		self.optimizer_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=warmup_steps, start_factor=0.1, end_factor=1)
		self.optimizer_scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = main_steps, eta_min=1e-7)
		self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.optimizer_scheduler_warmup, self.optimizer_scheduler_main], milestones=[warmup_steps])
		self.scaler = torch.amp.GradScaler('cuda')
		
	def Normalize(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		x = torch.log1p(x)		
		x_flat = x.view(Bz, num_chans, num_bands, num_patch * patch_size).contiguous()
		x_min = x_flat.min(dim=-1, keepdim=True).values
		x_max = x_flat.max(dim=-1, keepdim=True).values
		x_flat = (x_flat - x_min) / (x_max - x_min + 1e-4)
		x = x_flat.view(Bz, num_chans, num_bands, num_patch, patch_size).contiguous()
		return x
	
	def train_for_binaryclass(self):
		for epoch in range(self.params.epochs):
			if self.local_rank == 0:
				print(f'Epoch {epoch} starts')
			
			# CRITICAL FOR DDP: Shuffles the data properly at each epoch
			if isinstance(self.data_loader['train'].sampler, DistributedSampler):
				self.data_loader['train'].sampler.set_epoch(epoch)
			
			losses = []
			self.model.train()	
			
			loader = tqdm(self.data_loader['train'], mininterval=10, disable=self.local_rank != 0)
			for i, (x, label) in enumerate(loader):	
				self.optimizer.zero_grad()
				x = x.to(self.device)	
				label = label.to(self.device)
				
				with torch.no_grad():
					psd_x = STFTTransform(x, self.params.d_model, self.params.bands)
					psd_x = self.Normalize(psd_x)
				
				x = x.float()
				psd_x = psd_x.float()
				with torch.amp.autocast('cuda'):
					with cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
						logit = self.model(x, psd_x)
						logit = logit.squeeze(-1).contiguous()
						label = label.squeeze(-1).contiguous()
						loss = self.criterion_for_binary_class(logit, label.float())
				loss = loss.float()
				
				self.scaler.scale(loss).backward()
				
				if self.params.clip_value > 0:	
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
				
				self.scaler.step(self.optimizer)
				self.scaler.update()
				self.scheduler.step()
				losses.append(loss.item())	
				if not loss.isfinite().any():
					print('FUCK LOSSSSSSSSSSS')
				if not psd_x.isfinite().any():
					print('FUCK PSD_X!!!!')
				if not x.isfinite().any():
					print('FUCK X !!!!!!')
				#if i == 160:
				#	break
				
			mean_loss = np.mean(losses)	
			
			loss_tensor = torch.tensor(mean_loss, dtype=torch.float32, device=self.device)
			dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
			world_size = dist.get_world_size()
			global_mean_loss = loss_tensor / world_size
			
			# Wait for all processes to finish training epoch before eval/saving
			dist.barrier()
			
			# Rank 0 handles Saving and Evaluating
			if self.local_rank == 0:
				print(f'True Global Epoch Loss: {global_mean_loss.item()}')
				save_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}.pth')
				# DDP wraps original model in `.module`, save the inner model
				torch.save(self.model.module.state_dict(), save_path)
				   
				print('model saved @ ', save_path)
				print(f'Epoch {epoch} model trained. Now commencing testing.')
				print(f'loss: {mean_loss}')
				
			with torch.no_grad():	# All GPUs are going to handle evaluating. It is better than one GPU just waiting. Let some compute go to waste, that's okay.
				self.model.eval()
				evaluator = Evaluator(params=self.params, data_loader=self.data_loader['val'], model=self.model, epoch=epoch, local_rank=self.local_rank)
				b_acc, pr_auc, auroc, cohen, val_mean_loss = evaluator.get_metrics_for_binaryclass()
			
				if self.local_rank == 0:
					print(f'b_acc: {b_acc}, pr_auc: {pr_auc}, auroc: {auroc}, cohen:{cohen}, mean_loss: {val_mean_loss}')
					with open(os.path.expanduser('~/CantusCerebra/logs/finetune_metrics.txt'), 'a') as log:
						log.write(f'{b_acc}, {pr_auc}, {auroc}, {cohen}, {val_mean_loss}\n')
			
			torch.cuda.empty_cache()
			
			dist.barrier()
				
	def train_for_multiclass(self):
		for epoch in range(self.params.epochs):
			if self.local_rank == 0:
				print(f'Epoch {epoch} starts')
			
			if isinstance(self.data_loader['train'].sampler, DistributedSampler):
				self.data_loader['train'].sampler.set_epoch(epoch)
				
			losses = []
			self.model.train()	
			
			loader = tqdm(self.data_loader['train'], mininterval=10, disable=self.local_rank != 0)
			for x, label in loader:	
				self.optimizer.zero_grad()
				x = x.to(self.device)
				with torch.no_grad():	
					psd_x = STFTTransform(x, self.params.d_model, self.params.bands)
					psd_x = self.Normalize(psd_x)
				
				x = x.float()
				psd_x = psd_x.float()
				
				label = label.to(self.device).long()
				with cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
					logit = self.model(x, psd_x).contiguous()
				loss = self.criterion_for_multiclass(logit, label)
				
				loss.backward()
				
				if self.params.clip_value > 0:	
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
			   
				self.optimizer.step()
				self.scheduler.step()
				losses.append(loss.item())	
				
			mean_loss = np.mean(losses)	
			dist.barrier()
			
			if self.local_rank == 0:
				save_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}.pth')
				torch.save(self.model.module.state_dict(), save_path)
				
				print('model saved @ ', save_path)
				print(f'Epoch {epoch} model trained. Now commencing testing.')
				print(f'loss: {mean_loss}')
				
				with torch.no_grad():
					self.model.eval()
					evaluator = Evaluator(params=self.params, data_loader=self.data_loader['val'], model=self.model, epoch=epoch, local_rank=self.local_rank)
					b_acc, cohen, f1, val_mean_loss = evaluator.get_metrics_for_multiclass()
					
					print(f'b_acc: {b_acc}, cohen: {cohen}, f1: {f1}, mean_loss: {val_mean_loss}')
					with open(os.path.expanduser('~/CantusCerebra/logs/finetune_metrics.txt'), 'a') as log:
						log.write(f'{b_acc}, {cohen}, {f1}, {val_mean_loss}\n')
			
			dist.barrier()
				
	def train_for_regression(self):
		for epoch in range(self.params.epochs):
			if self.local_rank == 0:
				print(f'Epoch {epoch} starts')
				
			if isinstance(self.data_loader['train'].sampler, DistributedSampler):
				self.data_loader['train'].sampler.set_epoch(epoch)
			
			losses = []
			self.model.train()	
			
			loader = tqdm(self.data_loader['train'], mininterval=10, disable=self.local_rank != 0)
			for x, label in loader:	
				self.optimizer.zero_grad()
				x = x.to(self.device)	
				with torch.no_grad():
					psd_x = STFTTransform(x, self.params.d_model, self.params.bands)
					psd_x = self.Normalize(psd_x)
					
				x = x.float()
				psd_x = psd_x.float()
				label = label.to(self.device)
				with cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
					logit = self.model(x, psd_x)
				logit = logit.squeeze(-1).contiguous()
				label = label.squeeze(-1).contiguous()
				loss = self.criterion_for_regression(logit, label.float())
				
				loss.backward()
				
				if self.params.clip_value > 0:	
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
				
				self.optimizer.step()
				self.scheduler.step()
				losses.append(loss.item())	
				
			mean_loss = np.mean(losses)	
			dist.barrier()
			
			if self.local_rank == 0:
				save_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}.pth')
				torch.save(self.model.module.state_dict(), save_path)
				
				print('model saved @ ', save_path)
				print(f'Epoch {epoch} model trained. Now commencing testing.')
				print(f'loss: {mean_loss}')
				
				with torch.no_grad():
					self.model.eval()
					evaluator = Evaluator(params=self.params, data_loader=self.data_loader['val'], model=self.model, epoch=epoch, local_rank=self.local_rank)
					corrcoef, r2, rmse, val_mean_loss = evaluator.get_metrics_for_regression()
					
					print(f'corrcoef: {corrcoef}, r2: {r2}, rmse: {rmse}, mean_loss: {val_mean_loss}')
					with open(os.path.expanduser('~/CantusCerebra/logs/finetune_metrics.txt'), 'a') as log:
						log.write(f'{corrcoef}, {r2}, {rmse}, {val_mean_loss}\n')
			
			dist.barrier()

def setup_seed(seed):
	torch.manual_seed(seed)	
	torch.cuda.manual_seed_all(seed)	
	np.random.seed(seed)	
	random.seed(seed)
	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = True

# --- DDP Helper: Convert existing DataLoaders into DDP Sampler DataLoaders ---
def prepare_ddp_loaders(loaders_dict, local_rank, world_size):
	if world_size <= 1: 
		return loaders_dict
	
	new_loaders = {}
	for split, loader in loaders_dict.items():
		if split == 'train':
			# Create a DistributedSampler to ensure each GPU gets unique data
			sampler = DistributedSampler(loader.dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
			# Recreate the DataLoader. (Note: standard DDP turns shuffle=False in the loader itself since the sampler handles it)
			new_loaders[split] = DataLoader(
				loader.dataset, 
				batch_size=loader.batch_size, 
				sampler=sampler,
				num_workers=loader.num_workers, 
				pin_memory=loader.pin_memory, 
				drop_last=loader.drop_last
			)
		else:
			# Validation remains the same. Evaluated fully on Rank 0 in this architecture.
			new_loaders[split] = loader
			
	return new_loaders

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
	

def main():
	# --- DDP Environment Setup ---
	if "LOCAL_RANK" in os.environ:
		local_rank = int(os.environ["LOCAL_RANK"])
		world_size = int(os.environ["WORLD_SIZE"])
		dist.init_process_group(backend='nccl')
		torch.cuda.set_device(local_rank)
	else:
		# Fallback to single GPU if not run with torchrun
		local_rank = 0
		world_size = 1

	parser = argparse.ArgumentParser(description='Finetuning EEG FM')
	parser.add_argument('--use_pretrained_weights', action='store_false', default=True, help='or scratch?')
	parser.add_argument('--state_dict_path', type=str, default='~/CantusCerebra/saved_fm/pretrain_weights.pth', help='state_dict_path')
	parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
	parser.add_argument('--multi_lr', action='store_false', default=True, help='variable learning rates')
	parser.add_argument('--lr', type=float, default=5e-6, help='lr')
	parser.add_argument('--weight_decay', type=float, default=5e-2, help='wt dikey')
	parser.add_argument('--epochs', type=int, default=50, help='num_epochs')
	parser.add_argument('--clip_value', type=float, default=1, help='clip value')
	parser.add_argument('--model_dir', type=str, default='~/CantusCerebra/saved_fm', help='weight storage')
	parser.add_argument('--dataset_dir', type=str, default='~/CantusCerebra/data/TUAB/edf/process_refine', help='your 3 json files folder')
	parser.add_argument('--seed', type=int, default=42, help='seed')
	parser.add_argument('--frozen', action='store_true', help='Pretrained big model will be frozen.')
	parser.add_argument('--batch_size', type=int, default=64, help='Bz')
	parser.add_argument('--ddst', type=str, default='TUAB')
	
	parser.add_argument('--in_dim', type=int, default=200, help='Number of samples in 1s raw')		
	parser.add_argument('--out_dim', type=int, default=200, help='Output dimension')
	parser.add_argument('--d_model', type=int, default=200, help='Model operating dimension')
	parser.add_argument('--d_ffn', type=int, default=800, help='Standard 2-layer FFN dimensions')
	parser.add_argument('--num_layers', type=int, default=6, help='Number of Transformer layers')
	parser.add_argument('--num_heads', type=int, default=8, help='Number of Heads in MHSA')
	parser.add_argument('--convolution_set', type=str, default="[(1,), (3,), (5,)]", help='Concentrated convolution sizes. < num_chans')
	parser.add_argument('--seq_len', type=int, default=30, help='num_patches')
	parser.add_argument('--is_causal', action='store_true', help='If you want causal Temporal Attention')
	parser.add_argument('--need_key_padding', action='store_true', help='if any padding that could be added is to be ignored')
	parser.add_argument('--stride', type=int, default=1, help='stride for temp convs')
	parser.add_argument('--dataset', type=str, default='TUAB')
	parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
	parser.add_argument('--bands', type=str, default='[(0.5, 5), (4, 9), (8, 14), (13, 31), (30, 76)]')   
	
	params = parser.parse_args()
	setup_seed(params.seed)	
	
	params.model_dir = os.path.expanduser(params.model_dir)
	params.dataset_dir = os.path.expanduser(params.dataset_dir)
	params.state_dict_path = os.path.expanduser(params.state_dict_path)
	
	params.convolution_set = ast.literal_eval(params.convolution_set)
	params.bands = ast.literal_eval(params.bands)
	
	if local_rank == 0:
		print('The downstream dataset is {}'.format(params.ddst))
		
	if params.ddst == 'FACED': 
		params.num_of_classes = 9
		load_dataset = faced_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_faced.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_multiclass()
	elif params.ddst == 'SEED-V':
		params.num_of_classes = 5
		load_dataset = seedv_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_seedv.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_multiclass() 
	elif params.ddst == 'PhysioNet-MI':
		params.num_of_classes = 4
		load_dataset = physio_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_physio.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_multiclass()
	elif params.ddst == 'SHU-MI':
		params.num_of_classes = 2
		load_dataset = shu_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_shu.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_binaryclass()
	elif params.ddst == 'ISRUC':
		params.num_of_classes = 5
		load_dataset = isruc_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_isruc.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_multiclass()
	elif params.ddst == 'CHB-MIT':
		params.num_of_classes = 2
		load_dataset = chb_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_chb.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_binaryclass()
	elif params.ddst == 'BCIC2020-3':
		params.num_of_classes = 5
		load_dataset = speech_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_speech.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_multiclass() 
	elif params.ddst == 'Mumtaz2016':	
		params.num_of_classes = 2
		load_dataset = mumtaz_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_mumtaz.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_binaryclass()
	elif params.ddst == 'SEED-VIG':
		params.num_of_classes = 1
		load_dataset = seedvig_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_seedvig.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_regression()
	elif params.ddst == 'MentalArithmetic':
		params.num_of_classes = 2
		load_dataset = stress_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_stress.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_binaryclass()
	elif params.ddst == 'TUEV':
		params.num_of_classes = 6
		load_dataset = tuev_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_tuev.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_multiclass() 
	elif params.ddst == 'TUAB':
		params.num_of_classes = 2
		tuab_dataset = LoadDataset_tuab(params)
		data_loader = prepare_ddp_loaders(tuab_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_tuab.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_binaryclass()
	elif params.ddst == 'TUSL':
		params.num_of_classes = 8
		load_dataset = tusl_dataset.get_data_loader(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_dataloader, local_rank, world_size)
		model = model_for_tusl.Model(params) 
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_multiclass()
	elif params.ddst == 'BCIC-IV-2a':
		params.num_of_classes = 4
		load_dataset = bciciv2a_dataset.LoadDataset(params)
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_bciciv2a.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_multiclass() 
	elif params.ddst == 'siena': 
		params.num_of_classes = 2
		load_dataset = siena_dataset.LoadDataset(params) 
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_siena.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_binaryclass()
	elif params.ddst == 'HMC': 
		params.num_of_classes = 5 
		load_dataset = hmc_dataset.LoadDataset(params) 
		data_loader = prepare_ddp_loaders(load_dataset.get_data_loader(), local_rank, world_size)
		model = model_for_hmc.Model(params)
		t = FineTune_Trainer(params, data_loader, model, local_rank)
		t.train_for_multiclass()
		
	if local_rank == 0:
		print('Done!!!!!')
		
	dist.destroy_process_group()
	
if __name__== '__main__':
	main()
