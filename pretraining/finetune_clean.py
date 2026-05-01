import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random
from model.TmpEncoder_stage5_clean import *
import os
import mne

from datasets.faced_dataset import LoadDataset as LoadDataset

from model.TmpEncoder_stage5_clean import Final
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
        precision_recall_curve, auc, r2_score, mean_squared_error

import ast


hcp_positions_path = os.path.expanduser('~/simpletmp/data/HCP/positions_100_7.txt')
connectivity_path = os.path.expanduser('~/simpletmp/processed_data/connectivity_matrix.txt')

class Evaluator:
	def __init__(self, params, data_loader, model, epoch):
	
		self.params = params
		self.data_loader = data_loader
		self.device = torch.device(f'cuda:{self.params.cuda}') if torch.cuda.is_available() else 'cpu'
		self.model = model.to(self.device)
		
		self.criterion_for_binary_class = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)	# BCEWithLogitsLoss because of the asymmetry in the data labels !
		self.criterion_for_multiclass = nn.CrossEntropyLoss().to(self.device)
		self.criterion_for_regression = nn.MSELoss().to(self.device)
		
		#weight = torch.tensor([10.0]).to(self.device)
		#self.criterion = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
			
	def get_metrics_for_multiclass(self):
		# 1. Safety first: Lock BatchNorm and turn off Dropout
		self.model.eval()

		truths = []
		preds = []
		losses = []

		# 2. Stop gradient tracking to prevent Out Of Memory (OOM) GPU crashes!
		with torch.no_grad():
			# Added a desc to tqdm so you know which phase is running
			for i, (x, y) in enumerate(tqdm(self.data_loader, mininterval=1, desc="Evaluating Multiclass")):
				# Using self.device is generally safer than hardcoding .cuda()
				x = x.to(self.device)
				y = y.to(self.device).long() 

				pred = self.model(x)
				
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
				# Forward pass
				logit = self.model(x)
				
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
				
				pred = self.model(x)
				
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
	
class ConfiguredModel(nn.Module):
	def __init__(self, model, params):
		super().__init__()
		
		self.backbone = model	
		self.params = params
			
		self.FFN = nn.Sequential(
						nn.Linear(200, 100),
						nn.ELU(),
						nn.Dropout(params.dropout),
						nn.Linear(100, self.params.nc),
					)
						
	def forward(self, x):
		Bz, num_chans, num_patches, patch_size = x.shape
		
		emb = self.backbone(x)
		emb = emb.mean(dim=(1, 2))
		
		out = self.FFN(emb)	
		out = out.reshape(Bz, self.params.nc)
		
		return out

class FineTune_Trainer(object):	
	def __init__(self, params, data_loader, model):
		super().__init__()
		
		self.params = params
		self.model = model
				
		self.device = torch.device(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
		self.model = self.model.to(self.device)
			
		backbone_parameters = []
		other_parameters = []
		
		for name, parameter in model.named_parameters():
			if 'backbone' in name:
				backbone_parameters.append(parameter)
				
				if self.params.frozen:
					parameter.requires_grad = False
					
				else:
					parameter.requires_grad = True
					
			else:
				other_parameters.append(parameter)
				
		#if self.params.multi_lr:
		#	self.optimizer = torch.optim.AdamW([{'params':backbone_parameters, 'lr':self.params.lr}, {'params':other_parameters, 'lr':self.params.lr * 100}], weight_decay=self.params.weight_decay)
			
		#else:
			self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
			
		self.data_loader = data_loader
		self.length = len(data_loader['train'])
		
		#weight = torch.tensor([10.0]).to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
		self.criterion_for_binary_class = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)	# BCEWithLogitsLoss because of the asymmetry in the data labels !
		self.criterion_for_multiclass = nn.CrossEntropyLoss().to(self.device)
		self.criterion_for_regression = nn.MSELoss().to(self.device)
		
		warmup_steps = (self.params.epochs * 0.4) // 10 * self.length
		total_steps = self.params.epochs * self.length
		main_steps = total_steps - warmup_steps
		
		self.optimizer_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=warmup_steps, start_factor=0.1, end_factor=1)
		self.optimizer_scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = main_steps, eta_min=1e-7)
		
		self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.optimizer_scheduler_warmup, self.optimizer_scheduler_main], milestones=[warmup_steps])
	
	def train_for_binaryclass(self):
	
		best_loss = float('inf')
		
		for epoch in range(self.params.epochs):
			print(f'Epoch {epoch} starts')
			
			losses = []
			self.model.train()	
			
			for i, (x, label) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):	
			
				self.optimizer.zero_grad()
				
				x = x.to(self.device)	
				label = label.to(self.device)
				
				logit = self.model(x)
				logit = logit.squeeze(-1)
				label = label.squeeze(-1)
				loss = self.criterion_for_binary_class(logit, label.float())
				
				loss.backward()
				
				if self.params.clip_value > 0:	
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

				self.optimizer.step()
				self.scheduler.step()
				
				losses.append(loss.item())
				
			mean_loss = np.mean(losses)	
			
			save_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}.pth')
			torch.save(self.model.state_dict(), save_path)
				
			print('model saved @ ', save_path)
			#best_loss = mean_loss
				
			print(f'Epoch {epoch} model trained. Now commencing testing.')
			print(f'loss: {mean_loss}')
			
			with torch.no_grad():
				self.model.eval()
				
				evaluator = Evaluator(params=self.params, data_loader=self.data_loader['val'], model=self.model, epoch=epoch)
				b_acc, pr_auc, auroc, cohen, mean_loss = evaluator.get_metrics_for_binaryclass()
				
				print(f'b_acc: {b_acc}, pr_auc: {pr_auc}, auroc: {auroc}, cohen:{cohen}, mean_loss: {mean_loss}')
				
	def train_for_multiclass(self):
		best_loss = float('inf')
		
		for epoch in range(self.params.epochs):
			print(f'Epoch {epoch} starts')
			
			losses = []
			self.model.train()	
			
			for i, (x, label) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):	
			
				self.optimizer.zero_grad()
				
				x = x.to(self.device)
				label = label.to(self.device).long()

				logit = self.model(x)	# logit is(Bz, self.num_classes)
				loss = self.criterion_for_multiclass(logit, label)
				
				loss.backward()
					
				if self.params.clip_value > 0:	
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
					
				self.optimizer.step()
				self.scheduler.step()
				
				losses.append(loss.item())	
				print(np.argmax(logit.detach().cpu().numpy(), axis=1))
				
			mean_loss = np.mean(losses)	
			
			save_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}.pth')
			torch.save(self.model.state_dict(), save_path)
				
			print('model saved @ ', save_path)
				
			print(f'Epoch {epoch} model trained. Now commencing testing.')
			print(f'loss: {mean_loss}')
			
			with torch.no_grad():
				self.model.eval()
				
				evaluator = Evaluator(params=self.params, data_loader=self.data_loader['val'], model=self.model, epoch=epoch)
				b_acc, cohen, f1, mean_loss = evaluator.get_metrics_for_multiclass()
				
				print(f'b_acc: {b_acc}, cohen: {cohen}, f1: {f1}, mean_loss: {mean_loss}')
				
	def train_for_regression(self):
		best_loss = float('inf')
		
		for epoch in range(self.params.epochs):
			print(f'Epoch {epoch} starts')
			
			losses = []
			self.model.train()	
			
			for i, (x, label) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):	
			
				self.optimizer.zero_grad()
				
				x = x.to(self.device)	
				label = label.to(self.device).float()
				
				logit = self.model(x)	# logit is(Bz)
				
				logit = logit.squeeze(-1)
				label = label.squeeze(-1)
				
				loss = self.criterion_for_regression(logit, label)	# It just outputs one PERCLOS number.
				
				loss.backward()
				
				if self.params.clip_value > 0:	
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
					
				self.optimizer.step()
				self.scheduler.step()
				
				losses.append(loss.item())	
				
			mean_loss = np.mean(losses)	
			
			save_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}.pth')
			torch.save(self.model.state_dict(), save_path)
				
			print('model saved @ ', save_path)
			#best_loss = mean_loss
				
			print(f'Epoch {epoch} model trained. Now commencing testing.')
			print(f'loss: {mean_loss}')
			
			with torch.no_grad():
				self.model.eval()
				
				evaluator = Evaluator(params=self.params, data_loader=self.data_loader['val'], model=self.model, epoch=epoch)
				corrcoef, r2, rmse, mean_loss = evaluator.get_metrics_for_regression()
				
				print(f'corrcoef: {corrcoef}, r2: {r2}, rmse: {rmse}, mean_loss: {mean_loss}')

def setup_seed(seed):
	torch.manual_seed(seed) 	
	torch.cuda.manual_seed_all(seed)	
	np.random.seed(seed)	
	random.seed(seed)
	torch.backends.cudnn.deterministic = True	
	torch.backends.cudnn.benchmark = False
def sorted_maps():
	ch_names = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", 
    "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6", 
    "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz", "O1", "O2"
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
	# It is ok to not have a masking parameter here. Because it is not like the model inherently masks. It is the pretraining script that is masking. Here, we don't have that if block, so nothing to worry about.
	parser = argparse.ArgumentParser(description='Finetuning EEG FM')
	
	parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
	parser.add_argument('--parallel', action='store_false', default=True, help='want gpu?')
	
	parser.add_argument('--multi_lr', action='store_false', default=True, help='variable learning rates for big model and small layer')
	parser.add_argument('--lr', type=float, default=5e-6, help='lr')
	
	parser.add_argument('--weight_decay', type=float, default=5e-2, help='wt dikey')
	parser.add_argument('--epochs', type=int, default=50, help='num_epochs')
	
	parser.add_argument('--clip_value', type=float, default=1, help='clip value')
	parser.add_argument('--model_dir', type=str, default='~/simpletmp/saved_fm', help='weight storage')
	
	parser.add_argument('--dataset_dir', type=str, default='~/CantusCerebra/data/TUAB/edf/process_refine', help='your 3 json files folder')
	parser.add_argument('--seed', type=int, default=42, help='seed')
	
	parser.add_argument('--cuda', type=int, default=0, help='What is the primary gpu?')
	parser.add_argument('--avail_gpus', type=str, default='0', help='Provide the explicit numbers a/c the motherboard of available GPUs.')
	
	parser.add_argument('--frozen', action='store_true', help='Pretrained big model will be frozen. Only FFN will be trained.')
	parser.add_argument('--batch_size', type=int, default=64, help='Bz')
	
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
	parser.add_argument('--mode', type=str, default='bin')
	
	parser.add_argument('--nc', type=int, default=1)
	
	params = parser.parse_args()
	setup_seed(params.seed)	
	
	params.model_dir = os.path.expanduser(params.model_dir)
	params.dataset_dir = os.path.expanduser(params.dataset_dir)
	#params.state_dict_path = os.path.expanduser(params.state_dict_path)
	
	dataset = LoadDataset(params)
	data_loader = dataset.get_data_loader()
	
	sorted_map = sorted_maps().to(f'cuda:{params.cuda}' if torch.cuda.is_available() else 'cpu')
	random_map = torch.randint_like(sorted_map, high=22)
	
	model = ConfiguredModel(model=Final(random_map, in_dim=params.in_dim, out_dim=params.out_dim, d_model=params.d_model, num_layers=params.num_layers, convolution_set=ast.literal_eval(params.convolution_set), stride=params.stride, dropout=params.dropout, d_ffn=params.d_ffn, num_heads=params.num_heads), params=params)
	
	trainer = FineTune_Trainer(params=params, model=model, data_loader=data_loader)	
	
	if params.mode == 'bin':
		trainer.train_for_binaryclass()
	if params.mode == 'mul':
		trainer.train_for_multiclass()
	if params.mode == 'reg':
		trainer.train_for_regression()
	
if __name__== '__main__':
	main()
