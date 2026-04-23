import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random
from model.TmpEncoder_stage5_clean import *
import os
import mne
from datasets.siena_dataset import LoadDataset as LoadDataset_siena
from datasets.tuab_dataset import LoadDataset as LoadDataset_tuab
from model.TmpEncoder_stage5_clean import Final
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
import ast

hcp_positions_path = os.path.expanduser('~/CantusCerebra/data/HCP/positions_100_7.txt')
connectivity_path = os.path.expanduser('~/CantusCerebra/processed_data/connectivity_matrix.txt')

class Evaluator:
	def __init__(self, params, data_loader, model, epoch):
	
		self.params = params
		self.data_loader = data_loader
		
		self.model = model
		self.model = self.model.to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
		
		finetune_weights_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}.pth')	
		
		map_location = torch.device(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')	
		finetune_state_dict = torch.load(finetune_weights_path, map_location=map_location)	
		
		model_state_dict = self.model.state_dict()	
		new_state_dict = {k.replace('module.', ''):v for k, v in finetune_state_dict.items()}
		
		matching_state_dict = {k:v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
		
		model_state_dict.update(matching_state_dict)
		self.model.load_state_dict(model_state_dict)
		
		#weight = torch.tensor([10.0]).to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
		self.criterion = nn.BCEWithLogitsLoss(reduction='mean').to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
			
		
	def BinaryClassEval(self):
		all_preds = []
		all_targets = []
		all_probs = []
		
		losses = []
		
		for x, label in tqdm(self.data_loader, mininterval=10):
			x = x.to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
			label = label.to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
			
			logit = self.model(x)
			loss = self.criterion(logit, label.float())	
			
			prob = torch.sigmoid(logit)
			pred = torch.gt(prob, 0.5)
			
			losses.append(loss.item())
			all_targets.append(label.detach().cpu().numpy())
			
			all_probs.append(prob.detach().cpu().numpy())
			all_preds.append(pred.detach().cpu().numpy())	# Added .detach to be consistent !
			
		mean_loss = np.mean(np.array(losses))
		
		final_targets = np.concatenate(all_targets).flatten()
		final_preds = np.concatenate(all_preds).flatten()
		final_probs = np.concatenate(all_probs).flatten()
		
		balanced_accuracy = balanced_accuracy_score(final_targets, final_preds)
		cohen_kappa = cohen_kappa_score(final_targets, final_preds)
		AUROC = roc_auc_score(final_targets, final_probs)
		
		print(f"Unique Predictions: {np.unique(final_preds)}")
		print(f"Average Probability: {np.mean(final_probs):.4f}")
		
		return balanced_accuracy, cohen_kappa, AUROC, mean_loss
	
class ConfiguredModel_siena(nn.Module):
	def __init__(self, model, params):
		super().__init__()
		
		self.backbone = model	
		self.params = params
		
		if self.params.use_pretrained_weights:
			map_location = torch.device(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')	
			
			state_dict = torch.load(self.params.state_dict_path, map_location=map_location)
			new_state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
				
			model_state_dict = self.backbone.state_dict()
			matching_state_dict = {k:v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}	
			
			model_state_dict.update(matching_state_dict)		
			model.load_state_dict(model_state_dict)	
			
		self.FFN = nn.Sequential(
						nn.Linear(200, 100),
						nn.ELU(),
						nn.Dropout(params.dropout),
						nn.Linear(100, 1),
					)
						
	def forward(self, x):
		Bz, num_chans, num_patches, patch_size = x.shape
		
		emb = self.backbone(x)
		emb = emb.mean(dim=(1, 2))
		
		out = self.FFN(emb)	
		out = out.reshape(Bz)
		
		return out

class FineTune_Trainer(object):	
	def __init__(self, params, data_loader, model):
		super().__init__()
		
		self.params = params
		self.model = model
				
		self.model = self.model.to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
		
		if self.params.parallel:
			device_ids = [int(i) for i in self.params.avail_gpus.split(' ')]
			self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
			
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
				
		if self.params.multi_lr:
			self.optimizer = torch.optim.AdamW([{'params':backbone_parameters, 'lr':self.params.lr}, {'params':other_parameters, 'lr':self.params.lr * 100}], weight_decay=self.params.weight_decay)
			
		else:
			self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
			
		self.data_loader = data_loader
		self.length = len(data_loader['train'])
		
		#weight = torch.tensor([10.0]).to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
		self.criterion = nn.BCEWithLogitsLoss(reduction='mean').to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')	# BCEWithLogitsLoss because of the asymmetry in the data labels !
		
		warmup_steps = (self.params.epochs * 0.4) // 10 * self.length
		total_steps = self.params.epochs * self.length
		main_steps = total_steps - warmup_steps
		
		self.optimizer_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=warmup_steps, start_factor=0.1, end_factor=1)
		self.optimizer_scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = main_steps, eta_min=1e-7)
		self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.optimizer_scheduler_warmup, self.optimizer_scheduler_main], milestones=[warmup_steps])
		self.scaler = torch.amp.GradScaler('cuda')
	
	def train(self):
	
		best_loss = float('inf')
		
		for epoch in range(self.params.epochs):
			print(f'Epoch {epoch} starts')
			
			losses = []
			self.model.train()	
			
			for x, label in tqdm(self.data_loader['train'], mininterval=10):	
			
				self.optimizer.zero_grad()
				
				x = x.to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')	
				label = label.to(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
				psd_x = STFT
				with torch.amp.autocast('cuda'):
					logit = self.model(x, psd_x)
					self.criterion.to(logit.device)	
					loss = self.criterion(logit, label.to(logit.device).float())
					
					prob = torch.sigmoid(logit)
					pred = torch.gt(prob, 0.5)
				
				self.scaler.scale(loss).backward()
				
				if self.params.clip_value > 0:	
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
					
				self.scaler.step(self.optimizer)
				self.scaler.update()
				self.scheduler.step()
				
				losses.append(loss.item())	
				
			mean_loss = np.mean(losses)	
			
			save_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}.pth')
			torch.save(self.model.state_dict(), save_path)
				
			print('model saved @ ', save_path)
			best_loss = mean_loss
				
			print(f'Epoch {epoch} model trained. Now commencing testing.')
			print(f'loss: {mean_loss}')
			
			with torch.no_grad():
				self.model.eval()
				
				evaluator = Evaluator(params=self.params, data_loader=self.data_loader['val'], model=self.model, epoch=epoch)
				b_acc, cohen, auroc, mean_loss = evaluator.BinaryClassEval()
				
				print(f'b_acc: {b_acc}, cohen: {cohen}, auroc: {auroc}, mean_loss: {mean_loss}')
			with open(os.path.expanduser('~/CantusCerebra/logs/finetune_metrics.txt'), 'a') as log:
				log.write(f'{b_acc}, {cohen}, {auroc}, {mean_loss}\n')


def setup_seed(seed):
	torch.manual_seed(seed) 	
	torch.cuda.manual_seed_all(seed)	
	np.random.seed(seed)	
	random.seed(seed)
	torch.backends.cudnn.deterministic = True	
	torch.backends.cudnn.benchmark = False
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
	# It is ok to not have a masking parameter here. Because it is not like the model inherently masks. It is the pretraining script that is masking. Here, we don't have that if block, so nothing to worry about.
	parser = argparse.ArgumentParser(description='Finetuning EEG FM')
	parser.add_argument('--use_pretrained_weights', action='store_false', default=True, help='or scratch?')
	parser.add_argument('--state_dict_path', type=str, default='~/CantusCerebra/saved_fm/pretrain_weights.pth', help='state_dict_path')
	parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
	parser.add_argument('--parallel', action='store_false', default=True, help='want gpu?')
	parser.add_argument('--multi_lr', action='store_false', default=True, help='variable learning rates for big model and small layer')
	parser.add_argument('--lr', type=float, default=5e-6, help='lr')
	parser.add_argument('--weight_decay', type=float, default=5e-2, help='wt dikey')
	parser.add_argument('--epochs', type=int, default=50, help='num_epochs')
	parser.add_argument('--clip_value', type=float, default=1, help='clip value')
	parser.add_argument('--model_dir', type=str, default='~/CantusCerebra/saved_fm', help='weight storage')
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
	parser.add_argument('--nheads', type=int, default=8, help='Number of Heads in MHSA')
	parser.add_argument('--convolution_set', type=str, default="[(1,), (3,), (5,)]", help='Concentrated convolution sizes. < num_chans')
	parser.add_argument('--seq_len', type=int, default=30, help='num_patches')
	parser.add_argument('--is_causal', action='store_true', help='If you want causal Temporal Attention')
	parser.add_argument('--need_key_padding', action='store_true', help='if any padding that could be added is to be ignored')
	parser.add_argument('--stride', type=int, default=1, help='stride for temp convs')
	
	params = parser.parse_args()
	setup_seed(params.seed)	
	
	params.model_dir = os.path.expanduser(params.model_dir)
	params.dataset_dir = os.path.expanduser(params.dataset_dir)
	params.state_dict_path = os.path.expanduser(params.state_dict_path)
	
	tuab_dataset = LoadDataset_tuab(params)
	data_loader = tuab_dataset.get_data_loader()
	
	sorted_map = sorted_maps().to(f'cuda:{params.cuda}' if torch.cuda.is_available() else 'cpu')
	
	model = ConfiguredModel_siena(model=Final(sorted_map, in_dim=params.in_dim, out_dim=params.out_dim, d_model=params.d_model, num_layers=params.num_layers, convolution_set=ast.literal_eval(params.convolution_set), stride=params.stride, dropout=params.dropout, d_ffn=params.d_ffn, nheads=params.nheads), params=params)
	
	trainer = FineTune_Trainer(params=params, model=model, data_loader=data_loader)	
	trainer.train()
	
if __name__== '__main__':
	main()
