import torch
import os

state_dict_path = os.path.expanduser('/home/mk1b/CantusCerebra/saved_fm/pretrain_weights.pth')
state_dict = torch.load(state_dict_path, map_location='cpu')

sum = 0
for key, value in state_dict.items():
	sum += value.numel()
	print(key, value.numel())

print(sum)
