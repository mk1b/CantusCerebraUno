import torch
import torch.nn as nn	
import torch.nn.functional as F
import math

def backward_hook(self, module, inputs, output):
	print(f"\n--- Backward Hook Triggered on {module.__class__.__name__} ---")
	if grad_input[0] is not None:
		print(f"Gradient of Loss w.r.t Module Input: \n{grad_input[0]}")
	if grad_output[0] is not None:
		print(f"Gradient of Loss w.r.t Module Output: \n{grad_output[0]}")
	
class _ff_block_spc(nn.Module):
	def __init__(self, d_model=200, d_ffn=800):
		super().__init__()
		
		self.FFN = nn.Sequential(
						nn.Linear(d_model, d_ffn),
						nn.GELU(),
						nn.Dropout(0.1),
						nn.Linear(d_ffn, d_model),
					)
	def forward(self, x):
		out = self.FFN(x)
		
		return out

class SimpleEmbeddingSpc(nn.Module):
	def __init__(self, in_dim=200):
		super().__init__()
		
		self.extract_features = nn.Sequential(
					nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
					nn.GELU(),
					nn.GroupNorm(5, 25),
					
					nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
					nn.GELU(),
					nn.GroupNorm(5, 25),
					
					nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
					nn.GELU(),
					nn.GroupNorm(5, 25)		
				)
		
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		
		x = x.reshape(Bz, 1, num_chans * num_bands * num_patch, patch_size)	
		out = self.extract_features(x)

		out = out.permute(0, 2, 1, 3).contiguous().reshape(Bz, num_chans * num_bands * num_patch, 1, -1)

		out = out.reshape(Bz, num_chans, num_bands, num_patch, patch_size)	
		
		return out

class InterBandConv(nn.Module):
	def __init__(self, d_model=200, kernel_size=3):
		super().__init__()
		
		self.conv = nn.Sequential(nn.Conv2d(in_channels=d_model, out_channels=d_model, stride=(1, 1), kernel_size=(kernel_size, 1), padding=((kernel_size - 1) // 2, 0), groups=d_model))
	
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		
		x = x.permute(0, 1, 3, 2, 4)
		x = x.reshape(Bz * num_chans, 1, num_patch, num_bands, patch_size)
		
		x = x.reshape(Bz * num_chans * num_patch, 1, num_bands, patch_size)
		x = x.permute(0, 3, 2, 1)
		
		out = self.conv(x)
		out = out.permute(0, 3, 1, 2)
		
		out = out.reshape(Bz, num_chans, num_patch, patch_size, num_bands)
		
		out = out.permute(0, 1, 4, 2, 3)
		
		return out
		
class InterChannelConvACPE(nn.Module):
	def __init__(self, d_model=200):
		super().__init__()
		self.conv = nn.Sequential(nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), padding=(9, 3), stride=(1, 1), groups=d_model))	
		
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		x = x.permute(0, 2, 1, 3, 4)
		
		x = x.reshape(Bz * num_bands, num_chans, num_patch, patch_size)
		x = x.permute(0, 3, 1, 2)	
		
		out = self.conv(x)
		out = out.reshape(Bz, num_bands, patch_size, num_chans, num_patch)
		
		out = out.permute(0, 3, 1, 4, 2)
		
		return out

class TemEmbedSpcLayer(nn.Module):
	def __init__(self, convolution_set=[(1,), (3,), (5,)], d_model=200, stride=1):
		super().__init__()
		
		self.convolution_set = convolution_set
		
		dims = [d_model // 2 ** i for i in range(1, len(self.convolution_set))]	
		self.dim_scales = [*dims, d_model - sum(dims)]
		
		self.embed_layers = nn.ModuleList([nn.Conv2d(in_channels=d_model, out_channels=dim_scale, stride=(stride, 1), kernel_size=(kernel_size, 1), padding=((kernel_size - 1) // 2, 0))
					for (kernel_size, ), dim_scale in zip(self.convolution_set, self.dim_scales)
				])
		
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		x = x.reshape(Bz * num_chans, num_bands, num_patch, patch_size)	
		
		x = x.reshape(Bz * num_chans * num_bands, 1, num_patch, patch_size)
		x = x.permute(0, 3, 2, 1)
		
		f_maps = [conv(x) for conv in self.embed_layers]
		out = torch.cat(f_maps, dim=1)
		
		out = out.permute(0, 3, 2, 1)
		out = out.reshape(Bz * num_chans, num_bands, num_patch, patch_size)
		
		out = out.reshape(Bz, num_chans, num_bands, num_patch, patch_size)
		
		return out

class BandEmbedSpcLayer(nn.Module):
	def __init__(self, sorted_map, convolution_set=[(1,), (3,), (5,)], d_model=200):
		super().__init__()
		
		self.register_buffer('sorted_map', sorted_map)
		self.convolution_set = convolution_set
		
		max_size = convolution_set[-1][0]
		num_chans = sorted_map.shape[0]
		
		W_gather = torch.zeros(num_chans, num_chans * max_size, device=sorted_map.device)
		row_idx = sorted_map[:, :max_size].contiguous().view(-1)
		col_idx = torch.arange(num_chans * max_size, device=sorted_map.device)
		W_gather[row_idx, col_idx] = 1.0
		self.register_buffer('W_gather', W_gather)
		
		dim_scales = [d_model // 2 ** i for i in range(1, len(convolution_set))]
		dim_scales = [*dim_scales, d_model - sum(dim_scales)]
		
		mix_features = [nn.Conv2d(in_channels=d_model, out_channels=dim_scale, stride=(1, 1), kernel_size=(conv_set, 1), padding=((conv_set - 1) // 2, 0)) 
							for dim_scale, (conv_set,) in zip(dim_scales, convolution_set) if conv_set > 1
					]	
								
		if convolution_set[0][0] == 1:
			mix_features = [nn.Conv2d(in_channels=d_model, out_channels=dim_scales[0], stride=(1, 1), kernel_size=(1, 1)), *mix_features]
			
		self.mix_features = nn.ModuleList(mix_features)
		
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		
		max_size = self.convolution_set[-1][0]
		x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
		N = Bz * num_bands * num_patch
		x_flat = x_perm.reshape(N, patch_size, num_chans)
		
		gathered = torch.matmul(x_flat, self.W_gather) 
		
		gathered = gathered.reshape(Bz, num_bands, num_patch, patch_size, num_chans, max_size)
		sorted_x = gathered.permute(0, 4, 1, 2, 3, 5)
		
		sorted_x = sorted_x.reshape(Bz * num_chans * num_bands * num_patch, patch_size, max_size, 1)
		
		f_maps = [conv(sorted_x) for conv in self.mix_features]
		
		final = torch.cat(f_maps, dim=1)
		
		out = final[:, :, 0, 0]
		out = out.reshape(Bz, num_chans, num_bands, num_patch, patch_size)
		
		return out
		
class TemporalAttentionSpc(nn.Module):
	def __init__(self, convolution_set=[(1,), (3,), (5,)], d_model=200, num_heads=8, dropout=0.1, batch_first=True):
		super().__init__()
		
		self.convolution_set = convolution_set
		self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
		
		self.hook_handle = self.attn.register_full_backward_hook(self.backward_hook)
	
	def backward_hook(self, module, grad_input, grad_output):
		print(f"\n--- Backward Hook Triggered on {module.__class__.__name__} ---")
		if grad_input[0] is not None:
			print(f"Gradient of Loss w.r.t Module Input: \n{grad_input[0]}")
		if grad_output[0] is not None:
			print(f"Gradient of Loss w.r.t Module Output: \n{grad_output[0]}")
	
	def forward(self, x, is_causal=False, need_key_padding=False):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		
		window_size = min(num_patch, self.convolution_set[-1][0])
		original_num_patch = num_patch	
		
		padded_x = x
		padding_size = 0
			
		if num_patch % window_size != 0:
			padding_size = window_size - num_patch % window_size	
			
			padded_x = F.pad(x, (0, 0, 0, padding_size))	
			num_patch = num_patch + padding_size	
			
		num_windows = num_patch // window_size
		
		key_padding_mask = None
		attn_mask = None
		
		if need_key_padding:
			mask_list = [False] * original_num_patch + [True] * padding_size
			key_padding_mask = torch.tensor(mask_list, dtype=torch.bool, device=x.device)
			
			key_padding_mask = key_padding_mask.reshape(1, 1, 1, -1).expand(Bz, num_chans, num_bands, num_patch)
			
			key_padding_mask = key_padding_mask.reshape(Bz, num_chans, num_bands, num_windows, window_size)	
			key_padding_mask = key_padding_mask.permute(0, 1, 2, 4, 3)
			
			key_padding_mask = key_padding_mask.reshape(Bz * num_chans * num_bands * window_size, num_windows)	
			
		if is_causal:
			attn_mask = torch.triu(torch.ones(num_windows, num_windows) * float('-inf'), diagonal=1)
			
			attn_mask = attn_mask.to(x.dtype)
			attn_mask = attn_mask.to(x.device)
			
		padded_x = padded_x.reshape(Bz, num_chans, num_bands, num_windows, window_size, patch_size)
		padded_x = padded_x.permute(0, 1, 2, 4, 3, 5)
		
		padded_x = padded_x.reshape(Bz * num_chans * num_bands * window_size, num_windows, patch_size)
		attn_out = self.attn(padded_x, padded_x, padded_x, need_weights=False)[0]
		#, key_padding_mask=key_padding_mask, attn_mask=attn_mask, was removed for ultra-fast attention.
		
		attn_out = attn_out.reshape(Bz, num_chans, num_bands, window_size, num_windows, patch_size)
		attn_out = attn_out.permute(0, 1, 2, 4, 3, 5)
		
		attn_out = attn_out.reshape(Bz, num_chans, num_bands, num_windows * window_size, patch_size)
		out = attn_out[:, :, :, :original_num_patch, :]	
		
		return out	

class InterBandAttention(nn.Module):
	def __init__(self, d_model=200, num_heads=8, dropout=0.1, batch_first=True):
		super().__init__()
		self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
		
		self.hook_handle = self.attn.register_full_backward_hook(self.backward_hook)
	
	def backward_hook(self, module, grad_input, grad_output):
		print(f"\n--- Backward Hook Triggered on {module.__class__.__name__} ---")
		if grad_input[0] is not None:
			print(f"Gradient of Loss w.r.t Module Input: \n{grad_input[0]}")
		if grad_output[0] is not None:
			print(f"Gradient of Loss w.r.t Module Output: \n{grad_output[0]}")
	
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		x = x.permute(0, 1, 3, 2, 4)
		x = x.reshape(Bz * num_chans * num_patch, num_bands, patch_size)
		out = self.attn(x, x, x, need_weights=False)[0]
		out = out.reshape(Bz, num_chans, num_patch, num_bands, patch_size)
		out = out.permute(0, 1, 3, 2, 4)
		return out
		
class InterChannelAttention(nn.Module):
	def __init__(self, sorted_map, d_model=200, dropout=0.1, num_heads=8, batch_first=True):
		super().__init__()
		self.register_buffer('sorted_map', sorted_map)
		self.attn = nn.MultiheadAttention(embed_dim=d_model, dropout=dropout, num_heads=num_heads, batch_first=batch_first)
		self.group_transform = nn.Linear(d_model, d_model)
		
		self._precompute_matrices()
		
		self.hook_handle = self.attn.register_full_backward_hook(self.backward_hook)
	
	def backward_hook(self, module, grad_input, grad_output):
		print(f"\n--- Backward Hook Triggered on {module.__class__.__name__} ---")
		if grad_input[0] is not None:
			print(f"Gradient of Loss w.r.t Module Input: \n{grad_input[0]}")
		if grad_output[0] is not None:
			print(f"Gradient of Loss w.r.t Module Output: \n{grad_output[0]}")
	
	def _precompute_matrices(self):
		num_chans = self.sorted_map.shape[0]
		K = self.sorted_map.shape[1] 
		num_groups = K // 4
   
		col_idx_mean = self.sorted_map[:, :num_groups * 4].reshape(-1, 4).long()	 
		col_idx_first = self.sorted_map[:, 0:num_groups * 4:4].reshape(-1, 1).long()
		
		W_mean = torch.zeros(num_chans * num_groups, num_chans, device=self.sorted_map.device)
		W_first = torch.zeros(num_chans * num_groups, num_chans, device=self.sorted_map.device)

		W_mean.scatter_(dim=1, index=col_idx_mean, value=0.25)

		W_first.scatter_(dim=1, index=col_idx_first, value=1.0)
		
		self.register_buffer('W_mean', W_mean.T)   
		self.register_buffer('W_first', W_first.T) 
		
		self.num_chans = num_chans
		self.num_groups = num_groups
		
	def forward(self, x):
		Bz, num_chans, num_bands, num_patch, patch_size = x.shape
		x_perm = x.permute(0, 2, 3, 4, 1)
 
		N = Bz * num_bands * num_patch
		x_flat = x_perm.reshape(N, patch_size, num_chans)
		
		mean_groups = torch.matmul(x_flat, self.W_mean)
		first_tensors = torch.matmul(x_flat, self.W_first)
		
		mean_groups = mean_groups.transpose(1, 2)
		first_tensors = first_tensors.transpose(1, 2)

		final_stacked = first_tensors + self.group_transform(mean_groups)
		final_stacked = final_stacked.reshape(N, num_chans, self.num_groups, patch_size)

		final_stacked = final_stacked.reshape(N * num_chans, self.num_groups, patch_size)
		
		query = final_stacked[:, 0:1, :] 
		key_value = final_stacked
		
		out = self.attn(query, key_value, key_value, need_weights=False)[0] 
		
		out = out.squeeze(1) 
		out = out.reshape(Bz, num_bands, num_patch, num_chans, patch_size)
		out = out.permute(0, 3, 1, 2, 4)
		
		return out
		
class EmbeddingStepSpc(nn.Module):	
	def __init__(self, in_dim=200, d_model=200, kernel_size=3):
		super().__init__()
		self.SimpleEmbedding = SimpleEmbeddingSpc(in_dim=in_dim)
		self.InterBandConv = InterBandConv(d_model=d_model, kernel_size=kernel_size)
		self.InterChannelConvACPE = InterChannelConvACPE(d_model=d_model)
		
	def forward(self, x):
		embed_x = self.SimpleEmbedding(x)
		ibc_x = self.InterBandConv(embed_x)
		icc_x = self.InterChannelConvACPE(ibc_x)
		return icc_x
		
		
class SpcEncoderLayer(nn.Module):
	def __init__(self, sorted_map, in_dim=200, d_model=200, kernel_size=3, convolution_set=[(1,), (3,), (5,)], dropout=0.1, num_heads=8, d_ffn=800):
		super().__init__()
		#self.EmbeddingStep = EmbeddingStep(in_dim=in_dim, d_model=d_model, kernel_size=kernel_size)
		self.TemEmbedSpcLayer = TemEmbedSpcLayer(convolution_set=convolution_set, d_model=d_model, stride=1)
		self.BandEmbedSpcLayer = BandEmbedSpcLayer(sorted_map, convolution_set=convolution_set, d_model=d_model)
		self.TemporalAttentionSpc = TemporalAttentionSpc(convolution_set=convolution_set, d_model=d_model, num_heads=num_heads, dropout=dropout)
		self.InterBandAttention = InterBandAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
		self.InterChannelAttention = InterChannelAttention(sorted_map, d_model=d_model, dropout=dropout, num_heads=num_heads)
		
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.norm4 = nn.LayerNorm(d_model)
		
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)
		self.dropout4 = nn.Dropout(dropout)
		
		self._ff_block_spc = _ff_block_spc(d_model=d_model, d_ffn=d_ffn)
			
	def forward(self, x):	
		tmp_x = x + self.TemEmbedSpcLayer(x)
		bnd_x = tmp_x + self.BandEmbedSpcLayer(tmp_x)
		
		tmp_attn_x = bnd_x + self.dropout1(self.TemporalAttentionSpc(self.norm1(bnd_x)))
		bnd_attn_x = tmp_attn_x + self.dropout2(self.InterBandAttention(self.norm2(tmp_attn_x)))
		
		chn_attn_x = bnd_attn_x + self.dropout3(self.InterChannelAttention(self.norm3(bnd_attn_x)))
		out = chn_attn_x + self.dropout4(self._ff_block_spc(self.norm4(chn_attn_x)))
		
		return out

'''
	You can edit this copy of code freely because it is independent and doesn't affect prior work.
	
	So we are aiming to remove the phase entirely. Instead, just focus on the band power dynamics and that attention. The combination is to be mandatorily done via cross-attention. That is the only thing that facilitates good learning
	between such different features. 
	There is also one more thing here. Do you want to apply attention in a bidirectional manner? I am inclined to answer NO. This is because the spectral features are to inform the temporal features.
	You know, that this is where the bottleneck arises. In unified architectures, you see that the models learn the embeddings by simultaneuously attending to both temporal and spectral. But here, one attends to only temporal and the
	other attends to only spectral. And then you merge them. It is always better if you were consistent from the start rather than pick it up in the middle. You can work on that on the side, but do not jeopardize what has already been built. We will have to think about the loss function as well. Do we want a more relaxed loss function or MSE is fine? I think that as long as you can predict things using only the SpcEncoder, then, only MSE should be fine. Our main goal here is to only predict things using Spc. If that is achieved, then our model will have a 80% chance to work.
	Now, it is time to answer some questions.
	
	We must keep torch.abs() because that signifies the PSD, not the simple raw amplitude; Also if you ever compare the reconstruct and the truth, they must be on the same scale.
	As a matter of fact, if you are going to compare anything, they must be on the same scale and appropriate preprocessing must be applied to ensure this. Have good normalization and residuals so as to never cause instability in training.
	1) We have removed the phases. Now what kind of learning are you planning to implement here?
	- We would have to remove custom attention and apply criss-cross attention for the bands. 
	
	2) How are you going to implement masking here?
	- We have discussed this:
	Okay, so there are pretty drastic changes here. We are going to no longer mask the spc encoder.
	----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	So as per the discussed architecture, we are going to adopt the philosophy that Spc exists to assist the temporal encoder. This is because of two reasons:
	1) The monolothic design does not work. This is because we have seen that the phase is ill-conditioned and we are going to remove it. Also, the merging is really bad. Two layers of transformer may be enough to mix everything,
	but monolithic architectures don't work due to the embeddings NOT being conditioned while they were being formed. This leads to weak gradient flow and causes convergence to take a hit because the gradients don't really flow that
	effectively. Like I am saying that it is the decoder that gets tuned more than the attention mechanisms because the spectral attention is going to look at things that is going to improve temporal domain when it has never ever seen
	the temporal domain ! It is LITERALLY confused about where to look. Add the ill-conditioned phase on top and you got yourself a divergent model. I think that just adding attention with NO temporal input is the main culprit. It is not
	the fault of the optimizer that the model did not converge.
	2) The phase is ill-conditioned. So we remove it. Fine. But then what about the PSDs? Well, since there is little meaning of PSDs + time without phase if we are going to extract coupling, we abandon the coupling idea and use PSD as a glorified FFT. Now there are a number of advantages of this glorified FFT. One is that it comes with the time series and the other is that we can split it into bands. We are going to leverage both of them in the following manner:
	a) First, we will use THIS band splitting as a neurological prior. We know that the split of the bands is meaningful. Okay. That much even FFT can do. The other thing is that we can literally apply attention and convs to capture the dynamics. Okay, fine. Then? THEN, we are going to make sure that our reconstructed signal actually CAN model the band power dynamics? 
	Now. How are you going to manage that without the phase ?! Well, that's where the temporal encoder and the MSE loss comes in. We are not going to have just one MSE loss, but we are going to have SIX MSE losses. One for temporal and the other five for the five different bands. This ensures that the temporal and the interactions are captured.
	You can really think of the problem in this way. What do we want our signal to be? Well, we'd obviously like our model to fit the general trend of the signal and understand the nuances. Well, Okay. Since brain is a complicated machine, you would think that if you could capture the relations between different parts of the nodes, it'd be helpful. True, that's where functional connectivity comes in. But, functional connectivity also needs to be supplemented by spectral functions as well ! If something is connected, it is connected electrically and spectrally. So, to take the load off of temporal encoder(which is by the way, the trend. More load off and more efficient, means more accurate) where load is basically understanding the general trend of the signal(of course through correlations), the temporal encoder could better focus on the sharp features which it excels at.
	Thus, to supplement this, we add more losses to force the model to obey and look at the spectral dynamics as well. But the temporal encoder will get stretched. It CANNOT look at both fine and general trends AT THE SAME TIME. It is an information bottleneck !!! Therefore, another module needs to come which aids the temporal encoder to easily look and find the spectral dynamics and correlate the electrical domain signals to the normalized easy spectral features.
	Now, to do this, I have devised a strategy(okay, its a misunderstanding of the wavelet2vec architecture). To tackle:
	1) The information bottleneck if we add losses
	2) Adding band power dynamics
	3) Aid functional connectivity not in just the temporal domain, but also spectral
	4) Alleviate the masking strategy search by just not masking
	5) To handle the loss of phase problem
	6) Monolithic architecture
	We need to interleave the temporal and the spectral encoders. They aren't really encoders, rather its a unified architecture which uses many attentions in one layer. Instead of synchronizing at the very last, they synchronize
	every layer. This ensures that they mix well, and since no phase, this loss is more likely to converge. To further strengthen the links between temporal and spectral, we add the 5 losses and don't mask the wavelet coefficients. This completely alleviates the cos, sine problems as well and fits right into the standard attention. We merge them using cross attention.
	Its solved. The architecture is mathematically and logically sound. All that's remaining is the implementation. Go have some fun now !
'''
