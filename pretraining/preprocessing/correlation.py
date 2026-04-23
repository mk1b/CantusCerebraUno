import torch
import numpy as np
import os
import sys
import pandas as pd

base_path = os.path.expanduser('~/CantusCerebra/data/HCP/HCPYoungAdult'	)
output_dir = os.path.expanduser('~/CantusCerebra/processed_data/connectivity_matrix/connectivity_matrix.txt')
atlas_name = 'Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T'

# We will have to absolutely get the files first.
def partial_correlations(matrix):
	inverse_matrix = np.linalg.inv(matrix)
	diag = np.diag(inverse_matrix)
	
	mult_correlation = np.outer(diag, diag)
	sqrt_correlation = np.sqrt(mult_correlation)
	
	partial_corr = -1.0 * inverse_matrix / sqrt_correlation
	
	np.fill_diagonal(partial_corr, 1)
	
	return partial_corr
	

def average_connectivity(base_path, epsilon=1e-9):
	raw_matrices = []
	
	# Load the matrices
	for subject_id in os.listdir(base_path):
		subject_path = os.path.join(base_path, subject_id)
		
		for session in os.listdir(subject_path):
			session_path = os.path.join(subject_path, session)
			
			final_folder = os.path.join(session_path, atlas_name)
			
			for file_name in os.listdir(final_folder):	
				if file_name.endswith('.txt'):
					final_file = os.path.join(final_folder, file_name)
					
					try:
						data = np.loadtxt(final_file)
					except Exception as e:
						print(f'File {final_file} failed to load with error {e}.')
						continue
						
					raw_matrices.append(data)
		if len(raw_matrices) % 30 == 0:
			print(f'Loaded {len(raw_matrices)} subjects.')	
						
	# Now we convert the raw_matrices into Z-matrices due to stats.
	raw_matrices = np.array(raw_matrices)
	clipped_matrices = np.clip(raw_matrices, -1 + epsilon, 1 - epsilon)
	
	z_matrices = np.arctanh(clipped_matrices)
	
	avg_z_matrix = np.mean(z_matrices, axis=0)
	avg_correlation_matrix = np.tanh(avg_z_matrix)
	
	print('Files loaded successfully, Matrix computed.')
	return True, avg_correlation_matrix
	
def process_connectivity(base_path, output_dir):
	loaded, matrix = average_connectivity(base_path)
	
	if not loaded:
		sys.exit('Failed to load connectivity matrices.')
	
	matrix_final = partial_correlations(matrix)
	
	print('Connectivity matrices loaded.')
	with open(output_dir, 'w') as f:
		np.savetxt(f, matrix_final, delimiter = '	')
	return matrix_final


final_output = process_connectivity(base_path, output_dir)
print('Final shape: ', final_output.shape)	
	
		
	
	
	
	
	
			
			
		
		
			
		
