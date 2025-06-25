import numpy as np
import pandas as pd
from numba import jit

def Xmatch(cat_test, cat_ref, pos_cols_test, maj_col_ref, threshold, pos_cols_ref=[], other_cols_test=[], other_cols_ref=[]):
	"""
	Performs a cross-match between two catalogs (cat_test and cat_ref) based on positional and other parameter constraints.
	This function works on "small" catalogs. It is fully vectorized thanks to the numpy function meshgrid, however,
	if any of the two catalogs have too many sources, you may not be able to perform a cross match due to the
	weight of huge float64 matrices.
	
	Safe and user friendly.
	
	This function return an 2D array where rows are matched sources, 1st column is the index in ref of the matched source,
	2nd is the index in test of the matched source, last column is the multiparameter error of the match.

	This function has been build to cross match sources within an single pointing. It may not work in general use.	
	Sky coordinates and major axis are expected all three to be consistants, others parameters are expected consistants between test and ref.
	
	Parameters:
		cat_test (pd.DataFrame): Test catalog to be matched.
		cat_ref (pd.DataFrame): Reference catalog to match against.
		pos_cols_test (list): Names of position columns in test catalog (e.g., ['x', 'y']).
		maj_col_ref (str): Column name in reference catalog where major axis is stored.
		threshold (float or array): Threshold for matching. If array, first value is for position, rest for other parameters.
		pos_cols_ref (list): Optional. Position column names in reference catalog (defaults to pos_cols_test).
		other_cols_test (list): Optional. Additional test catalog columns to match on.
		ther_cols_ref (list): Optional. Corresponding reference catalog columns.

	Returns:
		candidates (np.ndarray): Array of matches [ref_index, test_index, match_error].
	"""
        
	#If no positional or other column names are given for ref/test, use corresponding defaults
	if not pos_cols_ref:
		pos_cols_ref = pos_cols_test
			
	if not other_cols_ref:
		other_cols_ref = other_cols_test
	
	if not other_cols_test:
		other_cols_test = other_cols_ref
		
	
	#Ensure test and ref other_cols match in count
	if len(other_cols_test) != len(other_cols_ref):
		raise ValueError("Different number of columns used in test and ref")
		
	#Handle threshold: allow single float or array (for multiple parameters)
	if isinstance(threshold, (list, np.ndarray)):
		multiple_param_flag = True
		if len(threshold) != len(other_cols_test) + 1:
			raise ValueError("Must have one threshold for position and one threshold for each ellement in other_cols_test/other_cols_ref")
		threshold = np.asarray(threshold)
	else:
		#Warn if thresholds are underspecified for multiple matching dimensions
		print(type(threshold))
		if len(other_cols_test)>0 or len(other_cols_ref)>0:
			print("WARNING: Only 1 threshold specified")
			print("Convert single threshold into array.")
			nb_threshold = max(len(other_cols_test), len(other_cols_ref)) + 1
			threshold = np.ones(nb_threshold)*threshold
		multiple_param_flag = False
	
	
	
	
	N_test = cat_test.shape[0]
	N_ref = cat_ref.shape[0]
	
	
	#Placeholder for matched candidates [ref_index, test_index, total_error]
	candidates = np.empty((N_test, 3))*np.nan
	
	#Create 2D grids for vectorized distance calculations
	XX_test, XX_ref = np.meshgrid(cat_test[pos_cols_test[0]].to_numpy(), cat_ref[pos_cols_ref[0]].to_numpy())
	YY_test, YY_ref = np.meshgrid(cat_test[pos_cols_test[1]].to_numpy(), cat_ref[pos_cols_ref[1]].to_numpy())
	_, MAJ_mat = np.meshgrid(np.zeros(N_test), cat_ref[maj_col_ref].to_numpy())
	
	#Calculate normalized positional error (e.g., sky_distance/major axis size)
	position_error = np.sqrt((XX_test - XX_ref)**2 + (YY_test - YY_ref)**2)/MAJ_mat
	
	#Mask values beyond position threshold
	position_error[position_error > threshold[0]] = np.inf
	squared_multi_parameter_error = position_error**2
	
	#If using additional parameters, calculate normalized error for each and accumulate
	if multiple_param_flag:
		specific_errors = np.zeros((len(other_cols_test), N_ref, N_test))
		for k, col in enumerate(other_cols_test):
			if col in pos_cols_test:
				continue  #Avoid duplicating positional columns
			else:
				grid_test, grid_ref = np.meshgrid(cat_test[col].to_numpy(), cat_ref[other_cols_ref[k]].to_numpy())
				sp_err = np.abs(grid_test - grid_ref)/grid_ref
				sp_err[sp_err > threshold[k+1]] = np.inf
				
				specific_errors[k] = sp_err
				
	
		#Sum all squared errors to form total error matrix
		squared_multi_parameter_error += np.sum(specific_errors**2, axis=0)
		
		
	#Final error after combining position + parameter errors
	multi_parameter_error = np.sqrt(squared_multi_parameter_error)
	
	
	#Greedy matching: select smallest error pairs, avoid double matches
	ind_candidate = 0
	for n in range(min(N_test, N_ref)):
	
		i, j = np.unravel_index(multi_parameter_error.argmin(), multi_parameter_error.shape)
		
		if not(np.isinf(multi_parameter_error[i,j])):
			#Store match [ref_index, test_index, error]
			candidates[ind_candidate] = np.array([j, i, multi_parameter_error[i, j]])
			ind_candidate += 1
			#Remove these entries from future consideration
			multi_parameter_error[:, j] = np.inf
			multi_parameter_error[i, :] = np.inf
		
		else:
			break #No more valid matches
	
	
	
	candidates = candidates[:ind_candidate,:]
	
	return candidates
	
#===================================================================================================================================================

@jit(cache=True, nopython=True, parallel=True)
def fast_Xmatch(test, ref, thresholds):
	"""
	Performs a cross-match between two catalgos of sources (test and ref) based on positional and other parameter constraints.
	This function is faster than Xmatch, and works with larger catalogs.
	
	Faster and more versatile.
	
	This function returns a tuple of list, the first list is the indices of the matched test sources, the 2nd is the list of the matched refs
	the third and last list is the multiparameter error of the match.
	
	Parameters:
		test (np.ndarray): Test dataset of shape (N, M), with at least 3 columns [RA, Dec, param1, param2, ...].
		ref (np.ndarray): Reference dataset of shape (K, M), same format as test.
		thresholds (np.ndarray): Array of thresholds. First is for position, others for additional parameters.

	Returns:
		test_match_idx (list): Indices in test array that were matched.
		ref_match_idx (list): Corresponding indices in ref array for each match.
		score (list): Multiparameter error (combined position + attribute error) for each match.
	
	"""
	
	#Lists to store the matching results
	test_match_idx = []
	ref_match_idx = []
	score = []
	
	#Define loop limits
	#Either all the possible match has been made
	#or all sources has been tested.
	max_it = max(test.shape[0], ref.shape[0])	#Max number of iterations
	max_match = min(test.shape[0], ref.shape[0])	#Max number of possible matches
	
	it_count = 0
	match_count = 0
	
	#Allocate reusable matrix for attribute errors (excluding RA/Dec)
	specific_error_matrice = np.zeros((ref.shape[0], test.shape[1] - 2))
	
	#Iterative matching process
	while it_count <= max_it and match_count <= max_match:
	
		for i in range(test.shape[0]):
		
			#Skip if this test source already matched
			if i not in test_match_idx:
		
				source_test = test[i,:] #Current test source
				
				#Normalize position error by major axis size (assumed in ref[:,2])
				position_error = np.sqrt((source_test[0] - ref[:,0])**2 + (source_test[1] - ref[:,1])**2)/ref[:,2]
				
				#Compute sum of squared relative errors for other parameters
				squared_specific_error_sum = np.zeros((ref.shape[0]))
				for comp_col in range(2,test.shape[1]):

					specific_error_matrice[:,comp_col - 2] = np.abs(source_test[comp_col] - ref[:, comp_col])/ref[:, comp_col]
					squared_specific_error_sum += specific_error_matrice[:,comp_col - 2]**2
				
				#Final matching error (position + attribute)
				multi_parameter_error = np.sqrt(position_error**2 + squared_specific_error_sum)

				#Get indices of candidates sorted by lowest total error			
				sorting = np.argsort(multi_parameter_error)

				for id_candidate in sorting:
					
					#Skip if position error exceeds threshold
					if position_error[id_candidate] > thresholds[0]:
						continue
					else:
						#Check attribute thresholds
						exceeds = False
						for comp_col in range(0, test.shape[1] - 2):
							if specific_error_matrice[id_candidate,comp_col] > thresholds[comp_col + 1]:
								exceeds = True
								break
						if exceeds:
							continue


					#If candidate not already matched, assign it
					if id_candidate not in ref_match_idx:
					
						test_match_idx.append(i)
						ref_match_idx.append(id_candidate)
						score.append(multi_parameter_error[id_candidate])
						
						match_count += 1
						
						break
					
					#If candidate is already matched, allow match if new one has lower error
					else:
						ind_to_compare = ref_match_idx.index(id_candidate)
						
						if score[ind_to_compare] < multi_parameter_error[id_candidate]:
							
							test_match_idx.pop(ind_to_compare)
							ref_match_idx.pop(ind_to_compare)
							score.pop(ind_to_compare)
							
							
							test_match_idx.append(i)
							ref_match_idx.append(id_candidate)
							score.append(multi_parameter_error[id_candidate])
							
							#Retry current test source in next round
							it_count -= 1
								
							break
				
				it_count += 1


	return test_match_idx, ref_match_idx, score
		
#===================================================================================================================================================

@jit(cache=True, nopython=True, parallel=True)
def even_faster_Xmatch(test, ref, max_sep):
	"""
	PPerform a very quick nearest-neighbour cross-match between two catalogs.
	
	Even faster, but dumber.
	
	This function assumes that only the first two columns (e.g., x and y positions) are relevant.

	Parameters:
		test (np.ndarray): Test dataset of shape (N, 2+), where only columns [0,1] are used.
		ref (np.ndarray): Reference dataset of shape (K, 2+), same format.
		max_sep (float): Maximum separation allowed for a match (in same units as positions).

	Returns:
		test_match_idx (list): Indices of matched sources in the test array.
		ref_match_idx (list): Corresponding matched indices in the ref array.
		dist (list): Euclidean distances of the matches.
	
	"""
	
	#Lists to store the matching results
	test_match_idx = []
	ref_match_idx = []
	dist = []
	
	#Define loop limits
	#Either all the possible match has been made
	#or all sources has been tested.
	max_it = max(test.shape[0], ref.shape[0])	#Max number of iterations
	max_match = min(test.shape[0], ref.shape[0])	#Max number of possible matches
	
	it_count = 0
	match_count = 0
	
	#Iterative matching process
	while it_count <= max_it and match_count <= max_match:
	
		for i in range(test.shape[0]):
		
			#Skip if this test source already matched
			if i not in test_match_idx:
		
				source_test = test[i,:]	#Current test source
				
				#Compute distance to all reference sources
				separation = np.sqrt((source_test[0] - ref[:,0])**2 + (source_test[1] - ref[:,1])**2)
			
				#Sort candidate indices by increasing distance
				sorting = np.argsort(separation)

				for id_candidate in sorting:

					#Skip if candidate is too far away
					if separation[id_candidate] > max_sep:
						continue
					
					#If candidate hasn't been matched, accept it
					if id_candidate not in ref_match_idx:
					
						test_match_idx.append(i)
						ref_match_idx.append(id_candidate)
						dist.append(separation[id_candidate])
						
						match_count += 1
						
						break
					
					#If candidate already matched, see if current match is better
					else:
						ind_to_compare = ref_match_idx.index(id_candidate)
						
						if dist[ind_to_compare] < separation[id_candidate]:
							
							test_match_idx.pop(ind_to_compare)
							ref_match_idx.pop(ind_to_compare)
							dist.pop(ind_to_compare)
							
							
							test_match_idx.append(i)
							ref_match_idx.append(id_candidate)
							dist.append(multi_parameter_error[id_candidate])
							
							#Retry current test source in next round
							it_count -= 1
								
							break
			
				it_count += 1


	return test_match_idx, ref_match_idx, dist
