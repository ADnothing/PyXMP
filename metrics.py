from Xmatch import *
import numpy as np

def eval(matching_func, test, ref, verbose=False, beta=1, *args, **kwargs):
	"""
	Evaluate a matching function's performance using Recall, Precision, and F-beta score.
	
	Parameters:
		matching_func (callable): The matching function to evaluate.
		test: The catalog to test (passed to matching_func).
		ref: The reference catalog (passed to matching_func).
		verbose (bool): If True, print evaluation metrics. Default is False.
		beta (float): Weighting factor for the F-score. Default is 1 (i.e., F1-score).
		*args: Positional arguments passed to matching_func.
		**kwargs: Keyword arguments passed to matching_func.

	Returns:
		tuple: (Recall, Precision, F_score, Nmatch)
	"""

	matches = matching_func(test, ref,*args, **kwargs)
	if isinstance(matches, (list, np.ndarray)):
		Nmatch = len(matches)
	elif isinstance(matches, tuple):
		Nmatch = len(matches[0])
	else:
		raise TypeError("Unknown output type")
		
	Nref = len(ref)
	Ntest = len(test)
	
	Recall = Nmatch/Nref
	Precision = Nmatch/Ntest
	
	F_score = ((1 + beta**2)*Precision*Recall)/(Precision*beta**2 + Recall)
	
	if verbose:
		print(f"Nb match: {Nmatch})
		print(f"Recall: {np.round(100*Recall)} %")
		print(f"Precision: {np.round(100*Precision)} %")
		print(f"F{beta} score: {np.round(F_score)}")
		
	return Recall, Precision, F_score, Nmatch
	
#===================================================================================================================

def recall(test, ref, thresholds, method="fast", *args, **kwargs):
	"""
	Compute recall for a matching function based on the specified method from Xmatch.

	Parameters:
		test: The catalog to test.
		ref: The reference catalog.
		thresholds (list, array or float): Thresholds to be used for matching.
		method (str): Matching method to use ('fast', 'vectorized', or 'dumb').
		*args: Additional positional arguments passed to the matching function.
		**kwargs: Additional keyword arguments passed to the matching function.

	Returns:
		float: Recall value (fraction of reference sources matched).
	"""
	
	Nref = len(ref)
	
	if method == "fast":
		test_match_idx, ref_match_idx, score = fast_Xmatch(test, ref, thresholds)
		Recall = len(test_match_idx)/Nref
	elif method == "vectorized":
		candidates = Xmatch(cat_test, cat_ref, *args, thresholds, **kwargs)
		Recall = len(candidates)/Nref
	elif method == "dumb":
		test_match_idx, ref_match_idx, score = even_faster_Xmatch(test, ref, thresholds)
			Recall = len(test_match_idx)/Nref
	else:
		raise ValueError("{method} is unknown, use fast, vectorized or dumb.")
	
	return Recall
	
#===================================================================================================================
	
def precision(test, ref, thresholds, method="fast", *args, **kwargs):
	"""
	Compute precision for a matching function based on the specified method.

	Parameters:
		test: The catalog to test.
		ref: The reference catalog.
		thresholds (list, array, or float): Thresholds to be used for matching.
		method (str): Matching method to use ('fast', 'vectorized', or 'dumb').
		*args: Additional positional arguments passed to the matching function.
		**kwargs: Additional keyword arguments passed to the matching function.

	Returns:
		float: Precision value (fraction of test sources matched)
	"""
	
	Ntest = len(test)
	
	if method == "fast":
		test_match_idx, ref_match_idx, score = fast_Xmatch(test, ref, thresholds)
		Precision = len(test_match_idx)/Ntest
	elif method == "vectorized":
		candidates = Xmatch(cat_test, cat_ref, *args, thresholds, **kwargs)
		Precision = len(candidates)/Ntest
	elif method == "dumb":
		test_match_idx, ref_match_idx, score = even_faster_Xmatch(test, ref, thresholds)
			Precision = len(test_match_idx)/Ntest
	else:
		raise ValueError("{method} is unknown, use fast, vectorized or dumb.")
	
	return Precision
	
#===================================================================================================================
	
def f_beta_score(test, ref, thresholds, beta=1, method="fast", *args, **kwargs):
	"""
	Compute the F-beta score between test and reference catalogs.
	The F-beta score balances precision and recall, with a weighting factor beta.

	Parameters:
		test: The catalog to test.
		ref: The reference catalog.
		thresholds (list, array, or float): Matching thresholds.
		beta (float): Weighting factor for recall in the F-score.
		method (str): Matching method ('fast', 'vectorized', or 'dumb').
		*args: Additional positional arguments for the matching function.
		**kwargs: Additional keyword arguments for the matching function.

	Returns:
		float: The computed F-beta score.
	"""
	
	Precision = precision(test, ref, thresholds, method, *args, **kwargs)
	Recall(test, ref, thresholds, method, *args, **kwargs)
	
	F_score = ((1 + beta**2)*Precision*Recall)/(Precision*beta**2 + Recall)
	
	return F_score
	
#===================================================================================================================
	
def match_score(test, ref, thresholds, beta=1, method="fast", *args, **kwargs):
	"""
	Compute statistics (mean, median, std) of match quality scores between test and reference catalogs.

	Parameters:
		test: The catalog to test.
        	ref: The reference catalog.
		thresholds (list or array): Matching thresholds.
		beta (float): Unused currently, included for interface compatibility.
		method (str): Matching method to use ('fast', 'vectorized', 'dumb').
		*args: Additional positional arguments passed to the matching function.
		**kwargs: Additional keyword arguments passed to the matching function.

	Returns:
		tuple: (mean_score_match, median_score_match, std_score_match)
	"""
	
	if method == "fast":
		test_match_idx, ref_match_idx, score = fast_Xmatch(test, ref, thresholds)
		mean_score_match = np.mean(np.array(score))
		median_score_match = np.median(np.array(score))
		std_score_match = np.std(np.array(score))
	elif method == "vectorized":
		candidates = Xmatch(cat_test, cat_ref, *args, thresholds, **kwargs)
		mean_score_match = np.mean(candidates[:,-1])
		median_score_match = np.median(candidates[:,-1])
		std_score_match = np.std(candidates[:,-1])
	elif method == "dumb":
		test_match_idx, ref_match_idx, score = even_faster_Xmatch(test, ref, thresholds)
			mean_score_match = np.mean(np.array(score))
			median_score_match = np.median(np.array(score))
			std_score_match = np.std(np.array(score))
	else:
		raise ValueError("{method} is unknown, use fast, vectorized or dumb.")
		
		
	return mean_score_match, median_score_match, std_score_match
