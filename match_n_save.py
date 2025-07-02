#!/usr/bin/env python

import numpy as np

from Xmatch import Xmatch, fast_Xmatch, even_faster_Xmatch


def save_match(test, ref, thresholds, method="fast", header_test="", fmt="", verbose=False, match_only=False, filename="",  *args, **kwargs):
	"""
	"""
	
	if method == "fast":
		test_match_idx, ref_match_idx, score = fast_Xmatch(test, ref, thresholds)
	elif method == "vectorized":
		candidates = Xmatch(cat_test, cat_ref, *args, thresholds, **kwargs)
		test_match_idx = candidates[:,0]
		ref_match_idx = candidates[:,1]
		score = candidates[:,2]
	elif method == "dumb":
		test_match_idx, ref_match_idx, score = even_faster_Xmatch(test, ref, thresholds)
	else:
		raise ValueError("{method} is unknown, use fast, vectorized or dumb.")
	
	if verbose:
		Nref = len(ref)
		Ntest = len(test)
		Nmatch = len(test_match_idx)
		
		recall = Nmatch/Nref
		precision = Nmatch/Ntest
		
		print(f"Recall: {np.round(100*recall)} %")
		print(f"Precision: {np.round(100*precision)} %")
		
	
	if len(headedr_test)==0:
		ncol_test = test.shape[1]
		for col in range(ncol_test):
			header_test += f"col_{col} "
		header_test += "matched ref_id match_score"
		
	else:
		header_test += " matched ref_id match_score"
		
	
	matched_cat = np.zeros((test.shape[0], test.shape[1]+3))*np.nan	
	matched_cat[:,:-3] = test
	matched_cat[test_match_idx.astype(int), -3] = 1
	matched_cat[test_match_idx.astype(int), -2] = ref_match_idx
	matched_cat[test_match_idx.astype(int), -1] = score
	
	
	if match_only:
	
		if len(filename)==0:
			filemane = "matched_cat_match_only.txt"
		
		np.savetxt(filename, matched_cat[test_match_idx.astype(int),:], header=header_test, fmt=fmt)
		
	else:
	
		if len(filename)==0:
			filemane = "matched_cat.txt"
		
		np.savetxt(filename, matched_cat, header=header_test, fmt=fmt)
