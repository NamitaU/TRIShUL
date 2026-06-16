import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt, colors
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time
from joblib import Parallel, delayed
import multiprocessing
import concurrent.futures # running code in parallel
from scipy.stats import mannwhitneyu
import os
################################### SETP -1 ################################################

################### calculating cumulative Mahalanobis distance  ##################################

def cummulative_pol(df, pt= True): #window=3,
	df = df.sort_values('mu')
	cumm_dMaha = np.cumsum(df['dMaha'])
	if(pt == True):
		fig, ax1= plt.subplots(figsize=(8,6))
		plt.plot(df.index.values, cumm_dMaha, label="Cumulative dMaha", color="green", marker = 'o', ls = 'none')
		plt.xlabel("Distance")
		plt.ylabel("cumulative Mahalanobis distance")
		plt.legend()
		plt.show()
	df['cum_dMaha'] = cumm_dMaha
	return(df)


################################### SETP -2 ################################################

########################## detecting breakpoints ###############################################
def Rbreaks(Dpol):
	### using R-strucchange package ######
	import rpy2.robjects as ro
	from rpy2.robjects import pandas2ri
	pandas2ri.activate()
	df1 = pd.DataFrame({'indices': Dpol.index.values, 'V': Dpol['cum_dMaha'].values})
	ro.globalenv['df1'] = df1
	ro.r('library(strucchange)')
	try:
		ro.r('bp_model <- breakpoints(V ~ indices, data = df1, h  = 5)') #, h = 5, breaks = 5
	except Exception as e:
		print("Initial breakpoint model failed:", e)
		return np.array([]), np.array([]), np.array([]), np.array([])
	breaks = ro.r('df1$indices[bp_model$breakpoints]') 
	break_test = list(breaks)
	break_test.append(df1['indices'].max())
	if any((j - i) < 5 for i, j in zip(break_test[:-1], break_test[1:])):
		print("Some breakpoints are too close; re-running with h=5")
		ro.r('bp_model <- breakpoints(V ~ indices, data = df1, h = 5, breaks = 5)')
		breaks = (ro.r('df1$indices[bp_model$breakpoints]'))
	breaks_np = np.array([int(b) if b is not ro.NA_Integer and not pd.isna(b) else -1 for b in breaks])
	if np.all(breaks_np == -1):
		print("All breakpoints are NA — returning empty outputs.")
		return np.array([]), np.array([]), np.array([]), np.array([])
	ro.r('conf <- confint(bp_model, level = 0.683)')
	full = ro.r('conf$confint')
	full = np.clip(full, 0, len(Dpol) - 1)    
	print(full)
	lower_error = np.abs(full[:, 1] - full[:, 0])
	upper_error = np.abs(full[:, 2] - full[:, 1])
	lower_error = np.where(np.isnan(lower_error), 1, lower_error)
	upper_error = np.where(np.isnan(upper_error), 1, upper_error)
	final_error = np.maximum(lower_error, upper_error)
	return(breaks, final_error, lower_error, upper_error)

################################### SETP -3 ################################################

###################### rejecting spurious layers ########################


######## some useful function to be used in rejection process ############################### 
def sigma_clipping_errors(dataframe, eq_col='eq', eu_col='eu'):                             #
	# Calculate the mean and standard deviation for the error columns                       #
	eq_mean, eq_std = dataframe[eq_col].mean(), dataframe[eq_col].std()                     #
	eu_mean, eu_std = dataframe[eu_col].mean(), dataframe[eu_col].std()                     #
	# Define 3-sigma range for errors                                                       #
	eq_lower, eq_upper = eq_mean - 2 * eq_std, eq_mean + 2 * eq_std	                        #
	eu_lower, eu_upper = eu_mean - 2 * eu_std, eu_mean + 2 * eu_std                         #
	# Apply the 3-sigma clipping condition for the error columns                            #
	condition = ((dataframe[eq_col] >= eq_lower) & (dataframe[eq_col] <= eq_upper) &                      #
	(dataframe[eu_col] >= eu_lower) & (dataframe[eu_col] <= eu_upper))                                    #
	# Filter the DataFrame based on the condition                                           #
	print('####### stars before sigma clipping ####### = ', len(dataframe))                 #
	print('####### stars left after sigma clipping ####### = ', len(dataframe[condition]))  #
	return dataframe[condition]                                                             #
	                                                                                        #
# vectors and covariance matrix #                                                           #                                 #
def extract_components(data):                                                               #
    q = data['q'].values                                                                    #
    u = data['u'].values                                                                    #
    eq = data['eq'].values                                                                  #
    eu = data['eu'].values                                                                  #
    Cqu = data['Cqu'].values  # Off-diagonal covariance (Cqu)                               #
    X = np.stack((q, u), axis=1)                                                            #
    covs = np.array([                                                                       #
        [[eq[i]**2, Cqu[i]], [Cqu[i], eu[i]**2]] for i in range(len(eq))                    #
    ])                                                                                      #
    return X, covs                                                                          #
                                                                                            #
# Function to calculate weighted mean, unbiased covariance, and error on weighted mean #    #
def weighted_cov_mean(X, covs):                                                             #                                                 #
	W = np.linalg.inv(covs)                                                                 #
	W_sum = np.sum(W, axis=0)                                                               #  
	weighted_sum = np.einsum('nij,ni->j', W, X)                                             #
	mean = np.linalg.solve(W_sum, weighted_sum)                                             #
	mean_cov = np.linalg.inv(W_sum)   # error on weighted means                             #
	res = X - mean                                                                          #
	S = np.einsum('nij,ni,nj->ij', W, res, res)                                             #
	tr_W = np.trace(W_sum)                                                                  #
	tr_W2 = np.sum(np.einsum('nij,njk->', W, W))                                            #
	nu = tr_W - (tr_W2 / tr_W)                                                              #
	cov_unbiased = S / nu                                                                   #
	return mean, cov_unbiased                                                               #	
def weighted_mean_err(X, covs):                                                             #
	W = np.linalg.inv(covs) 	                                                            #
	W_sum = np.sum(W, axis=0)                                                               # 
	weighted_sum = np.einsum('nij,ni->j', W, X)                                             #
	mean = np.linalg.solve(W_sum, weighted_sum)	                                            #
	mean_cov = np.linalg.inv(W_sum)                                                         #
	return mean, mean_cov                                                                   #
#############################################################################################

# 1a. Define Hotelling's T^2 test                                                                     
from pingouin import multivariate_ttest                                                                                                                   #
def HT(d1, d2):                                                                             
	H = multivariate_ttest(d1[['q', 'u']], d2[['q', 'u']])                                   
	T2_val = H['T2'].values                                                                 
	pval_val = H['pval'].values                                                             
	return(T2_val[0], pval_val[0]) 

# 1b. Applying Hotteling test on two samples 
def run_one_sim(mean, cov_intrinsic, Covs1, Covs2):
	mock1 = [np.random.multivariate_normal(mean, cov_intrinsic + cov_i) for cov_i in Covs1]
	mock2 = [np.random.multivariate_normal(mean, cov_intrinsic + cov_i) for cov_i in Covs2]
	T2_val, p_val = HT(pd.DataFrame(mock1, columns=['q', 'u']), pd.DataFrame(mock2, columns=['q', 'u']))
	return T2_val
    
# 1c. Checking Null hypothesis if the two segments following same parent distribution .
def check_break_significance(segment1, segment2, alpha=0.05, n_iter=1000):
	segment1_clip = sigma_clipping_errors(segment1, eq_col='eq', eu_col='eu') 
	segment2_clip = sigma_clipping_errors(segment2, eq_col='eq', eu_col='eu')
	X1, Covs1 = extract_components(segment1_clip)
	X2, Covs2 = extract_components(segment2_clip)
	T2_val_t, p_val_t = HT(pd.DataFrame(X1, columns=['q', 'u']), pd.DataFrame(X2, columns=['q', 'u']))
    ########### preparing for random draws #################
	combined_clip = pd.concat([segment1_clip, segment2_clip])
	# Extract components and covariance matrices
	X_comb, covs_comb = extract_components(combined_clip)
	# Calculate the weighted mean and unbiased covariance
	mean, cov_unbiased = weighted_cov_mean(X_comb, covs_comb) # this cov_unbiased has both intrinsic and uncertinties contribution
	########## contribution from errors - simple mean of covs_comb ################# 
	mean_cov = np.mean(covs_comb, axis=0)
	########################## intrinsic spread estimate ##################
	cov_intrinsic = cov_unbiased - mean_cov
	print(cov_intrinsic)
	if np.any(cov_intrinsic < 0):
		cov_intrinsic = np.zeros_like(cov_intrinsic)
	print(cov_intrinsic)
	# Simulate the null hypothesis by generating mock segments
	HT_p_vals = Parallel(n_jobs=-1)(delayed(run_one_sim)(mean, cov_intrinsic, Covs1, Covs2) for _ in range(n_iter))
	frac_rej = sum(HT_p_vals >= T2_val_t)/len(HT_p_vals)
	print(' fraction of cases accepting Null hypothesis (should be less than 0.05 - 50 out of 1000) = ', frac_rej)
	sigma = np.sqrt(0.01*(1-0.01)/n_iter)
	if((frac_rej > 0.01 - sigma) & (frac_rej < 0.01 + sigma)):
		n_iter = 10000
		print('true value lies at the tail of null hypothesis distribution')
		HT_p_vals = Parallel(n_jobs=-1)(delayed(run_one_sim)(mean, cov_intrinsic, Covs1, Covs2) for _ in range(n_iter))
		frac_rej = sum(HT_p_vals >= T2_val_t)/len(HT_p_vals)
		print(' fraction of cases accepting Null hypothesis (should be less than 0.05 - 50 out of 1000) = ', frac_rej)
	return(frac_rej < 0.01)

#######################################################################################

# 2. Remove Jumpers 
	
def remove_jumpers(df, left_segment, right_segment, break_point, break_error, max_fraction = 0.1):
	left_segment['distance'] = abs(left_segment['plx'] - df['plx'].iloc[int(break_point)]) /np.sqrt(left_segment['eplx']**2)
	right_segment['distance'] = abs(right_segment['plx'] - df['plx'].iloc[int(break_point)]) /np.sqrt(right_segment['eplx']**2 )
	print('############## \n number of stars in left segment of ', break_point, ' = ', len(left_segment))
	print('############## \n number of stars in right segment of ', break_point, ' = ', len(right_segment))
	# Sort the stars by their distance to the break (ascending)
	left_candidates = left_segment[left_segment['distance'] <= 3]
	right_candidates = right_segment[right_segment['distance'] <= 3]
	left_max_remove = int(len(left_segment) * max_fraction)
	right_max_remove = int(len(right_segment) * max_fraction)
	if len(left_candidates) <= left_max_remove:
		cleaned_left = left_segment.drop(index=left_candidates.index)
	else:
		closest_left = left_segment.nsmallest(left_max_remove, 'distance')
		cleaned_left = left_segment.drop(index=closest_left.index)
	if len(right_candidates) <= right_max_remove:
		cleaned_right = right_segment.drop(index=right_candidates.index)
	else:
		closest_right = right_segment.nsmallest(right_max_remove, 'distance')
		cleaned_right = right_segment.drop(index=closest_right.index)
	left_segment = cleaned_left
	right_segment = cleaned_right
	print('\n ############## \n number of stars in left segment after removing 10% sample around ', break_point, ' = ', len(left_segment))
	print('\n ############## \n number of stars in right segment after removing 10% sample around ', break_point, ' = ', len(right_segment))
	return left_segment, right_segment
	

##########################################################################

# 3. Testing Null hypothesis on forward and backward pass 

# 3a. Backward pass 
def test_segments_backward(df, mu_clouds, e_mu_clouds, param,nth_layer, alpha=0.05, n_iter=1000):
	mu_clouds = np.round(mu_clouds).astype(int)
	A =1
	while(A==1):
		back_breaks = []
		back_uncertainties = []
		next_break = max(df.index)
		for i in range(len(mu_clouds)- nth_layer):
			l = len(mu_clouds)-1-i
			current_break = mu_clouds[l]
			current_error = e_mu_clouds[l]
			print('cloud =  ', current_break)
			segment1 = df.iloc[current_break+1 : next_break+1]
			if(l-nth_layer > 0):
				segment2 = df.iloc[mu_clouds[l-1]+1 : current_break+1]
			else:
				segment2 = df.iloc[mu_clouds[nth_layer-1]+1: current_break+1] if nth_layer > 0 else df.iloc[min(df.index): current_break+1]
			if len(segment1) < 5 or len(segment2)< 5:
				continue
			segment1, segment2 = remove_jumpers(df,segment1, segment2, current_break,current_error, max_fraction = 0.1)
			is_significant = check_break_significance(segment1, segment2, alpha=alpha, n_iter=n_iter)
			if is_significant:
				print(current_break)
				back_breaks.append(current_break)
				back_uncertainties.append(e_mu_clouds[l])
				next_break = current_break
			else:
				next_break = next_break 
		if(len(mu_clouds)-nth_layer> len(back_breaks)):
			A = 1
		else:
			A = 0
		added_layers = list(mu_clouds[:int(nth_layer)])[::-1]
		added_layer_error = list(e_mu_clouds[:int(nth_layer)])[::-1]
		back_breaks = back_breaks + added_layers 
		back_uncertainties = back_uncertainties + added_layer_error
		print('breaks left after each run = ', back_breaks )
		mu_clouds = list(reversed(back_breaks))
		print('new mu_clouds = ', mu_clouds)
		e_mu_clouds = list(reversed(back_uncertainties))
	return mu_clouds, e_mu_clouds
	
	
# 3b. Forward pass	

def test_segments_forward(df, mu_clouds, e_mu_clouds, param, nth_layer, alpha=0.05, n_iter = 1000):
    mu_clouds = np.round(mu_clouds).astype(int)
    A = 1
    while(A==1):
    	forward_breaks = []
    	forward_uncertainties = []
    	if(nth_layer == 0):
    		prev_break = min(df.index)
    	else:
    		prev_break = mu_clouds[int(nth_layer-1)] 
    	for i in range(int(nth_layer), len(mu_clouds)):
    		current_break = mu_clouds[i]
    		print('cloud =  ', current_break)
    		current_error = e_mu_clouds[i]
    		segment1 = df.iloc[prev_break+1:current_break+1] 
    		if(i+1 < len(mu_clouds)):
    			segment2 = df.iloc[current_break+1 : mu_clouds[i+1]+1]
    		else:
    			segment2 = df.iloc[current_break+1 : max(df.index)+1]
    		
    		if len(segment1) < 5 or len(segment2) < 5: continue
    		segment1, segment2 = remove_jumpers(df, segment1, segment2, current_break,current_error, max_fraction = 0.1)
    		is_significant = check_break_significance(segment1, segment2, alpha=alpha, n_iter=n_iter)
    		if is_significant:
    			forward_breaks.append(current_break)
    			forward_uncertainties.append(e_mu_clouds[i])
    			prev_break = current_break
    		else:
    			prev_break = prev_break 
    	if(len(mu_clouds)-nth_layer> len(forward_breaks)):
    		A = 1
    	else:
    		A = 0
    	mu_clouds =list(mu_clouds[:int(nth_layer)]) + forward_breaks
    	e_mu_clouds = list(e_mu_clouds[:int(nth_layer)]) + forward_uncertainties
    return mu_clouds, e_mu_clouds
  
    
# 3c. Combined results from forwards and backward pass

def detect_significant_breaks(df, mu_clouds, e_mu_clouds, param, nth_layer, alpha=0.05, n_iter=10000):
    forward_breaks, forward_uncertainties = test_segments_forward(df, mu_clouds, e_mu_clouds, param=param,nth_layer = nth_layer, alpha=alpha, n_iter=n_iter)
    print('############# \n  \n \n Significant breaks in forward run = ', forward_breaks, forward_uncertainties, '\n \n \n ###############')
    backward_breaks, backward_uncertainties = test_segments_backward(df, mu_clouds, e_mu_clouds, param=param, nth_layer=nth_layer, alpha=alpha, n_iter=n_iter)
    print('############# \n \n \n  Significant breaks in backward run = ', backward_breaks, backward_uncertainties, '\n \n \n ###############')
    all_breaks = forward_breaks + backward_breaks
    all_uncertainties = forward_uncertainties + backward_uncertainties
    unique_breaks = []
    unique_uncertainties = []
    for break_point, uncertainty in zip(all_breaks, all_uncertainties):
        if break_point not in unique_breaks:
            unique_breaks.append(break_point)
            unique_uncertainties.append(uncertainty)
    sort_ind = np.argsort(np.array(unique_breaks))
    unique_breaks = np.array(unique_breaks)[sort_ind]
    unique_uncertainties = np.array(unique_uncertainties)[sort_ind]
    mask = np.isin(np.round(mu_clouds), unique_breaks) 
    mu_clouds = np.array(mu_clouds)[mask]
    e_mu_clouds = np.array(e_mu_clouds)[mask]
    flag_array = np.array([
        0 if (break_point in forward_breaks and break_point in backward_breaks) else 1 
        for break_point in unique_breaks
    ])
   
    if(np.any(flag_array)==1):
    	forward_breaks, forward_uncertainties = test_segments_forward(df, mu_clouds, e_mu_clouds, param=param,nth_layer = nth_layer, alpha=alpha, n_iter=n_iter)
    	backward_breaks, backward_uncertainties = test_segments_backward(df, mu_clouds, e_mu_clouds, param=param, nth_layer = nth_layer,alpha=alpha, n_iter=n_iter)
    	all_breaks = forward_breaks + backward_breaks
    	all_uncertainties = forward_uncertainties + backward_uncertainties
    	unique_breaks = []
    	unique_uncertainties = []
    	for break_point, uncertainty in zip(all_breaks, all_uncertainties):
    		if break_point not in unique_breaks:
    			unique_breaks.append(break_point)
    			unique_uncertainties.append(uncertainty)
    	sort_ind = np.argsort(np.array(unique_breaks))
    	unique_breaks = np.array(unique_breaks)[sort_ind]
    	unique_uncertainties = np.array(unique_uncertainties)[sort_ind]
    	flag_array = np.array([0 if (break_point in forward_breaks and break_point in backward_breaks) else 1 for break_point in unique_breaks])
    return unique_breaks, unique_uncertainties, flag_array      
        
#####################################################################################################        


## 4. optional step: Adding the known external foreground layer for a hybrid solution. 
#If we already know the foregorund layers and want to add that information, 'res_hybrid should be true and 
# res_hybrid should be a csv file having the layeout similar to the output file of Trishul'

def merge_foreground(df, res_hybrid, mu_clouds, e_mu_clouds):
	if((len(res_hybrid) == 0) & (len(mu_clouds) == 0)):
		result_clouds = []
		result_e_clouds = []
		return result_clouds, result_e_clouds
	hybrid_plx = res_hybrid.loc['plx_calc'].values
	hybrid_stat_left = res_hybrid.loc['stats_plx_left'].values
	hybrid_stat_right = res_hybrid.loc['stats_plx_right'].values
	hybrid_sys_right = res_hybrid.loc['sys_plx_right'].values
	hybrid_sys_left = res_hybrid.loc['sys_plx_left'].values
	
	hybrid_indx = []
	hybrid_e_indx= []
	
	for k in range(len(hybrid_plx)):
		plx_diff = np.abs(df['plx'] - hybrid_plx[k])
		plx_left = np.abs(df['plx'] - max(hybrid_stat_left[k], hybrid_sys_left[k] ))
		plx_right = np.abs(df['plx'] - min(hybrid_stat_right[k], hybrid_sys_right[k] ))
		plx_indx = np.argmin(plx_diff)
		left_err = plx_indx - np.argmin(plx_left)
		right_err = np.argmin(plx_right) - plx_indx
		hybrid_indx.append(plx_indx)
		hybrid_e_indx.append(max(left_err, right_err))
	if(len(mu_clouds)!= 0):
		x0 = mu_clouds[0]
		ex0 = e_mu_clouds[0]
		for i, x in enumerate(hybrid_indx):
			ex = hybrid_e_indx[i]
			if(int(x0-ex0) <= x+ex):
				hybrid_foreground = hybrid_indx[:i]
				result_clouds = np.concatenate([hybrid_foreground, mu_clouds])
				result_e_clouds = np.concatenate([hybrid_e_indx[:i], e_mu_clouds])
				return result_clouds, result_e_clouds
	result_clouds = np.concatenate([hybrid_indx, mu_clouds])
	result_e_clouds = np.concatenate([hybrid_e_indx, e_mu_clouds])
	return result_clouds, result_e_clouds
	
##############################################################################################

# 5. optional step: special foreground identification ####################################	
	
def Layer1(df, mu_clouds, e_mu_clouds):
	# check if the weighted av q and u of the first segment is consistent with (0,0) within 2 sigma.
	if((len(mu_clouds)==0) & (len(df)!=0)):
		cl1 = max(df.index)
		ecl1 = 0
		cloud1 = max(df.index)
		cloud1_from_mu = False
	else:
		cloud1 = mu_clouds[0]
		cl1 = np.floor(mu_clouds[0]-(e_mu_clouds[0])) 
		ecl1 = e_mu_clouds[0]
		cloud1_from_mu = True
	valid = 0
	while(valid == 0):
			seg1 = df[min(df.index): int(cl1)+1]
			if(len(seg1) == 0):
				valid = 1
				cloud1 = 0
				ecl1 = 0
			else:
				q_mean = weighted_avg(seg1['q'], seg1['eq'])
				q_err = weighted_error(seg1['eq'])
				u_mean = weighted_avg(seg1['u'], seg1['eu'])
				u_err = weighted_error(seg1['eu'])
				if((np.abs(u_mean) < 2* u_err) &  (np.abs(q_mean) < 2* q_err)): 
					print('first layer is valid')
					cloud1 = cloud1
					ecl1 = ecl1
					flag1 = 0
					valid = 1
				else:
					valid = 0
					cl1 = cl1-1
					cloud1 = cl1    
					print(cloud1)
					ecl1 = 1	
	if(cloud1_from_mu):
		if(mu_clouds[0]!=cloud1):
			mu_clouds = np.append(cloud1, mu_clouds)
			e_mu_clouds = np.append(ecl1, e_mu_clouds)
		else:
			mu_clouds = mu_clouds
			e_mu_clouds = e_mu_clouds
	else:
		mu_clouds = np.array([cloud1])
		e_mu_clouds = np.array([ecl1])	
	return mu_clouds, e_mu_clouds
	
	
##################################################################################################
	
# 6. identifying the layers in index space

def trisul_run(df, param, res_hybrid = None, pt1 = False, pt2= False, pt3=False):
	Dpol = cummulative_pol(df, pt=False)
	p_mu_c, p_e_mu_c, pel, peu = Rbreaks(Dpol)
	print('solution of R-script = ', p_mu_c, p_e_mu_c, pel, peu)
	
	if res_hybrid is None:   
		p_mu_clouds, e_p_mu_clouds = Layer1(df, p_mu_c, p_e_mu_c)
		
	else:
		p_mu_clouds, e_p_mu_clouds = merge_foreground(df, res_hybrid, p_mu_c, pel)
	
	print('breakpoint solution: cloud detected in p  =  ', p_mu_clouds)
	print('their errors = ', e_p_mu_clouds)
	nth_layer = len(p_mu_clouds) - len(p_mu_c) 
	pel = np.append(e_p_mu_clouds[:nth_layer], pel)
	peu = np.append(e_p_mu_clouds[:nth_layer], peu)
	print('complete solution after adding layers = ', p_mu_clouds, e_p_mu_clouds, pel, peu)
	if(nth_layer == 0):
		nth_layer = 1 
	if(pt1==True):
		plt.figure(figsize=(10, 6))
		plt.plot(Dpol.index.values, Dpol['cum_'+param], label='Cumulative '+param , color="blue", marker = 'o', ls = 'none')
		for r in range(len(p_mu_clouds)):
			plt.axvline(x = p_mu_clouds[r], color = 'm', ls = '-')
			plt.fill_betweenx(y=[Dpol['cum_'+param].min(), Dpol['cum_'+param].max()], 
			x1= p_mu_clouds[r] - e_p_mu_clouds[r], x2= p_mu_clouds[r] + e_p_mu_clouds[r], color='magenta', alpha=0.1)
		plt.title('breaks identified in cumulative '+param)
		plt.show()
	if(pt2==True):
		plt.figure(figsize=(10, 6))
		plt.errorbar(df['mu'], df['q'], xerr = df['s_mu'],  yerr = df['eq'], marker = 'o', color = 'green', ecolor = 'green', fmt='--', ls='none', capsize = 3, alpha = 0.5)
		plt.errorbar(df['mu'], df['u'],xerr= df['s_mu'], yerr = df['eu'], marker = 'o', color = 'blue', ecolor = 'blue', fmt='--', ls='none', capsize = 3, alpha = 0.5)
		for r in range(len(p_mu_clouds)):
			plt.axvline(x = df['mu'].iloc[int(np.round(p_mu_clouds[r]))], color = 'm', ls = '-')
			plt.fill_betweenx(y=[min(df['q'].min(),df['u'].min()), max(df['q'].max(), df['u'].max())], 
			x1= df['mu'].iloc[int(np.round(p_mu_clouds[r] - e_p_mu_clouds[r]))], x2= df['mu'].iloc[int(np.round(p_mu_clouds[r] + e_p_mu_clouds[r]))], color='red', alpha=0.1)
		plt.xlabel('mu')
		plt.ylabel('q / u')
		plt.show()
	mu_clouds, e_mu_clouds, flag = detect_significant_breaks(df, p_mu_clouds, e_p_mu_clouds, param, nth_layer, alpha=0.05, n_iter = 1000)
	print('flagged (1) or not (0): ', flag)
	print('number of breaks left in combined P = ', mu_clouds)
	if(pt3 == True):
		plt.figure(figsize=(10, 6))
		plt.errorbar(df['mu'], df['q'], xerr = df['s_mu'],  yerr = df['eq'], marker = 'o', color = 'green', ecolor = 'green', fmt='--', ls='none', capsize = 3, alpha = 0.5)
		plt.errorbar(df['mu'], df['u'],xerr= df['s_mu'], yerr = df['eu'], marker = 'o', color = 'blue', ecolor = 'blue', fmt='--', ls='none', capsize = 3, alpha = 0.5)
		for r in range(len(mu_clouds)):
			plt.axvline(x = df['mu'].iloc[int(np.round(mu_clouds[r]))], color = 'r', ls = '-', label =param+' breaks')
			plt.fill_betweenx(y=[min(df['q'].min(),df['u'].min()), max(df['q'].max(), df['u'].max())], 
			x1= df['mu'].iloc[int(np.round(mu_clouds[r] - e_mu_clouds[r]))], x2= df['mu'].iloc[int(np.round(mu_clouds[r] + e_mu_clouds[r]))], color='red', alpha=0.1)
		plt.xlabel('mu')
		plt.ylabel('q / u')
		plt.legend()
		plt.show()
	print('breaks together in q and u = ', mu_clouds)
	print('their errors = ', e_mu_clouds)
	if(len(mu_clouds)!=0):
		mask = np.isin(np.round(p_mu_clouds), mu_clouds) 
		mu_clouds = np.array(p_mu_clouds)[mask]
		e_mu_clouds = np.array(e_p_mu_clouds)[mask]
		e_mu_lower = pel[mask]
		e_mu_upper = peu[mask]
		if(len(flag) < len(mu_clouds)):
			flag = np.concatenate([np.zeros(len(mu_clouds)-len(flag)),flag]) 
	print('solution = ', mu_clouds, e_mu_clouds, e_mu_lower, e_mu_upper, flag) 
	return mu_clouds, e_mu_clouds, e_mu_lower, e_mu_upper, flag, nth_layer
########################################################################################################

def weighted_avg(values, errors):
    """Compute weighted average using weights = 1/error^2"""
    weights = 1 / (errors ** 2)
    return np.sum(values * weights) / np.sum(weights)

def weighted_error(errors):
    """Compute weighted error using weights = 1/error^2"""
    weights = 1 / (errors ** 2)
    return np.sqrt(1 / np.sum(weights))

########################################################################################################

## 7. uncertainties calculation in parallax space ################

## 7a. useful function to be used in sys_unc_cal() function
 
def process_segment(test_segment, ref_segment, q_av = None, u_av = None):
	L = len(test_segment)
	test_segment = test_segment.sort_values(by='distance', ignore_index=True)
	# Compute weighted averages for the reference segment
	if q_av is None and u_av is None:
		q_av = weighted_avg(ref_segment['q'], ref_segment['eq'])
		u_av = weighted_avg(ref_segment['u'], ref_segment['eu'])
	selected = []
	for i in range(1, len(test_segment)+1):
		selected = test_segment.iloc[:i]
		# Compute weighted avg and error for selected stars
		q_w = weighted_avg(selected['q'], selected['eq'])
		u_w = weighted_avg(selected['u'], selected['eu'])
		eq_w = weighted_error(selected['eq'])
		eu_w = weighted_error(selected['eu'])
		
		if abs(q_w - q_av) > 2*eq_w or abs(u_w - u_av) > 2*eu_w:
			last_star = test_segment.iloc[i-1]
			print(last_star)
			
			delta_indx = i 
			return delta_indx
	return None  

## 7b. Computing systematic uncertainty in q and u of the identified layers ####
def sys_unc_calc(df, plx_clouds, e_plx_clouds, flag, nth_layer, res_hybrid):
	final_values = []  
	for k in range(len(plx_clouds)):
		plx_diff = np.abs(df['plx'] - plx_clouds[k])
		cl_indx = np.argmin(plx_diff)
		final_values.append(cl_indx)
	wh_C =1
	while(wh_C == 1):
		left_uncertainties = []
		right_uncertainties = []
		plx_clouds_remaining = []
		e_plx_clouds_remaining = []
		final_values_remaining = []
		flag_remaining=flag
		if res_hybrid is not None :
			prev_break = final_values[nth_layer-1]+1
			start_layer = nth_layer
		else:
			prev_break = min(df.index)
			start_layer = 0
		for i in range(int(start_layer), len(final_values)):
			current_break = final_values[i]
			segment1 = df.iloc[prev_break : int(np.round(current_break))]
			segment1['distance'] = abs(segment1.index - final_values[i])
			if(i+1 < len(final_values)):
				segment2 = df.iloc[int(np.round(current_break))+1 : int(np.round(final_values[i+1]))]
			else:
				segment2 = df.iloc[int(np.round(current_break))+1 : max(df.index)]
			segment2['distance'] = abs(segment2.index - final_values[i])
			left_error =  process_segment(segment1, segment2)
			if((i == 0)): 
				left_error = 0	
			if((i !=0) & (left_error != None)):
				if(left_error >= len(segment1)-1 ):
					left_error = None
			right_error =  process_segment(segment2, segment1)
			if((i == 0)):
				right_error =  process_segment(segment2, segment1, q_av = 0, u_av = 0)
			if right_error == None and i+1 < len(final_values):
				flag_remaining[i+1] = 0
			if left_error is not None and right_error is not None:
				final_values_remaining.append(current_break)
				e_plx_clouds_remaining.append(e_plx_clouds[i])
				plx_clouds_remaining.append(plx_clouds[i])
				left_uncertainties.append(left_error)
				right_uncertainties.append(right_error)
				#flag_remaining.append(flag[i])
				prev_break = int(np.round(current_break))+1
			else:
				prev_break = prev_break
		if(len(final_values)-nth_layer > len(final_values_remaining)):
			wh_C = 1
		else:
			wh_C = 0
		mask_flag = np.isin(final_values, final_values_remaining)
		flag_remaining = np.array(flag_remaining)[mask_flag]
		if(res_hybrid is not None):
			final_values = final_values[:nth_layer] + final_values_remaining
			plx_clouds = list(plx_clouds[:nth_layer])+ plx_clouds_remaining
			e_plx_clouds =list(e_plx_clouds[:nth_layer])+ e_plx_clouds_remaining
			flag = list(flag[:nth_layer])+list(flag_remaining)
		else:
			final_values = final_values_remaining
			plx_clouds =  plx_clouds_remaining
			e_plx_clouds = e_plx_clouds_remaining
			flag = flag_remaining
	final_values = np.array(final_values, dtype=object)
	plx_clouds =  np.array(plx_clouds, dtype=object)
	e_plx_clouds =  np.array(e_plx_clouds, dtype=object)
	flag = np.array(flag, dtype=object)
	left_uncertainties = np.array(left_uncertainties, dtype=object)
	right_uncertainties = np.array(right_uncertainties, dtype=object)	
	if res_hybrid is None:	
		sys_left = final_values.astype(int) - np.array(left_uncertainties).astype(int)
		sys_right = final_values.astype(int) + np.array(right_uncertainties).astype(int)
		sys_left_bound = (df['plx'].iloc[sys_left]).values 
		sys_right_bound = (df['plx'].iloc[sys_right]).values 
	else:	
		final_values = np.array(final_values[nth_layer:], dtype=object)
		sys_left = final_values.astype(int) - np.array(left_uncertainties).astype(int)	
		sys_right = final_values.astype(int) + np.array(right_uncertainties).astype(int)
		sys_left_bound = (df['plx'].iloc[sys_left]).values 
		sys_right_bound = (df['plx'].iloc[sys_right]).values 
		hybrid_left = res_hybrid.loc['sys_plx_left'].values[:nth_layer]
		hybrid_right = res_hybrid.loc['sys_plx_right'].values[:nth_layer]
		sys_left_bound = np.concatenate([hybrid_left, sys_left_bound])
		sys_right_bound = np.concatenate([hybrid_right, sys_right_bound])
	sys_left_err = sys_left_bound - plx_clouds
	sys_right_err = plx_clouds - sys_right_bound
	return(plx_clouds, sys_left_err, sys_right_err, e_plx_clouds,flag)

#########################################################################################################

##### 8. Merging the layers within 1-sigma and redefining their flags. 
 
def merge_asymm(arr, left_err, right_err, plx_err, flag, threshold, min_value = None):
    """
    Merge breakpoints using weighted average and uncertainty propagation.
    """
    merged = []
    merged_left_err = []
    merged_right_err = []
    merged_plx_err = []
    merged_flag = []
    i = 0
    while i < len(arr) - 1:
        x1, sigma_left1, sigma_right1, sigma_plx1 = arr[i], left_err[i], right_err[i], plx_err[i]
        x2, sigma_left2, sigma_right2, sigma_plx2 = arr[i+1], left_err[i+1], right_err[i+1], plx_err[i+1]
        left_err1 = max(sigma_left1, sigma_plx1)
        right_err1 = max(sigma_right1, sigma_plx1)
        left_err2 = max(sigma_left2, sigma_plx2)
        right_err2 = max(sigma_right2, sigma_plx2)
        Flag = np.array([flag[i], flag[i+1] ])
        if i == 0 and min_value is not None and x1 > min_value:
        	merged.append(x1)
        	merged_left_err.append(sigma_left1)
        	merged_right_err.append(sigma_right1)
        	merged_plx_err.append(sigma_plx1)
        	merged_flag.append(flag[i])
        	i += 1
        	continue
        # Condition to merge
        if (x1 - threshold*right_err1) <= (x2 + threshold * left_err2):
            # Compute weights
            w1 = 1 / (right_err1)**2
            w2 = 1 / (left_err2)**2
            # Weighted average position
            new_x = (x1 * w1 + x2 * w2) / (w1 + w2)
            new_sigma_left = np.sqrt(1 / (1 / sigma_left1**2 + 1 / sigma_left2**2))
            new_sigma_right = np.sqrt(1 / (1 / sigma_right1**2 + 1 / sigma_right2**2))
            new_sigma_plx = np.sqrt(1 / (1 / sigma_plx1**2 + 1 / sigma_plx2**2))
            merged_flag.append(10 if flag[i] != flag[i + 1] else 0)
            merged.append(new_x)
            merged_left_err.append(new_sigma_left)
            merged_right_err.append(new_sigma_right)
            merged_plx_err.append(new_sigma_plx)
            i += 2  
        else:
            merged.append(arr[i])
            merged_left_err.append(left_err[i])
            merged_right_err.append(right_err[i])
            merged_plx_err.append(plx_err[i])
            merged_flag.append(flag[i])
            i += 1
    if i == len(arr) - 1:
        merged.append(arr[i])
        merged_left_err.append(left_err[i])
        merged_right_err.append(right_err[i])
        merged_plx_err.append(plx_err[i])
        merged_flag.append(flag[i])
    return merged, merged_left_err, merged_right_err, merged_plx_err, merged_flag

##############################################################################################################

# 9. Computing polarization properties of each layer

def compute_cloud_polarization(star_data, plx_cloud, e_plx_cloud, sys_left, sys_right, flag, nth_layer, res_hybrid):
    stat_left = plx_cloud+e_plx_cloud
    stat_right =  plx_cloud-e_plx_cloud
    num_clouds = len(sys_left)
    #segment the data based on cloud distance and its uncertainties 
    segment = []
    segment.append(star_data[star_data['plx'] >=  max(sys_left[0], stat_left[0])])
    for i in range(1, num_clouds):
    	segment.append(star_data[(star_data['plx'] < min(sys_right[i-1], stat_right[i-1])) & (star_data['plx'] >= max(sys_left[i], stat_left[i]))])
    segment.append(star_data[star_data['plx'] <  min(sys_right[-1], stat_right[-1])])
    
    segment_means = []
    segment_errors = []
    for j, seg in enumerate(segment):
    	if((j == 0) & (seg.empty)):
    			means = np.zeros(2)
    			err_means = np.zeros((2,2))
    	elif((j < nth_layer) & (seg.empty) & (res_hybrid is not None)):
    		q_mean = res_hybrid.loc['qC'].values[j-1]
    		u_mean = res_hybrid.loc['uC'].values[j-1]
    		eq_mean = res_hybrid.loc['eqC'].values[j-1]
    		eu_mean = res_hybrid.loc['euC'].values[j-1]
    		Cqu_mean = res_hybrid.loc['Cqu'].values[j-1]
    		means = np.array((q_mean, u_mean))
    		err_means =  np.array([
    					[eq_mean**2,     Cqu_mean],
    					[Cqu_mean,       eu_mean**2]
    					])
    		
    	elif((j< nth_layer) & (res_hybrid is None)):
    		means = np.zeros(2)
    		err_means = np.zeros((2,2))
    	elif np.any(seg):
    		X_seg, Cov_seg = extract_components(seg)
    		means, err_means = weighted_mean_err(X_seg, Cov_seg)
    		print(means, err_means)
    	else:
    		print('SOMETHING IS WRONG. CHECK THE LAYERS')
    		#q_mean, u_mean = 0,0
    		#eq_mean, eu_mean = 0,0
    	segment_means.append(means)
    	segment_errors.append(err_means)
    results = {}
    for cloud_idx in range(num_clouds):
    	qC = segment_means[cloud_idx + 1][0] - segment_means[cloud_idx][0]
    	uC = segment_means[cloud_idx + 1][1] - segment_means[cloud_idx][1]
    	cov_C = segment_errors[cloud_idx + 1] + segment_errors[cloud_idx]
    	eqC, euC = np.sqrt(np.diagonal(cov_C))
    	cov_qu = 0.5*(cov_C[0,1] + cov_C[1,0])
    	results[f'cloud_{cloud_idx}'] = {
    		'plx_calc' : plx_cloud[cloud_idx],
    		'stat_uncertainty': e_plx_cloud[cloud_idx],
        	'stats_plx_left': stat_left[cloud_idx],
        	'stats_plx_right': stat_right[cloud_idx],
        	'sys_plx_left': sys_left[cloud_idx],
        	'sys_plx_right': sys_right[cloud_idx],
            'qC': qC,
            'eqC': eqC,
            'uC': uC,
            'euC': euC,
            'Cqu': cov_qu,
            'pol': np.sqrt(qC**2 + uC**2),
            'epol': np.sqrt( ((qC**2 * eqC**2) + (uC**2 * euC**2) + (2*qC*uC*cov_qu) )/ (qC**2 + uC**2) ),    
            'PA': 0.5*np.arctan2(uC, qC)* 180 / np.pi,
            'ePA': 0.5*(np.sqrt( uC**2 * eqC**2 + qC**2 * euC**2 - 2*qC*uC*cov_qu)/(qC**2 + uC**2))*180/np.pi,
            'flag': flag[cloud_idx] 
            }
    return(results)

###############################################################################################################
#
# 10. optional: plotting and saving the results #####

def plot_qumu(df, df_output, outfile, nth_layer, res_hybrid):
	sys_mu_left = 5*np.log10(1000/df_output.loc['sys_plx_left'].values) - 5
	sys_mu_right =  5*np.log10(1000/df_output.loc['sys_plx_right'].values) - 5
	stats_mu_left = 5*np.log10(1000/df_output.loc['stats_plx_left'].values) - 5
	stats_mu_right = 5*np.log10(1000/df_output.loc['stats_plx_right'].values) - 5
	mu_final = 5*np.log10(1000/df_output.loc['plx_calc'].values) - 5
	Nc = df_output.shape[1]
	plt.figure(figsize=(10, 6))
	plt.errorbar(df['mu'], df['q'], xerr = df['s_mu'],  yerr = df['eq'], marker = 'o', color = 'green', ecolor = 'green', fmt='--', ls='none', capsize = 1, alpha = 0.25)
	plt.errorbar(df['mu'], df['u'],xerr= df['s_mu'], yerr = df['eu'], marker = 'o', color = 'blue', ecolor = 'blue', fmt='--', ls='none', capsize = 1, alpha = 0.25)
	for i, col in enumerate(df_output.columns):
		plt.axvline(x = mu_final[i], color = 'm', ls = '-',label = 'estimated')
		plt.fill_betweenx(y=[min(df['q'].min(),df['u'].min()), max(df['q'].max(), df['u'].max())], 
		x1= stats_mu_left[i], x2= stats_mu_right[i], color='red', alpha=0.1)
		
		plt.fill_betweenx(y=[min(df['q'].min(),df['u'].min()), max(df['q'].max(), df['u'].max())], 
		x1= sys_mu_left[i], x2= sys_mu_right[i], color='k', alpha=0.1)
	
	seg = df[df['mu'] <=  min(sys_mu_left[0], stats_mu_left[0])]
	if np.any(seg):
		prev_q = np.average(seg['q'] , weights=1 / seg['eq']**2)
		prev_u = np.average(seg['u'] , weights=1 / seg['eu']**2)
		prev_eq = weighted_error(seg['eq'])
		prev_eu = weighted_error(seg['eu'])
		xmin = df['mu'].min()
		if((res_hybrid is None) & (nth_layer !=0) & len(seg)==0): 
			prev_q,prev_u, prev_eq, prev_eu  = 0, 0,  0, 0
			xmin = df['mu'].min()
	else:
		prev_q, prev_u, prev_eq, prev_eu  = 0, 0,  0, 0
		xmin = min(sys_mu_left[0],stats_mu_left[0]) -1
	plt.hlines(y=prev_u, xmin=xmin, xmax=min(sys_mu_left[0],stats_mu_left[0]), color='blue', linewidth=3, linestyle='-')
	plt.hlines(y=prev_q, xmin=xmin, xmax=min(sys_mu_left[0],stats_mu_left[0]), color='green', linewidth=3, linestyle='-')
	plt.fill_between(x = [xmin,  min(sys_mu_left[0], stats_mu_left[0])],y1= prev_q - prev_eq, y2= prev_q + prev_eq, color='green', alpha=0.1)
	plt.fill_between(x = [xmin,  min(sys_mu_left[0], stats_mu_left[0])],y1= prev_u - prev_eu, y2= prev_u + prev_eu, color='blue', alpha=0.1)
	for j in range(Nc-1):
		seg_q = df_output.loc['qC', 'cloud_'+str(j)] + prev_q
		seg_u = df_output.loc['uC', 'cloud_'+str(j)] + prev_u
		seg_eq = np.sqrt(df_output.loc['eqC', 'cloud_'+str(j)]**2 + prev_eq**2)
		seg_eu = np.sqrt(df_output.loc['euC', 'cloud_'+str(j)]**2 + prev_eu**2)
		
		plt.hlines(y=seg_q, 
		xmax=min(sys_mu_left[j+1], stats_mu_left[j+1]) , 
		xmin = max(sys_mu_right[j], stats_mu_right[j]), 
		color='green', linewidth=3, linestyle='-')
		
		plt.hlines(y=seg_u, 
		xmax=min(sys_mu_left[j+1], stats_mu_left[j+1])  , 
		xmin = max(sys_mu_right[j], stats_mu_right[j]), 
		color='blue', linewidth=3, linestyle='-')
		
		plt.fill_between(x = [ max(sys_mu_right[j], stats_mu_right[j]), min(sys_mu_left[j+1], stats_mu_left[j+1])],
		y1= seg_q - seg_eq, y2= seg_q + seg_eq, color='green', alpha=0.1)
		
		plt.fill_between(x= [ max(sys_mu_right[j], stats_mu_right[j]), min(sys_mu_left[j+1], stats_mu_left[j+1])],
		y1= seg_u - seg_eu, y2= seg_u + seg_eu, color='blue', alpha=0.1)
		prev_q = seg_q
		prev_u = seg_u
		prev_eq = seg_eq
		prev_eu = seg_eu
	
	plt.fill_between(x = [ max(sys_mu_right[Nc-1], stats_mu_right[Nc-1]), df['mu'].max()], 
	y1 =  (df_output.loc['qC', 'cloud_'+str(Nc-1)] + prev_q) - np.sqrt(df_output.loc['eqC', 'cloud_'+str(Nc-1)]**2 + prev_eq**2), 
	y2 = (df_output.loc['qC','cloud_'+str(Nc-1)] + prev_q) + np.sqrt(df_output.loc['eqC', 'cloud_'+str(Nc-1)]**2 + prev_eq**2),
	color='green', alpha=0.1)
	plt.hlines(y=df_output.loc['qC', 'cloud_'+str(Nc-1)] + prev_q, 
		xmin= max(sys_mu_right[Nc-1], stats_mu_right[Nc-1]) , 
		xmax = df['mu'].max(), 
		color='green', linewidth=3, linestyle='-')
	plt.hlines(y=df_output.loc['uC', 'cloud_'+str(Nc-1)] + prev_u, 
		xmin= max(sys_mu_right[Nc-1], stats_mu_right[Nc-1]) , 
		xmax = df['mu'].max(),
		color='blue', linewidth=3, linestyle='-')
	plt.fill_between(x = [ max(sys_mu_right[Nc-1], stats_mu_right[Nc-1]), df['mu'].max()], 
	y1 =  (df_output.loc['uC', 'cloud_'+str(Nc-1)] + prev_u) - np.sqrt(df_output.loc['euC', 'cloud_'+str(Nc-1)]**2 + prev_eu**2), 
	y2 = (df_output.loc['uC', 'cloud_'+str(Nc-1)] + prev_u) + np.sqrt(df_output.loc['euC', 'cloud_'+str(Nc-1)]**2 + prev_eu**2),
	color='blue', alpha=0.1)
	plt.xlabel('mu')
	plt.ylabel('q / u')
	plt.legend()
	plt.savefig(outfile+'_trishul.png', dpi=300)
	plt.show()
	return()
	



