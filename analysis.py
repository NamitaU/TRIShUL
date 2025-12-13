import numpy as np
from astropy.io import ascii
from astropy.table import Table
#import matplotlib.pyplot as plt
import pandas as pd
import Trishul as T
import os, fnmatch


path2cat = '/Users/namita/Documents/PDF/IA_FORTH'
spath = path2cat
hybrid = ['sample_2C.csv']
F = hybrid[0]
pixID = F[-11:-4]
res_hybrid = None #pd.read_csv(os.path.join(path2hybrid, F_res_hybrid), index_col=0) #None
print('Filename = ', F)
data = ascii.read(os.path.join(path2cat, F),delimiter=',')
q = data['q_obs']*100
u = data['u_obs']*100
eq = data['s_q']*100
eu = data['s_u']*100
plx = data['plx']
eplx = data['s_plx']
dis = 1000/plx
edis = dis**2 * (eplx/1000)
mu = 5*np.log10(dis) - 5
s_mu = 5*edis/(2.303*dis) # derivative of ln x = 1/x not log x = ln(x)/2.303 = 1/(2.303*x)
GPA = 0.5 * np.arctan2(u, q) * 180 / np.pi
pol = np.sqrt(q**2 + u**2)
indx = np.where(GPA < 0)[0]
GPA[indx] = GPA[indx] + 180
Cqq = eq**2
Cuu = eu**2
if 'c_qu' in data.columns:
	Cqu = data['c_qu']*(10**4)
else:
	Cqu = 0*Cqq
detC = Cqq*Cuu - Cqu**2
invCqq = Cuu/detC
invCuu = Cqq/detC
invCqu = - Cqu/detC
dMaha = np.sqrt(q**2 * invCqq + 2*q*u*invCqu + u**2 * invCuu)
DATA = {'plx': plx, 'eplx': eplx, 'mu': mu,'s_mu':s_mu, 'dis': dis, 'p': pol, 'gpa': GPA, 'q': q, 'u': u, 'eu': eu, 'eq': eq, 'Cqu': Cqu, 'dMaha': dMaha}
df = pd.DataFrame(DATA)
df = df.sort_values(by='dis').reset_index(drop=True)

sc = 5/len(df)
mu_clouds, e_mu_clouds, e_mu_lower, e_mu_upper, flag, nth_layer = T.trisul_run(df, param = 'dMaha', res_hybrid = res_hybrid)
				
				
print('#################################################### \n ')
print('\n ################################################## \n')
print('range of breaks detected in p = ', mu_clouds, e_mu_clouds, flag)
print('#################################################### \n ')
print('\n #################################################### \n ')
				
if(len(mu_clouds) == 0):
	output_df = pd.DataFrame()
	file_path = spath+ '/result_'+F[:-4]+'.csv'
	with open(file_path, 'w') as f:
		f.write("sightline not resolved")  # Writing the comment line
		output_df.to_csv(f, index=True)

indx_clouds, e_indx_clouds = mu_clouds, e_mu_clouds			
neg_indx = np.where(indx_clouds-e_indx_clouds < 0)[0]
e_indx_clouds[neg_indx] = 0
av_plx_err = [df.loc[l:u, 'eplx'].median() for l, u in zip(np.floor(indx_clouds-e_mu_lower), np.ceil(indx_clouds+e_mu_upper))]
plx_upper =  df['plx'].iloc[np.ceil(indx_clouds+e_mu_upper)].values
plx_lower = df['plx'].iloc[np.floor(indx_clouds-e_mu_lower)].values
e_plx_clouds = np.sqrt( ((plx_lower - plx_upper)/2)**2 + np.array(av_plx_err)**2)
plx_clouds = df['plx'].iloc[np.floor(indx_clouds)].values #(plx_upper + plx_lower)/2
				
if res_hybrid is not None:
	plx_clouds[:nth_layer] = res_hybrid.loc['plx_calc'].values[:nth_layer]
	e_plx_clouds[:nth_layer] = res_hybrid.loc['stat_uncertainty'].values[:nth_layer]
	flag[:nth_layer] = res_hybrid.loc['flag'].values[:nth_layer]

plx_clouds, sys_left_err, sys_right_err, e_plx_clouds,flag = T.sys_unc_calc(df, plx_clouds, e_plx_clouds, flag, nth_layer, res_hybrid)
plx_clouds_f, sys_left_err_f, sys_right_err_f, e_plx_clouds_f, flag_f =  T.merge_asymm(plx_clouds, sys_left_err, sys_right_err, e_plx_clouds,flag, 2, min_value = 6)
sys_left_bound_f = np.add(plx_clouds_f, sys_left_err_f)
sys_right_bound_f = np.subtract(plx_clouds_f, sys_right_err_f)
################################# calculating qs and us #####################################
results_brkpt = T.compute_cloud_polarization(df, np.array(plx_clouds_f), np.array(e_plx_clouds_f), np.array(sys_left_bound_f), np.array(sys_right_bound_f), np.array(flag_f), nth_layer, res_hybrid)
outfile = os.path.join(spath, F[:-4])
output_df = pd.DataFrame(results_brkpt)
print(output_df)
T.plot_qumu(df, output_df, outfile = outfile, nth_layer = nth_layer, res_hybrid=res_hybrid)	
file_path = spath+ '/result_'+F[:-4]+'.csv'
with open(file_path, 'w') as f:
	output_df.to_csv(f, index=True) 
				
				