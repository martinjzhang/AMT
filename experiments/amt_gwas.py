## system settings
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import logging
import os
import sys
import argparse
import amt.method as md
import amt.data_loader as dl
import time
import matplotlib.pyplot as plt
import pickle

def main(args):
    title = args.title
    alpha = float(args.alpha)
    rs = int(args.rs)
    print('Title:%s, alpha=%0.2f, random_state=%d'%(title, alpha, rs))
    
    np.random.seed(0)
    print('# Loading Parkinson data')
    with open('../../parkinsons/parkinsons.pickle', 'rb') as f:
        X = pickle.load(f)
        y = pickle.load(f)
        miss_prop = pickle.load(f)
    ind_sample = (miss_prop<0.05)
    X = X[:, ind_sample]
    y = y-1
    file_map = '../../parkinsons/parkinsons.map'
    df_map = pd.read_csv(file_map, delimiter='\t', 
                         names=['chromosome', 'snp', 'start', 'end'])
    n_sample, n_snp = X.shape
    n_hypothesis = n_snp
    
    # Compute the expected observations
    print('# Compute statistics')
    Exp = np.zeros([8, n_snp], dtype=float)
    for iy in range(2):
        for ix in range(4):
            Exp[iy*4+ix,:] = np.mean(y==iy) * np.mean(X==ix,axis=0)
    Exp = Exp*n_sample
    r_Exp = 1/Exp.clip(min=1e-6)*(Exp>0)
    chi2_obs = md.compute_chi2(y, X, Exp, r_Exp)
    data_gwas = {'X':X, 'y':y, 'Exp':Exp, 'r_Exp':r_Exp, 'chi2_obs':chi2_obs}
    
    # # Small data
    # ind_small = np.zeros([n_snp], dtype=bool)
    # # alpha = 0.1
    # ind_small[0:1000] = True
    # n_hypothesis = np.sum(ind_small)
    # data_gwas = {'X':X[:,ind_small], 'y':y, 'Exp':Exp[:,ind_small],
    #              'r_Exp':r_Exp[:,ind_small], 'chi2_obs':chi2_obs[ind_small]}
    
    print('# Runing AMT')
    output_folder = '../../results/GWAS'
    start_time = time.time()
    n_fMC = int(n_hypothesis*100)
    p_hat_ub, p_hat_lb, p_hat, tau_hat, n_amt = md.amt(md.f_sample_chi2, data_gwas, n_hypothesis,
                                             alpha=alpha, n_fMC=n_fMC,
                                             verbose=False, delta=0.001,
                                             output_folder = output_folder, title=title,
                                             random_state=rs)
    h_amt = (p_hat_ub <= tau_hat)
    print('# AMT: avg. MC samples = %0.1f, time=%0.2fs'%(np.mean(n_amt),
                                                         time.time()-start_time))
    # print('# D_hat=%d, D_overlap=%d, D_full=%d'%(md.result_compare(h_amt, h_fmc)))
    res_AMT = {'p_hat_ub':p_hat_ub, 'p_hat_lb':p_hat_lb,
               'p_hat':p_hat, 'tau_hat':tau_hat, 
               'n_amt':n_amt, 'h_amt':h_amt}
    output_file = output_folder + '/GWAS_amt%s.pickle'%title
    with open(output_file, "wb") as f:
        pickle.dump(res_AMT, f)                 
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMT')
    parser.add_argument('-t', '--title', type=str, required=True)
    parser.add_argument('-a', '--alpha', type=str, required = True)
    parser.add_argument('-r', '--rs', type=str, required = True)
    # parser.add_argument('-d', '--data_loader', type=str, required=False)
    # parser.add_argument('-n', '--data_name', type=str, required=False)
    args = parser.parse_args()
    # main(args)
    main(args)