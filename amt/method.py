import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time
from statsmodels.stats.proportion import proportion_confint
import multiprocessing as mp
import pickle
    
def get_monte_carlo_sample(p, n_permutation=5000, random_state=0):
    """ Generate monte carlo samples given the p-value.
        The samples are binary indicating if an extreme event occurs
    """
    # Full Monte Carlo permutation
    np.random.seed(random_state)
    n_feature = p.shape[0]
    U = np.random.binomial(1, p, size=(n_permutation, n_feature))
    p_fmc = (np.sum(U, axis=0)+1)/(n_permutation+1)
    t_fmc = np.ones([n_feature], dtype=int) * n_permutation
    return U, p_fmc, t_fmc

"""
    Adaptive Monte Carlo test
"""
def amt(f_sample, data, n_hypothesis, n_fMC=None, alpha=0.1, increment=1.1,
        batch_size_start=100, delta=None, output_folder=None, verbose=False, title='',
        random_state=0, output_data_folder=None, n_core=2):
    """ Adaptive Monte Carlo test via multi-armed bandits.
    Args:
        f_sample (f(data,n_sample,sample_start)): the function to acquire MC samples.
        data: the data to generate MC samples, input for f_sample.
        n_hypothesis (int): the number of hypotheses.
        n_fMC (int): the maximum number of MC samples for each hypothesis.
        alpha (float): the nominal FDR level. 
        increment (float, >1): the multiplicative increment of the batch sizes.
        delta (float): the error probability.
        output_folder (str): the folder to output intermediate results. 
        verbose (boolean): if to output some intermediate results. 
      
    Returns:
        x_order ((d,) ndarray): the order (of x_val) from smallest alt/null ratio to 
            the largest.
    """
    if output_folder is not None:
        fout = open(output_folder+'/result_amt%s'%title, 'w')
    start_time = time.time()
    # Initialization
    if delta is None:
        delta = 1 / n_hypothesis
    delta_CI = delta/n_hypothesis/np.log(n_fMC)
    # delta_CI = delta
    ind_sample = np.ones([n_hypothesis], dtype=bool)
    p_hat_ub = np.ones([n_hypothesis], dtype=float)
    p_hat_lb = np.zeros([n_hypothesis], dtype=float)    
    r_hat = n_hypothesis
    tau_hat = r_hat/n_hypothesis*alpha    
    # n_amt: number of MC samples
    n_amt = np.zeros([n_hypothesis], dtype=int)
    # n_amt: number of MC sample batches
    n_amt_batch = np.zeros([n_hypothesis], dtype=int)
    # s_amt: number of successes
    s_amt = np.zeros([n_hypothesis], dtype=int)
    # Set the batch size
    n_batch = np.ceil(np.log(1 - n_fMC/batch_size_start*(1-increment))
                   / np.log(increment))
    batch_size = np.ceil(batch_size_start * increment**(np.arange(n_batch))).astype(int)
    batch_size[-1] = n_fMC - np.sum(batch_size[:-1])
    if verbose:
        print('# Initialization parameters')
        print('# n_hypothesis=%d, n_fMC=%d, alpha=%0.2f, increment=%0.2f'%
              (n_hypothesis, n_fMC, alpha, increment))
        print('# delta', delta)
        print('# delta_CI', delta_CI)
        print('# r_hat=%d, tau_hat=%0.4f'%(r_hat, tau_hat))
        print('# batch_size', batch_size)
        print('# sum of batch size = %d'%np.sum(batch_size))
        print('# Initialization completed: time=%0.3fs'%(start_time - time.time())) 
    if output_folder is not None:
        fout.write('# Initialization parameters\n')
        fout.write('# n_hypothesis=%d, n_fMC=%d, alpha=%0.2f, increment=%0.2f\n'%
                   (n_hypothesis, n_fMC, alpha, increment))
        fout.write('# delta=%0.4f\n'%delta)
        fout.write('# delta_CI=%0.4f\n'%delta_CI)
        fout.write('# r_hat=%d, tau_hat=%0.4f\n'%(r_hat, tau_hat))
        fout.write('# batch_size: %s\n'%batch_size)
        fout.write('# sum of batch size = %d\n'%np.sum(batch_size))
        fout.write('# Initialization completed: time=%0.3fs\n\n'%(start_time - time.time()))
    if output_data_folder is not None:
        res_dic = {'p_hat_ub':p_hat_ub, 'p_hat_lb':p_hat_lb, 
                   'n_amt':n_amt, 's_amt': s_amt, 'ind_sample':ind_sample,
                   'r_hat':r_hat, 'tau_hat':tau_hat}
        with open(output_data_folder+'/init.pickle', "wb") as f:
            pickle.dump(res_dic, f)         
    i_itr=0
    while True:
        # Record the data
        if output_data_folder is not None:
            res_dic = {'p_hat_ub':p_hat_ub, 'p_hat_lb':p_hat_lb, 
                       'n_amt':n_amt, 's_amt': s_amt, 'ind_sample':ind_sample,
                       'r_hat':r_hat, 'tau_hat':tau_hat}
            with open(output_data_folder+'/itr%d.pickle'%i_itr, "wb") as f:
                pickle.dump(res_dic, f)
        # Sample
        temp_n_amt_new = batch_size[n_amt_batch[ind_sample]]
        s_amt[ind_sample] = s_amt[ind_sample] + f_sample(data, ind_sample, 
                                                         temp_n_amt_new, 
                                                         n_amt[ind_sample],
                                                         random_state=random_state*1000+i_itr,
                                                         n_core=n_core)
        n_amt_batch[ind_sample] += 1 
        n_amt[ind_sample] = n_amt[ind_sample] + temp_n_amt_new                     
        # Update CIs                
        p_hat_lb[ind_sample],p_hat_ub[ind_sample] = update_p(s_amt[ind_sample], 
                                                             n_amt[ind_sample],
                                                             n_fMC,
                                                             extreme_prob=delta_CI)
        # Update estimates
        n_g = np.sum(p_hat_lb > tau_hat)
        while (r_hat > n_hypothesis - n_g):
            r_hat = n_hypothesis - n_g
            tau_hat = r_hat/n_hypothesis*alpha
            n_g = np.sum(p_hat_lb > tau_hat)
        n_l = np.sum(p_hat_ub <= tau_hat)
        ind_sample = (p_hat_lb <= tau_hat) & (p_hat_ub > tau_hat)\
                     & (p_hat_lb != p_hat_ub)
        # Record
        n_u = n_hypothesis - n_g - n_l
        if output_folder is not None:
            fout.write('# avg_sample=%0.1f, tau_hat=%0.5f, r_hat=%d, n_u=%d, n_g=%d, n_l=%d, '
                        %(np.mean(n_amt), tau_hat, r_hat, n_u, n_g, n_l))
            fout.write('time=%0.1fs\n'%(time.time()-start_time))
        if verbose:            
            print('# %d, avg_sample=%0.1f, tau=%0.5f, r_hat=%d, n_u=%d, n_g=%d, n_l=%d'\
                  %(i_itr, np.mean(n_amt), tau_hat, r_hat, n_u, n_g, n_l))
        i_itr = i_itr+1
        # Exit condition
        if np.sum(ind_sample)==0:
            break
    if output_folder is not None:
        fout.write('\n# AMT finished\n')
        fout.write('# avg_sample=%0.1f, tau_hat=%0.5f, r_hat=%d, n_u=%d, n_g=%d, n_l=%d, '
                   %(np.mean(n_amt), tau_hat, r_hat, n_u, n_g, n_l))
        fout.write('time=%0.1fs\n'%(time.time()-start_time))
        fout.write('# discoveries=%d'%(np.sum(p_hat_ub<=tau_hat)))
        fout.close()
    if output_data_folder is not None:
        res_dic = {'p_hat_ub':p_hat_ub, 'p_hat_lb':p_hat_lb, 
                   'n_amt':n_amt, 's_amt': s_amt, 'ind_sample':ind_sample,
                   'r_hat':r_hat, 'tau_hat':tau_hat}
        with open(output_data_folder+'/final.pickle', "wb") as f:
            pickle.dump(res_dic, f)
    p_hat = s_amt/n_amt
    return p_hat_ub, p_hat_lb, p_hat, tau_hat, n_amt

"""
    Sample dummy (this part should be parallelized for large-scale problems)
""" 
def f_sample_dummy(data, ind_sample, n_new_sample, sample_start=None,
                   random_state=None, n_core=2):
    """ Adaptive Monte Carlo test via multi-armed bandits.
    Args:
        data ((n,m) NumPy boolean array): the fMC samples.
        ind_sample ((m,) NumPy boolean array): the hypothesis that need new samples
        n_new_sample ((n_active,) NumPy int array): the number of fMC samples for each hypothesis.
        sample_start ((n_active,) NumPy int array): the number of fMC samples each hypothesis
            has so far.
      
    Returns:
        n_success ((m,) NumPy int array): the number of successes for each hypothesis.
    """ 
    n,m = data.shape
    n_active = np.sum(ind_sample)
    n_success = np.zeros([n_active], dtype=int)
    if sample_start is None:
        sample_start = np.zeros([n_active], dtype=int)
    sample_end = sample_start + n_new_sample
    for i,i_origin in enumerate(np.arange(m)[ind_sample]):
        n_success[i] = np.sum(data[sample_start[i]:sample_end[i], i_origin])
    return n_success

"""
    f_sample for GWAS
"""
def f_sample_chi2(data, ind_sample, n_new_sample, sample_start=None, n_core=2,
                  random_state=None):
    X = data['X'][:,ind_sample]
    y = data['y']
    Exp = data['Exp'][:, ind_sample]
    r_Exp = data['r_Exp'][:, ind_sample]
    chi2_obs = data['chi2_obs'][ind_sample]  
    n_active = np.sum(ind_sample)
    n_success = np.zeros([n_active], dtype=int)
    n_permute = np.max(n_new_sample) # Parallelization
    # Compute the MC samples 
    if n_core==1:
        B = permute_chi2_batch(y, X, Exp, r_Exp, chi2_obs, n_permute)
    else:
        Y_input = []
        n_permute_core = int(n_permute/n_core)
        for i_core in range(n_core):
            if i_core != n_core-1:
                Y_input.append([y, X, Exp, r_Exp, chi2_obs, n_permute_core,
                                random_state*n_core+i_core])
            else:
                Y_input.append([y, X, Exp, r_Exp, chi2_obs, 
                                int(n_permute-(n_core-1)*n_permute_core),
                                random_state*n_core+i_core])
        po = mp.Pool(n_core)
        res = po.map(permute_chi2_batch_wrapper, Y_input)
        po.close()
        B = res[0]
        # print(np.mean(B, axis=0)[0:5])
        for i_core in range(1,n_core):
            # print(np.mean(res[i_core], axis=0)[0:5])
            B = np.concatenate([B, res[i_core]], axis=0)        
    for i in range(n_active):
        n_success[i] = np.sum(B[0:n_new_sample[i], i])
    return n_success

def permute_chi2_batch_ncore(y, X, Exp, r_Exp, chi2_obs, n_permute, random_state=0,
                             verbose=False, n_core=32):
    Y_input = []
    n_permute_core = int(n_permute/n_core)
    for i_core in range(n_core):
        if i_core != n_core-1:
            Y_input.append([y, X, Exp, r_Exp, chi2_obs, n_permute_core,
                            random_state*n_core+i_core])
        else:
            Y_input.append([y, X, Exp, r_Exp, chi2_obs, 
                            int(n_permute-(n_core-1)*n_permute_core),
                            random_state*n_core+i_core])
    po = mp.Pool(n_core)
    res = po.map(permute_chi2_batch_wrapper, Y_input)
    po.close()
    B = res[0]
    for i_core in range(1,n_core):
        B = np.concatenate([B, res[i_core]], axis=0)
    return B

def permute_chi2_batch_wrapper(data):
    return permute_chi2_batch(data[0], data[1], data[2], 
                              data[3], data[4], data[5], 
                              random_state = data[6])

def permute_chi2_batch(y, X, Exp, r_Exp, chi2_obs, n_permute, random_state=None,
                       verbose=False):
    if random_state is not None:
        np.random.seed(random_state)
    start_time = time.time()
    n_feature = X.shape[1]
    B = np.zeros([n_permute, n_feature], dtype=bool)
    for i_permute in range(n_permute):
        y_new = np.random.permutation(y)
        temp_chi2 = compute_chi2(y_new, X, Exp, r_Exp)
        B[i_permute,:] = (temp_chi2>=chi2_obs)
        if verbose:
            if i_permute%1000==0:
                print('%d/%d, time=%0.1fs'%(i_permute, n_permute, time.time()-start_time))
    return B

def compute_chi2(y, X, Exp, r_Exp):
    # start_time = time.time()
    Obs = np.zeros([8, X.shape[1]], dtype=int)
    for iy in range(2):
        temp = X[y==iy,:]
        for ix in range(4):
            Obs[iy*4+ix,:] = np.sum(temp==ix, axis=0)
    chi2 = np.sum((Obs-Exp)**2*r_Exp, axis=0)
    # print('Time=%0.3fs'%(time.time()-start_time))
    return chi2

"""
    Sample correlation (this part should be parallelized for large-scale problems)
    think of this part later ...
""" 
def f_sample_correlation(X, y, ni, ti=None):
    """ Adaptive Monte Carlo test via multi-armed bandits.
    Args:
        X ((n,m) NumPy boolean array): the fMC samples.
        y ((n,m) NumPy boolean array): the fMC samples.
        ni ((m,) NumPy int array): the number of fMC samples for each hypothesis.
        ti ((m,) NumPy int array): the number of fMC samples each hypothesis has so far.
      
    Returns:
        si ((m,) NumPy int array): the number of successes for each hypothesis.
    """
    return si 


"""
    Old implementation
"""
# ada_permute
def amc_test(U, alpha=0.1, step_size=10, extreme_prob=0.01, max_iter = 20000,\
             p_true=None, output_folder=None, verbose=False):
    # Initialization
    n_max, m = U.shape
    ind_u = np.ones([m], dtype=bool)
    p_hat_ub = np.ones([m], dtype=float)
    p_hat_lb = np.zeros([m], dtype=float)    
    r_hat = m
    tau_hat = r_hat/m*alpha    
    # t: number of MC samples
    t = np.zeros([m], dtype=int)
    # s: number of successes
    s = np.zeros([m], dtype=int)
    n_itr = 0
    while True:
        # New Monte Carlo samples
        temp = t[ind_u]
        t[ind_u] = (t[ind_u] + step_size).clip(max=n_max)
        s[ind_u] = s[ind_u] + sample_MC(U[:,ind_u], temp, t[ind_u])                        
        # Update CIs                
        p_hat_lb[ind_u],p_hat_ub[ind_u] = update_p(s[ind_u], t[ind_u], n_max,\
                                                   extreme_prob=extreme_prob)
        # Update estimates
        n_g = np.sum(p_hat_lb > tau_hat)
        while (r_hat > m - n_g):
            r_hat = m - n_g
            tau_hat = r_hat/m*alpha
            n_g = np.sum(p_hat_lb > tau_hat)
        n_l = np.sum(p_hat_ub <= tau_hat)
        ind_u = (p_hat_lb <= tau_hat) & (p_hat_ub > tau_hat)\
                & (p_hat_lb != p_hat_ub)
        if np.sum(ind_u)==0:
            break
        n_itr += 1
        # Record
        if verbose and (n_itr%20==0):
            n_u = m - n_g - n_l
            tau_true = bh(p_true, alpha=alpha)
            print('## n_itr=%d, avg_sample=%0.1f'%(n_itr, np.mean(t)))
            print('r_hat=%d, n_u=%d, n_g=%d, n_l=%d'\
                  %(r_hat, n_u, n_g, n_l))
            print('tau=%0.5f, tau_true=%0.5f'%(tau_hat, tau_true))
            h = (p_true<=tau_true)
            h_hat = (p_hat_ub<=tau_hat)
            print('D_ap=%d, D_overlap=%d, D_true=%d, '\
                  %(result_compare(h_hat, h)))
        if n_itr > max_iter:
            print('## Early exit!')
        #     n_pts = 100
        #     plot_ind = np.argsort(p_true)[:n_pts]
        #     temp_U = ((p_hat_lb <= tau) & (p_hat_ub >=tau))[plot_ind]     
        #     y_val = 0.5*(p_hat_lb[plot_ind] + p_hat_ub[plot_ind])
        #     y_err = p_hat_ub[plot_ind] - y_val  
        #     xaxis = np.arange(n_pts)+1
        #     plt.figure(figsize = [6, 5])
        #     plot_p_sort(p_true, n_pts=n_pts, label='full_mc')
        #     plt.scatter(xaxis[temp_U], p_true[plot_ind][temp_U], color='r', label='uncertain pts')
        #     plt.errorbar(xaxis, y_val, yerr = y_err, alpha=0.6)
        #     plt.ylim([0, 0.03])
        #     plt.plot([1, n_pts], [tau, tau], color='r', label='tau_hat')
        #     plt.legend(loc='upper left')
        #     plt.title('avg_sample=%0.1f, r_hat=%d, |U|=%d, |L|=%d, |S|=%d'\
        #               %(np.mean(t), r_hat, n_uncertain, n_larger, n_smaller))
        #     if output_folder is not None:
        #         figname = 'progress_%d'%n_itr
        #         plt.savefig(output_folder + '/' + figname + '.png')
        #         plt.savefig(output_folder + '/' + figname + '.pdf')
        #     plt.show()
            
        # if n_smaller >= r_hat:
        #     tau_true = bh(p_true, alpha=alpha)
        #     if verbose:
        #         print('## n_itr=%d, avg_sample=%0.1f'%(n_itr, np.mean(t)))
        #         print('r_hat=%d, |U|=%d, |L|=%d, |S|=%d'%(r_hat, n_uncertain, n_larger, n_smaller))
        #         print('tau=%0.5f, tau_true=%0.5f'%(tau, tau_true))
        #         h = (p_true<=tau_true)
        #         h_hat = (p_hat_ub<=tau)
        #         print('D_ap=%d, D_true=%d, D_overlap=%d'\
        #               %(np.sum(h_hat), np.sum(h), np.sum(h_hat*h))) 
        #         n_pts = 100
        #         plot_ind = np.argsort(p_true)[:n_pts]
        #         temp_U = ((p_hat_lb <= tau) & (p_hat_ub >=tau))[plot_ind]     
        #         y_val = 0.5*(p_hat_lb[plot_ind] + p_hat_ub[plot_ind])
        #         y_err = p_hat_ub[plot_ind] - y_val  
        #         xaxis = np.arange(n_pts)+1
        #         plt.figure(figsize = [6, 5])
        #         plot_p_sort(p_true, n_pts=n_pts, label='full_mc')
        #         plt.scatter(xaxis[temp_U], p_true[plot_ind][temp_U], color='r',\
        #                     label='uncertain pts')
        #         plt.errorbar(xaxis, y_val, yerr = y_err, alpha=0.6)
        #         plt.ylim([0, 0.03])
        #         plt.plot([1, n_pts], [tau, tau], color='r', label='tau_hat')
        #         plt.legend(loc='upper left')
        #         plt.title('avg_sample=%0.1f, r_hat=%d, |U|=%d, |L|=%d, |S|=%d'\
        #               %(np.mean(t), r_hat, n_uncertain, n_larger, n_smaller))
        #         if output_folder is not None:
        #             figname = 'progress_%d'%n_itr
        #             plt.savefig(output_folder + '/' + figname + '.png')
        #             plt.savefig(output_folder + '/' + figname + '.pdf')
        #         plt.show()
        #     break 
        
    return p_hat_ub, tau_hat, t

def sample_MC(U, t_start, t_end):
    n_feature = U.shape[1]
    s = np.zeros([n_feature], dtype=int)
    for i_feature in range(n_feature):
        s[i_feature] = np.sum(U[t_start[i_feature]:t_end[i_feature], i_feature])
    return s

# def compute_s(t, U_cumsum):
#     n_feature = U_cumsum.shape[1]
#     s = np.zeros([n_feature], dtype=int)
#     for i_feature in range(n_feature):
#         s[i_feature] = np.sum(U_cumsum[t[i_feature]-1, i_feature])
#     return s
    
def update_p(s, t, n_max, extreme_prob=0.01):
    """ Compute the binomial confidence intervals to update the p-values.
    
    Args:
        s (ndarray): number of successes for each feature.
        t (ndarray): total number of trials for each feature.
        alpha (float): alpha/2 is used for both half_tails.
        
    To-do: 
        Consolidate this part.
    """
    # p_hat_lb, p_hat_ub = proportion_confint(s, t, alpha=extreme_prob, method='beta')
    p_hat_lb, p_hat_ub = proportion_confint(s, t, alpha=extreme_prob, method='agresti_coull')
    # p_hat_lb[np.isnan(p_hat_lb)] = 0
    # p_hat_ub[np.isnan(p_hat_ub)] = 1
    # For points that all sampled are exhausted.
    ind_sample_end = (t==n_max)
    p_hat_lb[ind_sample_end] = (s[ind_sample_end]+1)/(t[ind_sample_end]+1)
    p_hat_ub[ind_sample_end] = (s[ind_sample_end]+1)/(t[ind_sample_end]+1)
    return p_hat_lb, p_hat_ub

"""
    Baseline methods
"""
def ada_permute_oracle(U, p_true, alpha=0.1, step_size=10, option='strong', verbose=False):
    # Set up parameters
    n_permutation, n_feature = U.shape
    t = np.zeros([n_feature], dtype=int)
    p_hat_ub = np.ones([n_feature], dtype=float)
    p_hat_lb = np.zeros([n_feature], dtype=float)
    U_cumsum = np.cumsum(U, axis=0)
    n_itr = 0
    if option == 'strong':
        tau_true = bh(p_true, alpha)
    elif option == 'weak':
        sort_ind = np.argsort(p_true)
        tau_true = np.zeros([n_feature], dtype=float)
        tau_true[sort_ind] = (np.arange(n_feature)+1)/n_feature*alpha
        tau_true = tau_true.clip(min=bh(p_true, alpha))
    while True:
        # Permutation samples
        ind_uncertain = (p_hat_lb <= tau_true) & (p_hat_ub >=tau_true)\
                         & (p_hat_lb != p_hat_ub)
        n_uncertain = np.sum(ind_uncertain)
        t[ind_uncertain] += step_size
        t = t.clip(max=n_permutation)
        # Update parameters
        s = compute_s(t, U_cumsum)
        p_hat_lb, p_hat_ub = update_p(s, t, n_permutation, extreme_prob=0.01)
        if n_itr % 200 == 0 and verbose:            
            print('## n_itr=%d, avg_sample=%0.1f, |U|=%d'%(n_itr, np.mean(t), n_uncertain))
        n_itr += 1
        if n_uncertain == 0:
            break
    return p_hat_ub, t

def sMC(U, r=10):
    """ Sequenctial monte carlo method as described in the 1991 paper.
    """
    n_permutation, n_feature = U.shape
    cumsum_U = np.cumsum(U, axis=0)
    p_sMC = np.zeros([n_feature], dtype=float)
    t_sMC = np.zeros([n_feature], dtype=int)
    # sequential Monte Carlo
    for i_feature in range(n_feature):
        if cumsum_U[-1, i_feature] < r:
            p_sMC[i_feature] = (cumsum_U[-1, i_feature]+1) / (n_permutation+1)
            t_sMC[i_feature] = n_permutation
        else:
            B = np.where(cumsum_U[:, i_feature] == r)[0][0] + 1
            p_sMC[i_feature] = r/B
            t_sMC[i_feature] = B
    return p_sMC, t_sMC
    
def bh(p, alpha=0.1):
    sort_ind = np.argsort(p)
    p = p[sort_ind]
    m = p.shape[0]
    threshold = (np.arange(m)+1)*alpha/m
    if np.sum(p <= threshold) == 0:
        return 0
    else:
        critical_ind = np.where(p <= threshold)[0][-1]
        return threshold[critical_ind]

"""
    Anciliary functions
"""
def plot_p_sort(p, alpha=0.1, sort_ind=None, n_pts=100, label=None):
    m = p.shape[0]
    if sort_ind is None:
        sort_ind = np.argsort(p)
        t_bh = bh(p, alpha)
        plt.plot([1, n_pts], [t_bh, t_bh], linestyle='--', linewidth = 1,\
                 color='limegreen', label='bh threshold')
    p_sort = p[sort_ind]    
    x_axis = np.arange(m)+1
    bh_threshold = alpha*(np.arange(m)+1)/m    
    plt.scatter(x_axis[:n_pts], p_sort[:n_pts], s=16, alpha=0.4, label=label)
    plt.plot(x_axis[:n_pts], bh_threshold[:n_pts], linestyle='--', color='r')
    plt.xlabel('p-value rank')
    plt.ylabel('p-value')
    
def plot_p_sort_with_ci(t, U, sort_ind, n_pts = 100, label=None):
    n_permutation,n_featuer = U.shape
    U_cumsum = np.cumsum(U, axis=0)
    s = compute_s(t, U_cumsum)
    p_hat_lb, p_hat_ub = update_p(s, t, n_permutation, extreme_prob=0.01)
    plot_ind = sort_ind[:n_pts] 
    xaxis = np.arange(n_pts)+1
    y_val = 0.5*(p_hat_lb[plot_ind] + p_hat_ub[plot_ind])
    y_err = p_hat_ub[plot_ind] - y_val    
    plt.errorbar(xaxis, y_val, yerr = y_err, alpha=0.6, label=label)
    plt.title('Avg. sample = %0.1f'%(np.mean(t)))
    
def plot_pmc(p, n_mc, label=None):
    sort_ind = np.argsort(p)
    m = p.shape[0]
    p = p[sort_ind]
    n_mc =n_mc[sort_ind]
    ci = np.sqrt(p*(1-p)/n_mc)*2
    plt.errorbar(np.arange(m)+1, p, yerr = ci, alpha=0.8, label=label)
    
def result_compare(h_hat, h_true):
    D_hat = np.sum(h_hat)
    D_true = np.sum(h_true)
    D_overlap = np.sum(h_hat*h_true)
    return D_hat, D_overlap, D_true
 
"""
    Deprecated
"""
def ada_permute_1(U, alpha=0.1, step_size=5, verbose=False, p_true=None):
    # Set up parameters
    n_permutation, n_feature = U.shape
    U_pt = np.ones([n_feature], dtype=bool)
    R_pt = np.zeros([n_feature], dtype=bool)
    A_pt = np.zeros([n_feature], dtype=bool)
    t = np.zeros([n_feature], dtype=int)
    # p_hat = np.zeros([n_feature], dtype=float)
    p_hat_ub = np.ones([n_feature], dtype=float)
    p_hat_lb = np.zeros([n_feature], dtype=float)
    # r_hat = np.zeros([n_feature], dtype=int)
    r_hat_ub = np.ones([n_feature], dtype=int)*n_feature
    r_hat_lb = np.ones([n_feature], dtype=int)
    n_itr = 0
    while (np.sum(U_pt)>0):
        # Permutation samples
        t[U_pt] += step_size
        # Update parameters
        s = np.zeros([n_feature], dtype=int)
        for i_feature in range(n_feature):
            s[i_feature] = np.sum(U[:t[i_feature], i_feature])
        p_hat_lb, p_hat_ub = update_p(s, t, n_permutation, extreme_prob=0.05)
        r_hat_lb, r_hat_ub = update_r(p_hat_lb, p_hat_ub)
        # Update points
        ind_A = p_hat_lb > (r_hat_ub*alpha/n_feature)
        U_pt[ind_A] = False
        A_pt[ind_A] = True
        ind_R = p_hat_ub > (r_hat_lb*alpha/n_feature)
        n_itr += 1
        if verbose:
            print('## n_itr=%d, avg_sample=%0.1f'%(n_itr, np.mean(t)))
            print('|U|=%d, |A|=%d, |R|=%d'%(U_pt.sum(), A_pt.sum(), R_pt.sum()))
            print(r_hat_ub[0:5])
            print(r_hat_lb[0:5])
            sort_ind = np.argsort(p_true)
            temp_lb = p_hat_lb[sort_ind]
            temp_ub = p_hat_ub[sort_ind]
            temp_rlb = r_hat_lb[sort_ind]
            temp_rub = r_hat_ub[sort_ind]
            temp_U_pt = U_pt[sort_ind]
            y_val = (temp_lb+temp_ub)/2
            y_err = temp_ub-y_val
            n_pts = 200
            xaxis = np.arange(n_feature)+1
            plt.figure()
            plot_p_sort(p_true, n_pts=n_pts, sort_ind=None, label='full_mc')
            plt.scatter(xaxis[:n_pts][temp_U_pt[:n_pts]],\
                        p_true[sort_ind][:n_pts][temp_U_pt[:n_pts]],\
                        alpha=0.4, color='r')
            plt.errorbar(xaxis[:n_pts], y_val[:n_pts], yerr = y_err[:n_pts], alpha=0.6)
            plt.plot(xaxis[:n_pts], (temp_rlb[:n_pts]*alpha/n_feature),\
                     linestyle='--', color='r')
            plt.plot(xaxis[:n_pts], (temp_rub[:n_pts]*alpha/n_feature),\
                     linestyle='--', color='r')
            plt.show()
            print('')
        if n_itr == 5:
            break
    return t

def update_r(p_hat_lb, p_hat_ub):
    """ Compute the binomial confidence intervals
    
    Args:
        s (ndarray): number of successes for each feature.
        t (ndarray): total number of trials for each feature.
        alpha (float): alpha/2 is used for both half_tails.
    """
    n_feature = p_hat_lb.shape[0]
    r_hat_ub = np.zeros([n_feature], dtype=int)
    r_hat_lb = np.zeros([n_feature], dtype=int)
    for i_feature in range(n_feature):
        r_hat_ub[i_feature] = np.sum(p_hat_ub[i_feature] < p_hat_lb)
        r_hat_lb[i_feature] = np.sum(p_hat_lb[i_feature] > p_hat_ub)
    return r_hat_lb, r_hat_ub