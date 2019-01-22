import numpy as np
from scipy import stats

def load_demo_data(n_hypothesis=500, pi1=0.1, effect_size=2, random_state=0):
    """ Generate a toy dataset
    """
    np.random.seed(random_state)
    # Set up the parameters.
    n_alt = int(pi1 * n_hypothesis)
    n_null = n_hypothesis - n_alt
    h = np.zeros([n_hypothesis], dtype=bool)
    h[0:n_alt] = True
    # p-values
    p = np.zeros([n_hypothesis], dtype=float)
    z_null = np.random.randn(n_null)
    p[h==0] = 1 - stats.norm.cdf(z_null)
    z_alt = np.random.randn(n_alt) + effect_size
    p[h==1] = 1 - stats.norm.cdf(z_alt)
    return p, h