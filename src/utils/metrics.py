import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import entropy

def kl_divergence_from_uniform(dist):
    """
    Compute KL divergence from uniform distribution.
    Higher values = more specialized/focused distribution.
    """
    # Ensure distribution is normalized
    if dist.sum() > 0:
        dist = dist / dist.sum()
    else:
        return 0.0
    
    dims = len(dist)
    # Uniform distribution
    uniform = np.ones(dims) / dims
    
    # KL divergence: D(P||Q) = sum(P * log(P/Q))
    epsilon = 1e-10
    kl_div = np.sum(dist * np.log((dist + epsilon) / (uniform + epsilon)))
    
    return kl_div

def fit_temperature(dist, reference='tonic'):
    """
    Fit a temperature parameter to the distribution using softmax model.
    Lower T = more focused, Higher T = more uniform.
    """
    if dist.sum() == 0:
        return np.inf
    
    dist = dist / dist.sum()
    dims = len(dist)
    
    # Choose reference point
    if reference == 'tonic':
        if dims == 35:
            center_idx = 17  # Middle of 35-D
        else:
            center_idx = 0   # C for 12-D (assuming C-major aligned)
    else:
        center_idx = np.argmax(dist)
    
    # Create distance-based logits
    positions = np.arange(dims)
    
    if dims == 12:
        # Circular distance for 12-D
        diff = np.abs(positions - center_idx)
        distances = np.minimum(diff, 12 - diff)
    else:
        # Linear distance for 35-D
        distances = np.abs(positions - center_idx)
        
    base_logits = -distances
    
    def objective(temp):
        if temp <= 0: return 1e10
        model_dist = softmax(base_logits / temp)
        kl = np.sum(dist * np.log((dist + 1e-10) / (model_dist + 1e-10)))
        return kl
    
    result = minimize_scalar(objective, bounds=(0.01, 20), method='bounded')
    return result.x

def positional_entropy(dist):
    """Calculate Shannon entropy of the distribution."""
    if dist.sum() == 0: return 0.0
    dist = dist / dist.sum()
    return entropy(dist, base=2)

def calc_fifth_width(dist):
    """Calculate the span of non-zero positions (fifth-width)."""
    threshold = 0.01
    active = np.where(dist > threshold)[0]
    
    if len(active) == 0:
        return 0
        
    dims = len(dist)
    if dims == 12:
        # For 12-D, this is just number of active pitch classes
        return len(active)
    else:
        # For 35-D, it's the span
        return active[-1] - active[0] + 1
