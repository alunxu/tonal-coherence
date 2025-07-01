import numpy as np
import math
from scipy.optimize import minimize
from scipy.special import softmax

class ImprovedTDM:
    """Improved TDM with multiple random starts."""
    
    def __init__(self, dims=35):
        self.dims = dims
        self.lof_size = dims
        self.num_intervals = 6
        self.interval_steps = {
            '+P5': 1, '-P5': -1,
            '+M3': 4, '-M3': -4,
            '+m3': -3, '-m3': 3
        }
        self.interval_names = list(self.interval_steps.keys())
    
    def compute_all_probabilities(self, tonal_center, weights, lambda_param, max_length=10):
        """Compute probabilities for all targets from tonal center (Vectorized).
        
        Uses standard Poisson distribution for path length as per Lieck et al. (2020):
        P(n|λ) = e^(-λ) * λ^n / n!
        """
        # Initialize final probabilities
        final_probs = np.zeros(self.dims)
        
        # n=0 term: Poisson(0, λ) = e^(-λ), and the only reachable target is the center
        final_probs[tonal_center] = np.exp(-lambda_param)  # Standard Poisson(0) = e^(-λ)
        
        # DP State: probability distribution over positions after n steps
        current_dist = np.zeros(self.dims)
        current_dist[tonal_center] = 1.0
        
        steps = [self.interval_steps[name] for name in self.interval_names]
        
        for n in range(1, max_length + 1):
            next_dist = np.zeros(self.dims)
            
            for i, step in enumerate(steps):
                w = weights[i]
                if w < 1e-9: continue
                
                if self.dims == 12:
                    # Circular shift
                    shifted = np.roll(current_dist, step)
                else:
                    # Linear shift with zero padding
                    shifted = np.zeros(self.dims)
                    if step > 0:
                        shifted[step:] = current_dist[:-step]
                    elif step < 0:
                        shifted[:step] = current_dist[-step:]
                    else:
                        shifted = current_dist
                
                next_dist += shifted * w
            
            current_dist = next_dist
            
            # Add to final probabilities: Poisson(n, λ) = e^(-λ) * λ^n / n!
            path_weight = np.exp(-lambda_param) * (lambda_param ** n) / math.factorial(n)
            final_probs += current_dist * path_weight
            
        return np.maximum(final_probs, 1e-10)
    
    def infer_multistart(self, distribution, n_starts=5, verbose=False, forced_center=None):
        """Infer parameters with multiple random starts."""
        if distribution.sum() == 0:
            return None
        
        normalized = distribution / distribution.sum()
        
        if forced_center is not None:
            tonal_center = int(forced_center)
        else:
            tonal_center = int(np.argmax(distribution))
        
        best_result = None
        best_likelihood = -np.inf
        
        lambdas_tried = []
        
        for start in range(n_starts):
            # Vary initialization strategy
            if start == 0:
                # Data-driven initialization
                positions = np.arange(self.dims)
                if self.dims == 12:
                    diff = np.abs(positions - tonal_center)
                    distances = np.minimum(diff, 12 - diff)
                else:
                    distances = np.abs(positions - tonal_center)
                avg_dist = np.average(distances, weights=normalized)
                initial_lambda = np.clip(avg_dist / 5.0, 0.2, 2.5)
            else:
                initial_lambda = np.random.uniform(0.1, 3.0)
            
            initial_weights = np.random.dirichlet(np.ones(6))
            
            def objective(params):
                log_weights = params[:6]
                lambda_param = params[6]
                weights = softmax(log_weights)
                
                if lambda_param <= 0.01 or lambda_param > 5.0:
                    return 1e10
                
                # Vectorized probability computation
                probs = self.compute_all_probabilities(tonal_center, weights, lambda_param)
                
                # Log-likelihood
                # sum( data[i] * log(model[i]) )
                log_lik = np.sum(normalized * np.log(probs))
                
                return -log_lik
            
            initial_params = np.concatenate([
                np.log(initial_weights + 1e-10),
                [initial_lambda]
            ])
            
            try:
                result = minimize(
                    objective,
                    initial_params,
                    method='L-BFGS-B',
                    bounds=[(None, None)]*6 + [(0.01, 5.0)],
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-7}
                )
                
                lambdas_tried.append(result.x[6])
                
                if result.fun < -best_likelihood:
                    best_likelihood = -result.fun
                    best_result = result
                    
            except:
                continue
        
        if best_result is None:
            return None
        
        log_weights = best_result.x[:6]
        lambda_param = best_result.x[6]
        weights = softmax(log_weights)
        
        if verbose:
            print(f"  λ values tried: {[f'{l:.3f}' for l in lambdas_tried]}")
            print(f"  Best λ = {lambda_param:.3f}")
        
        return {
            'tonal_center': tonal_center,
            'lambda': lambda_param,
            'weights': weights,
            'converged': best_result.success
        }
