import numpy as np


def post_analysis(samples: np.ndarray,E_f: float,var_f: float,thinning:int = 10) -> float:
    """Placeholder for post analysis on MCMC samples.

    Returns a simple evaluation metric (e.g., mean squared norm of samples).
    Replace with domain-specific metrics later.
    """
    if samples.size == 0:
        return float('nan')
    #print(samples.shape)
    # If samples have shape (T, D) or (T, ...), compute a basic metric
    # Here: average squared norm across time
    flat = samples.reshape(samples.shape[0], -1)[::thinning]
    metric = np.mean((np.mean(flat**2, axis=0)-E_f)**2/var_f)
    print(np.mean(flat**2, axis=0),E_f)
    return metric
