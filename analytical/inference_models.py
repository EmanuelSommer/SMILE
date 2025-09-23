import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from typing import Literal, Tuple, Optional, Callable


def Ill_condition_Gaussian_log_density(
    N_total: int,
    dim_val: int,
    constant_val: float,
    batch_size: int,
    Condition_numbers: float = 100,
    seed: int = 0,
    random_rotation: bool = True,
    noise_type: Literal["isotropic", "anisotropic", "correlated", "spatially_varied", "none"] = "isotropic",
    noise_scale: float = 64.0,
    center: bool = True,
    rotation_seed: Optional[int] = None,
) -> Tuple[Callable[..., np.ndarray], Callable[..., Callable], np.ndarray, np.ndarray]:
    """Build ill-conditioned Gaussian dataset and log-density factories.

    Parameters
    - N_total, dim_val, constant_val, batch_size: core settings
    - Condition_numbers: target condition number spread across dimensions
    - seed: RNG seed for data
    - random_rotation: if True, apply a random orthogonal rotation to covariance
    - noise_type: distribution of synthetic noise
        * isotropic: iid Gaussian scaled by noise_scale
        * anisotropic: per-dimension scaling by condition spectrum
        * correlated: correlated via covariance Cholesky factor
        * spatially_varied: heteroskedastic likelihood in log-density
        * none: zero noise (deterministic)
    - noise_scale: scalar multiplier for base noise
    - center: center dataset to zero mean
    - rotation_seed: optional separate seed for rotation matrix

    Returns
    - create_dataset, make_batched_log_p, E_f (diag of cov), Var_f (2*E_f^2)
    """
    if Condition_numbers <= 0:
        raise ValueError("Condition_numbers must be positive.")
    if dim_val <= 0 or N_total <= 0:
        raise ValueError("dim_val and N_total must be positive integers.")

    rng = np.random.default_rng(seed)
    conds = np.logspace(-0.5 * np.log10(Condition_numbers), 0.5 * np.log10(Condition_numbers), dim_val)
    cov = np.diag(conds**2)

    if random_rotation:
        rot_rng = np.random.default_rng(rotation_seed if rotation_seed is not None else seed + 1)
        Q, _ = np.linalg.qr(rot_rng.standard_normal((dim_val, dim_val)))
        cov = Q @ cov @ Q.T

    # Use JAX arrays for downstream grad-friendly ops
    L = jnp.array(np.linalg.cholesky(cov))
    L_inv = jnp.linalg.inv(L)
    E_f = jnp.diag(jnp.array(cov))
    Var_f = 2 * E_f**2

    def create_dataset(N_total=N_total, dim_val=dim_val, constant_val=constant_val, seed=seed):
        """Create synthetic dataset with requested covariance/noise model."""
        rng_local = np.random.default_rng(seed)
        if noise_type == "none":
            noise = np.zeros((N_total, dim_val), dtype=float)
        else:
            noise = rng_local.standard_normal((N_total, dim_val)) * noise_scale
            if noise_type == "anisotropic":
                noise = noise * conds
            elif noise_type == "correlated":
                noise = noise @ np.asarray(L).T
            elif noise_type == "isotropic":
                pass  # already isotropic
            elif noise_type == "spatially_varied":
                # keep base; heteroskedasticity handled in log-density
                noise = noise @ np.asarray(L).T
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")

        y = noise + constant_val
        if center:
            y = y - np.mean(y, axis=0, keepdims=True)
        print(f"Dataset created with shape: {y.shape}")
        return y
    if noise_type == "spatially_varied":
        def make_batched_log_p(y_batch, N=N_total, batch_size=batch_size):
            """Return a batched log-density function for the current mini-batch."""
            def logdensity_fn(x):
                # Example heteroskedastic form; tune as needed
                diff = x @ L_inv.T
                log_p = -0.5 * jnp.sum(diff**2, axis=-1)
                return jnp.sum(log_p) + jnp.sum(jnp.sum(y_batch * x, axis=0)) * jnp.exp(-x[2]/6.5)
            return logdensity_fn
    else:
        def make_batched_log_p(y_batch, N=N_total, batch_size=batch_size):
            def logdensity_fn(x):
                diff =  x @ L_inv.T
                log_p = -0.5 * jnp.sum(diff**2, axis=-1)
                return jnp.sum(log_p)+ jnp.sum(jnp.sum(y_batch * x, axis=0))
            return logdensity_fn

    return create_dataset, make_batched_log_p, np.asarray(E_f), np.asarray(Var_f)



def Banana_log_density(N_total, dim_val, constant_val,batch_size,seed=0,return_cov=False):

    #conditions = np.logspace(-0.5 * jnp.log10(Condition_numbers), 0.5 * jnp.log10(Condition_numbers), dim_val)
    #cov_matrix = np.diag(conditions**2)

    def create_dataset(N_total=N_total, dim_val=dim_val, constant_val=constant_val, seed=seed):
        """Creates the synthetic dataset."""
        np.random.seed(seed)
        noise = np.random.randn(N_total, dim_val)
        y = noise + constant_val
        y[:,0] = y[:,0] - constant_val
        y[:,1] = y[:,1] + 3
        print(f"Dataset created with shape: {y.shape}")
        return y

    def make_batched_log_p(y_batch ,N=N_total,batch_size=batch_size):
        #logdensity_fn = lambda x: -0.5 * jnp.sum(jnp.square((y_batch-x)/conditions))*N/batch_size
        def logdensity_fn(x):
            x = x.at[1].set(x[1]-(300*x[0]**2-3))
            #x2 = x[:1]
            #y1 = y_batch[:, 0]
            #y2 = y_batch[:, 1]
            return -0.5 * jnp.sum(jnp.square(y_batch-x)) * N / batch_size
        return logdensity_fn

    if not return_cov:
        return create_dataset, make_batched_log_p, 1
    
    return create_dataset, make_batched_log_p, np.eye(dim_val)  


def Funnel_log_density(
    N_total: int,
    dim_val: int,
    constant_val: float,
    batch_size: int,
    Condition_numbers: float = 100,
    seed: int = 0,
    random_rotation: bool = True,
    noise_type: Literal["isotropic", "anisotropic", "correlated", "spatially_varied", "none"] = "isotropic",
    noise_scale: float = 64.0,
    center: bool = True,
    rotation_seed: Optional[int] = None,
) -> Tuple[Callable[..., np.ndarray], Callable[..., Callable], np.ndarray, np.ndarray]:
    """Build ill-conditioned Gaussian dataset and log-density factories.

    Parameters
    - N_total, dim_val, constant_val, batch_size: core settings
    - Condition_numbers: target condition number spread across dimensions
    - seed: RNG seed for data
    - random_rotation: if True, apply a random orthogonal rotation to covariance
    - noise_type: distribution of synthetic noise
        * isotropic: iid Gaussian scaled by noise_scale
        * anisotropic: per-dimension scaling by condition spectrum
        * correlated: correlated via covariance Cholesky factor
        * spatially_varied: heteroskedastic likelihood in log-density
        * none: zero noise (deterministic)
    - noise_scale: scalar multiplier for base noise
    - center: center dataset to zero mean
    - rotation_seed: optional separate seed for rotation matrix

    Returns
    - create_dataset, make_batched_log_p, E_f (diag of cov), Var_f (2*E_f^2)
    """
    if Condition_numbers <= 0:
        raise ValueError("Condition_numbers must be positive.")
    if dim_val <= 0 or N_total <= 0:
        raise ValueError("dim_val and N_total must be positive integers.")

    rng = np.random.default_rng(seed)
    conds = np.logspace(-0.5 * np.log10(Condition_numbers), 0.5 * np.log10(Condition_numbers), dim_val)
    cov = np.diag(conds**2)

    if random_rotation:
        rot_rng = np.random.default_rng(rotation_seed if rotation_seed is not None else seed + 1)
        Q, _ = np.linalg.qr(rot_rng.standard_normal((dim_val, dim_val)))
        cov = Q @ cov @ Q.T

    # Use JAX arrays for downstream grad-friendly ops
    L = jnp.array(np.linalg.cholesky(cov))
    #L_inv = jnp.linalg.inv(L)
    E_f = jnp.array([ 8.99972035, 87.89693536, 90.62155941, 88.93891196, 90.10817027,
        87.39360339, 88.12057611, 88.27542185, 89.08752328, 89.01422079])
    Var_f = jnp.array([1.61815063e+02, 42427520.22230221, 42427520.22230221, 4.78096295e+07,
        4.04881758e+07, 42427520.22230221, 42427520.22230221, 42427520.22230221,
        42427520.22230221, 42427520.22230221])

    def create_dataset(N_total=N_total, dim_val=dim_val, constant_val=constant_val, seed=seed):
        """Create synthetic dataset with requested covariance/noise model."""
        rng_local = np.random.default_rng(seed)
        if noise_type == "none":
            noise = np.zeros((N_total, dim_val), dtype=float)
        else:
            noise = rng_local.standard_normal((N_total, dim_val)) * noise_scale
            if noise_type == "anisotropic":
                noise = noise * conds
            elif noise_type == "correlated":
                noise = noise @ np.asarray(L).T
            elif noise_type == "isotropic":
                pass  # already isotropic
            elif noise_type == "spatially_varied":
                # keep base; heteroskedasticity handled in log-density
                noise = noise @ np.asarray(L).T
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")

        y = noise + constant_val
        if center:
            y = y - np.mean(y, axis=0, keepdims=True)
        print(f"Dataset created with shape: {y.shape}")
        return y
    if noise_type == "spatially_varied":
        def make_batched_log_p(y_batch, N=N_total, batch_size=batch_size):
            """Return a batched log-density function for the current mini-batch."""
            def logdensity_fn(x):
                # Example heteroskedastic form; tune as needed
                sigma_theta = 3.0
                ndims = dim_val
                theta = x[0]
                X = x[..., 1:]
                log_p = -0.5* jnp.square(theta / sigma_theta) - 0.5 * (ndims - 1) * theta - 0.5 * jnp.exp(-theta) * jnp.sum(jnp.square(X), axis = -1)
                return jnp.sum(log_p) + jnp.sum(jnp.sum(y_batch * x, axis=0)) * jnp.exp(-x[2]/9)
            return logdensity_fn
    else:
        def make_batched_log_p(y_batch, N=N_total, batch_size=batch_size):
            def logdensity_fn(x):
                sigma_theta = 3.0
                ndims = dim_val
                theta = x[0]
                X = x[..., 1:]
                log_p = -0.5* jnp.square(theta / sigma_theta) - 0.5 * (ndims - 1) * theta - 0.5 * jnp.exp(-theta) * jnp.sum(jnp.square(X), axis = -1)
                return jnp.sum(log_p)+ jnp.sum(jnp.sum(y_batch * x, axis=0))
            return logdensity_fn

    return create_dataset, make_batched_log_p, np.asarray(E_f), np.asarray(Var_f)


def Rosenbrock_log_density(
    N_total: int,
    dim_val: int,
    constant_val: float,
    batch_size: int,
    Condition_numbers: float = 100,
    seed: int = 0,
    random_rotation: bool = True,
    noise_type: Literal["isotropic", "anisotropic", "correlated", "spatially_varied", "none"] = "isotropic",
    noise_scale: float = 64.0,
    center: bool = True,
    rotation_seed: Optional[int] = None,
) -> Tuple[Callable[..., np.ndarray], Callable[..., Callable], np.ndarray, np.ndarray]:
    """Build ill-conditioned Gaussian dataset and log-density factories.

    Parameters
    - N_total, dim_val, constant_val, batch_size: core settings
    - Condition_numbers: target condition number spread across dimensions
    - seed: RNG seed for data
    - random_rotation: if True, apply a random orthogonal rotation to covariance
    - noise_type: distribution of synthetic noise
        * isotropic: iid Gaussian scaled by noise_scale
        * anisotropic: per-dimension scaling by condition spectrum
        * correlated: correlated via covariance Cholesky factor
        * spatially_varied: heteroskedastic likelihood in log-density
        * none: zero noise (deterministic)
    - noise_scale: scalar multiplier for base noise
    - center: center dataset to zero mean
    - rotation_seed: optional separate seed for rotation matrix

    Returns
    - create_dataset, make_batched_log_p, E_f (diag of cov), Var_f (2*E_f^2)
    """
    if Condition_numbers <= 0:
        raise ValueError("Condition_numbers must be positive.")
    if dim_val <= 0 or N_total <= 0:
        raise ValueError("dim_val and N_total must be positive integers.")

    rng = np.random.default_rng(seed)
    conds = np.logspace(-0.5 * np.log10(Condition_numbers), 0.5 * np.log10(Condition_numbers), dim_val)
    cov = np.diag(conds**2)

    if random_rotation:
        rot_rng = np.random.default_rng(rotation_seed if rotation_seed is not None else seed + 1)
        Q, _ = np.linalg.qr(rot_rng.standard_normal((dim_val, dim_val)))
        cov = Q @ cov @ Q.T

    # Use JAX arrays for downstream grad-friendly ops
    L = jnp.array(np.linalg.cholesky(cov))
    D = dim_val//2
    #L_inv = jnp.linalg.inv(L)
    e_x = jnp.array(
            [
                1.0,
            ]
            * D
            + [
                2.0,
            ]
            * D
        )
    E_f = jnp.array(
            [
                2.0,
            ]
            * D
            + [
                10.10017429,
            ]
            * D
        )
    Var_f = jnp.array(
            [
                6.00036273,
            ]
            * D
            + [
                668.69693635,
            ]
            * D
        )

    def create_dataset(N_total=N_total, dim_val=dim_val, constant_val=constant_val, seed=seed):
        """Create synthetic dataset with requested covariance/noise model."""
        rng_local = np.random.default_rng(seed)
        if noise_type == "none":
            noise = np.zeros((N_total, dim_val), dtype=float)
        else:
            noise = rng_local.standard_normal((N_total, dim_val)) * noise_scale
            if noise_type == "anisotropic":
                noise = noise * conds
            elif noise_type == "correlated":
                noise = noise @ np.asarray(L).T
            elif noise_type == "isotropic":
                pass  # already isotropic
            elif noise_type == "spatially_varied":
                # keep base; heteroskedasticity handled in log-density
                noise = noise @ np.asarray(L).T
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")

        y = noise + constant_val
        if center:
            y = y - np.mean(y, axis=0, keepdims=True)
        print(f"Dataset created with shape: {y.shape}")
        return y
    if noise_type == "spatially_varied":
        def make_batched_log_p(y_batch, N=N_total, batch_size=batch_size):
            """Return a batched log-density function for the current mini-batch."""
            def logdensity_fn(x):
                # Example heteroskedastic form; tune as needed
                X, Y = x[..., : dim_val // 2], x[..., dim_val // 2 :]
                log_p = -0.5 * jnp.sum(
            jnp.square(X - 1.0) + jnp.square(jnp.square(X) - Y) / 0.1, axis=-1
        )
                #log_p = -0.5* jnp.square(theta / sigma_theta) - 0.5 * (ndims - 1) * theta - 0.5 * jnp.exp(-theta) * jnp.sum(jnp.square(X), axis = -1)
                return jnp.sum(log_p) + jnp.sum(jnp.sum(y_batch * x, axis=0)) * jnp.exp(-x[2]/1.414)
            return logdensity_fn
    else:
        def make_batched_log_p(y_batch, N=N_total, batch_size=batch_size):
            def logdensity_fn(x):
                X, Y = x[..., : dim_val // 2], x[..., dim_val // 2 :]
                log_p = -0.5 * jnp.sum(
            jnp.square(X - 1.0) + jnp.square(jnp.square(X) - Y) / 0.1, axis=-1
        )
                return jnp.sum(log_p)+ jnp.sum(jnp.sum(y_batch * x, axis=0))
            return logdensity_fn

    return create_dataset, make_batched_log_p, np.asarray(E_f), np.asarray(Var_f)