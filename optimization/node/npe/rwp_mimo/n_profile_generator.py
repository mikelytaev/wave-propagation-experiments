import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import numpy as np

def generate_n_profile(
    height: jnp.ndarray,
    n_surface: float = 315.0,  # Typical surface N-unit value
    gradient: float = -40.0,   # Typical gradient in N-units/km
    duct_height: Optional[float] = None,
    duct_strength: Optional[float] = None,
    duct_width: Optional[float] = None,
    random_perturbations: bool = True,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Generate a tropospheric N-profile with optional ducting.
    
    Args:
        height: Array of heights in meters
        n_surface: Surface N-unit value (typically around 315)
        gradient: Vertical gradient in N-units/km (typically around -40)
        duct_height: Height of the duct center in meters (if None, no duct)
        duct_strength: Strength of the duct in N-units (if None, no duct)
        duct_width: Width of the duct in meters (if None, no duct)
        random_perturbations: Whether to add random perturbations
        key: JAX random key for reproducibility
        
    Returns:
        N-profile values at the given heights
    """
    # Convert height to kilometers for gradient calculation
    height_km = height / 1000.0
    
    # Base linear profile
    n_profile = n_surface + gradient * height_km
    
    # Add duct if specified
    if duct_height is not None and duct_strength is not None and duct_width is not None:
        # Gaussian duct profile
        duct_center = duct_height / 1000.0  # Convert to km
        duct_sigma = duct_width / (1000.0 * 2.355)  # Convert to km and FWHM to sigma
        duct = duct_strength * jnp.exp(-0.5 * ((height_km - duct_center) / duct_sigma) ** 2)
        n_profile += duct
    
    # Add random perturbations if requested
    if random_perturbations and key is not None:
        # Generate random perturbations that decrease with height
        key, subkey = jax.random.split(key)
        perturbations = jax.random.normal(subkey, shape=height.shape)
        height_factor = jnp.exp(-height_km / 2.0)  # Decrease perturbations with height
        n_profile += 2.0 * perturbations * height_factor
    
    return n_profile

def generate_batch_n_profiles(
    num_profiles: int,
    height_points: int = 100,
    min_height: float = 0.0,
    max_height: float = 10000.0,
    key: Optional[jax.random.PRNGKey] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a batch of N-profiles with random parameters.
    
    Args:
        num_profiles: Number of profiles to generate
        height_points: Number of height points per profile
        min_height: Minimum height in meters
        max_height: Maximum height in meters
        key: JAX random key for reproducibility
        
    Returns:
        Tuple of (heights, n_profiles)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Generate height points
    heights = jnp.linspace(min_height, max_height, height_points)
    
    # Generate random parameters for each profile
    key, subkey = jax.random.split(key)
    n_surface = 315.0 + 5.0 * jax.random.normal(subkey, shape=(num_profiles,))
    
    key, subkey = jax.random.split(key)
    gradient = -40.0 + 5.0 * jax.random.normal(subkey, shape=(num_profiles,))
    
    # Randomly decide which profiles will have ducts
    key, subkey = jax.random.split(key)
    has_duct = jax.random.bernoulli(subkey, p=0.3, shape=(num_profiles,))
    
    # Generate duct parameters
    key, subkey = jax.random.split(key)
    duct_height = 1000.0 + 3000.0 * jax.random.uniform(subkey, shape=(num_profiles,))
    duct_height = jnp.where(has_duct, duct_height, jnp.nan)
    
    key, subkey = jax.random.split(key)
    duct_strength = 20.0 + 10.0 * jax.random.normal(subkey, shape=(num_profiles,))
    duct_strength = jnp.where(has_duct, duct_strength, jnp.nan)
    
    key, subkey = jax.random.split(key)
    duct_width = 200.0 + 100.0 * jax.random.normal(subkey, shape=(num_profiles,))
    duct_width = jnp.where(has_duct, duct_width, jnp.nan)
    
    # Generate profiles
    def generate_single_profile(i):
        profile_key = jax.random.fold_in(key, i)
        return generate_n_profile(
            heights,
            n_surface[i],
            gradient[i],
            duct_height[i] if has_duct[i] else None,
            duct_strength[i] if has_duct[i] else None,
            duct_width[i] if has_duct[i] else None,
            False,
            profile_key
        )
    
    n_profiles = jax.vmap(generate_single_profile)(jnp.arange(num_profiles))
    
    return heights, n_profiles

# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate a single profile with a duct
    heights = jnp.linspace(0, 100, 1000)
    n_profile = generate_n_profile(
        heights,
        n_surface=315.0,
        gradient=-40.0,
        duct_height=2000.0,
        duct_strength=30.0,
        duct_width=300.0,
        random_perturbations=True,
        key=jax.random.PRNGKey(0)
    )
    
    # Plot the profile
    plt.figure(figsize=(10, 6))
    plt.plot(n_profile, heights/1000.0)
    plt.xlabel('N-units')
    plt.ylabel('Height (km)')
    plt.title('Example N-profile with Duct')
    plt.grid(True)
    plt.show()
    
    # Generate a batch of profiles
    heights, n_profiles = generate_batch_n_profiles(
        num_profiles=5,
        height_points=1000,
        min_height=0.0,
        max_height=100.0,
        key=jax.random.PRNGKey(1)
    )
    
    # Plot the batch
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(n_profiles[i], heights/1000.0, label=f'Profile {i+1}')
    plt.xlabel('N-units')
    plt.ylabel('Height (km)')
    plt.title('Batch of N-profiles')
    plt.legend()
    plt.grid(True)
    plt.show() 