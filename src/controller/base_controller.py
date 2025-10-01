"""Base controller interface for neural network controllers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import jax.numpy as jnp
import jax


class BaseController(ABC):
    """Abstract base class for neural network controllers."""
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    @abstractmethod
    def init_params(self, rng: jax.random.PRNGKey) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def forward(self, params: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        pass
    
    @abstractmethod
    def get_action(self, params: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        pass
