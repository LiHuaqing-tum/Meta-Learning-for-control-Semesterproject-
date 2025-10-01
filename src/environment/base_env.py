

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import jax.numpy as jnp
import jax


class BaseEnv(ABC):
  
    
    def __init__(self, **kwargs):
        
        pass
    
    @abstractmethod
    def reset(self, rng: jax.random.PRNGKey) -> jnp.ndarray:

        pass
    
    @abstractmethod
    def step(
        self,
        action: jnp.ndarray,
        rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, bool, Dict[str, Any]]:

        pass
    
    @abstractmethod
    def get_observation_space(self) -> Tuple[int, ...]:
        
        pass
    
    @abstractmethod
    def get_action_space(self) -> Tuple[int, ...]:
     
        pass
