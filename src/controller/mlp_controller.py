

from typing import Dict, Any, Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from .base_controller import BaseController


class MLPController(BaseController):
    
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        activation: str = "relu",
        output_activation: str = "tanh",
        **kwargs
    ):
        super().__init__(input_dim, output_dim, **kwargs)
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_activation = output_activation
        

        self.network = self._create_network()
        
    def _create_network(self) -> nn.Module:
        layers = []
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Dense(hidden_dim))
            if self.activation == "relu":
                layers.append(nn.relu)
            elif self.activation == "tanh":
                layers.append(nn.tanh)
            elif self.activation == "gelu":
                layers.append(nn.gelu)
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
        
        # Output layer
        layers.append(nn.Dense(self.output_dim))
        if self.output_activation == "tanh":
            layers.append(nn.tanh)
        elif self.output_activation == "linear":
            pass
        else:
            raise ValueError(f"Unknown output activation: {self.output_activation}")
            
        return nn.Sequential(layers)
    
    def init_params(self, rng: jax.random.PRNGKey) -> Dict[str, Any]:
        dummy_input = jnp.zeros((1, self.input_dim))
        params = self.network.init(rng, dummy_input)
        return params
    
    def forward(self, params: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, obs)
    
    def get_action(self, params: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        return self.forward(params, obs)
