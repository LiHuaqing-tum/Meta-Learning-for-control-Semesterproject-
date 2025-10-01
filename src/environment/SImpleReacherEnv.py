from typing import Any, Dict, Tuple, Optional, Sequence
import os
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.mjx as mjx

from .base_env import BaseEnv


class SimpleReacherEnv(BaseEnv):
    def __init__(
        self,
        *,
        model_path: str = "models/panda_push_scene.xml",
        return_full_qpos_qvel: bool = True,
        torso_world_free: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)

        self.site_name_block = "block_site"
        self.site_name_intermediate = "intermediate_site"
        self.site_name_target = "target_site"

        def _site_id_or_none(name: str) -> Optional[int]:
            try:
                return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            except Exception:
                return None

        self.site_id_block = _site_id_or_none(self.site_name_block)
        self.site_id_intermediate = _site_id_or_none(self.site_name_intermediate)
        self.site_id_target = _site_id_or_none(self.site_name_target)

        self.nu = self.mj_model.nu
        self.ctrlrange = np.array(self.mj_model.actuator_ctrlrange, copy=True) if self.nu > 0 else np.zeros((0, 2))

        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        self.model_path = model_path
        self.model = self.mj_model
        self.data = None
        self.m = self.mjx_model
        self.d = self.mjx_data

        self._compute_obs_dim(return_full_qpos_qvel)
        self.return_full_qpos_qvel = return_full_qpos_qvel
        self.torso_world_free = torso_world_free

        self._xy_abs_limit = 2.0
        # make sure our blocks don't go to far

    def _compute_obs_dim(self, return_full: bool):
        nq = int(self.mj_model.nq)
        nv = int(self.mj_model.nv)
        base = 0
        if return_full:
            base += nq + nv
        base += 2
        if self.site_id_intermediate is not None:
            base += 2
        if self.site_id_target is not None:
            base += 2
        self.state_dim = base

    def _get_site_world_pos(self, site_id: Optional[int]) -> Optional[jnp.ndarray]:
        if site_id is None:
            return None
        return self.d.site_xpos[site_id]

    def reset(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
        self.d = mjx.make_data(self.m)
        return self._get_observation()

    def step(
        self,
        action: jnp.ndarray,
        rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, bool, Dict[str, Any]]:
        if action.shape[-1] != self.nu:
            raise ValueError(f"Action dim mismatch: got {action.shape[-1]}, expected nu={self.nu}")
        if self.nu > 0 and self.ctrlrange.size > 0:
            lo = jnp.asarray(self.ctrlrange[:, 0])
            hi = jnp.asarray(self.ctrlrange[:, 1])
            u = jnp.clip(action, lo, hi)
        else:
            u = action
        self.d = self.d.replace(ctrl=u)
        self.d = mjx.step(self.m, self.d)
        obs = self._get_observation()
        reward = jnp.array(0.0, dtype=jnp.float32)
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, info

    def _get_observation(self) -> jnp.ndarray:
        parts = []
        if self.return_full_qpos_qvel:
            parts.append(self.d.qpos)
            parts.append(self.d.qvel)
        block_xy = jnp.array([0.0, 0.0])
        block_p = self._get_site_world_pos(self.site_id_block)
        if block_p is not None:
            block_xy = block_p[:2]
        parts.append(block_xy)
        inter_xy = self._get_site_world_pos(self.site_id_intermediate)
        if inter_xy is not None:
            parts.append(inter_xy[:2])
        target_xy = self._get_site_world_pos(self.site_id_target)
        if target_xy is not None:
            parts.append(target_xy[:2])
        return jnp.concatenate(parts, axis=0)

    def _compute_reward(self) -> jnp.ndarray:
        return jnp.array(0.0, dtype=jnp.float32)

    def _is_done(self) -> bool:
        block_p = self._get_site_world_pos(self.site_id_block)
        if block_p is None:
            return False
        px, py = float(block_p[0]), float(block_p[1])
        if not np.isfinite(px) or not np.isfinite(py):
            return True
        if abs(px) > self._xy_abs_limit or abs(py) > self._xy_abs_limit:
            return True
        return False

    def _get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "nu": int(self.nu),
            "ctrlrange": self.ctrlrange.tolist() if self.ctrlrange.size else [],
        }
        if self.return_full_qpos_qvel:
            info["qpos"] = np.array(self.d.qpos).tolist()
            info["qvel"] = np.array(self.d.qvel).tolist()

        def _opt_pos(site_id):
            p = self._get_site_world_pos(site_id)
            return np.array(p).tolist() if p is not None else None

        info["block_site_pos"] = _opt_pos(self.site_id_block)
        info["intermediate_site_pos"] = _opt_pos(self.site_id_intermediate)
        info["target_site_pos"] = _opt_pos(self.site_id_target)
        info["site_names"] = {
            "block": self.site_name_block,
            "intermediate": self.site_name_intermediate,
            "target": self.site_name_target,
        }
        return info

    def get_observation_space(self) -> Tuple[int, ...]:
        return (self.state_dim,)

    def get_action_space(self) -> Tuple[int, ...]:
        return (self.nu,)

