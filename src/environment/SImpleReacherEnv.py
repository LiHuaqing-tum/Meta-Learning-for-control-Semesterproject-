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
        self.site_name_obstacle = "obstacle_site"
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
        self.site_id_obstacle = _site_id_or_none(self.site_name_obstacle)
        assert self.site_id_obstacle is not None, "XML 缺少 <site name='obstacle_site'>"


        self.nu = self.mj_model.nu
        self.ctrlrange = np.array(self.mj_model.actuator_ctrlrange, copy=True) if self.nu > 0 else np.zeros((0, 2))

        if self.nu > 0 and self.ctrlrange.size > 0:
            self.ctrlrange_jnp = jnp.asarray(self.ctrlrange)
        else:
            self.ctrlrange_jnp = jnp.zeros((0, 2))


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

        if self.site_id_block is not None:
            base += 2
        if self.site_id_intermediate is not None:
            base += 2
        if self.site_id_target is not None:
            base += 2
        if self.site_id_obstacle is not None:
            base += 2

        self.state_dim = base


    def _get_site_world_pos(self, site_id: Optional[int]) -> Optional[jnp.ndarray]:
        if site_id is None:
            return None
        return self.d.site_xpos[site_id]

    def reset(self, rng):
        self.d = mjx.make_data(self.m)

        
        if self.site_id_obstacle is not None:
            self.obstacle_center_xy = jnp.array(self.d.site_xpos[self.site_id_obstacle, :2])
        else:
            self.obstacle_center_xy = jnp.array([0.0, 0.0])  

        
        if self.site_id_target is not None:
            self.target_center_xy = jnp.array(self.d.site_xpos[self.site_id_target, :2])
        else:
            self.target_center_xy = jnp.array([0.0, 0.0])

        return self._obs_from_data(self.d)


    def step(
        self,
        action: jnp.ndarray,
        rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, bool, Dict[str, Any]]:
        if action.shape[-1] != self.nu:
            raise ValueError(f"Action dim mismatch: got {action.shape[-1]}, expected nu={self.nu}")
        if self.nu > 0 and self.ctrlrange_jnp.size > 0:
            lo = self.ctrlrange_jnp[:, 0]
            hi = self.ctrlrange_jnp[:, 1]
            u  = jnp.clip(action, lo, hi)  
        else:
            u = action
        self.d = self.d.replace(ctrl=u)
        self.d = mjx.step(self.m, self.d)
        obs = self._get_observation()
        reward = jnp.array(0.0, dtype=jnp.float32)
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, info
    
    def train_step(
        self,
        data: mjx.Data,
        action: jnp.ndarray
    ) -> Tuple[mjx.Data, jnp.ndarray]:

        if self.nu > 0 and self.ctrlrange_jnp.size > 0:
            lo = self.ctrlrange_jnp[:, 0]
            hi = self.ctrlrange_jnp[:, 1]
            u  = jnp.clip(action, lo, hi)
        else:
            u  = action

        data_new = data.replace(ctrl=u)
        data_new = mjx.step(self.m, data_new)
        obs = self._obs_from_data(data_new)

        return data_new, obs

    def _obs_from_data(self, data: mjx.Data) -> jnp.ndarray:
        parts = []
        if self.return_full_qpos_qvel:
            parts.append(data.qpos)
            parts.append(data.qvel)

        if self.site_id_block is not None:
            parts.append(data.site_xpos[self.site_id_block, :2])
        if self.site_id_intermediate is not None:
            parts.append(data.site_xpos[self.site_id_intermediate, :2])
        if self.site_id_target is not None:
            parts.append(data.site_xpos[self.site_id_target, :2])
        if self.site_id_obstacle is not None:
            parts.append(data.site_xpos[self.site_id_obstacle, :2])

        return jnp.concatenate(parts, axis=0) if parts else jnp.zeros((0,))


    def _get_observation(self) -> jnp.ndarray:
         return self._obs_from_data(self.d)

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
            "obstacle": self.site_name_obstacle,
        }
        return info

    def get_observation_space(self) -> Tuple[int, ...]:
        return (self.state_dim,)

    def get_action_space(self) -> Tuple[int, ...]:
        return (self.nu,)
    
    def rollout_obs(
        self,
        controls: jnp.ndarray,                 # (T, nu) or (nu, T)
        data0: Optional[mjx.Data] = None) -> jnp.ndarray:
        """
        Roll out a sequence of actions and collect observations.
        Returns: obs_traj (T+1, state_dim), row 0 is observation at data0.
        """
        
        controls = controls if (controls.ndim == 2 and controls.shape[-1] == self.nu) else controls.T
        assert controls.ndim == 2 and controls.shape[1] == self.nu, "controls must be (T, nu) or (nu, T)" 

        if data0 is None:
            data0 = mjx.make_data(self.m)

        def body_fn(data, action):
            data_new, obs = self.train_step(data, action)   # train_step
            return data_new, obs                             # (state_dim,)

        obs0 = self._obs_from_data(data0)                   # (state_dim,)
        _, obs_seq = jax.lax.scan(body_fn, data0, controls) # (T, state_dim)
        return jnp.vstack([obs0[None, :], obs_seq])         # (T+1, state_dim)

    def rollout_xy(
        self,
        controls: jnp.ndarray,                 # (T, nu) 或 (nu, T)
        data0: Optional[mjx.Data] = None,
        use_block: bool = True,
        use_intermediate: bool = False,
        use_obstacle: bool = True,
        use_target: bool = True,
        keep_shape: bool = False,              # use 0 to occupy
    ):
      
        #  controls -> (T, nu)
        controls = controls if (controls.ndim == 2 and controls.shape[-1] == self.nu) else controls.T
        assert controls.ndim == 2 and controls.shape[1] == self.nu, "controls must be (T, nu) or (nu, T)"


        if data0 is None:
            data0 = mjx.make_data(self.m)

        #  site id（maybe None）
        names_flags = [
            ("block",        use_block,        self.site_id_block),
            ("intermediate", use_intermediate, self.site_id_intermediate),
            ("obstacle",     use_obstacle,     self.site_id_obstacle),
            ("target",       use_target,       self.site_id_target),
        ]

        # get the final name and trajectory
        names = []
        sids  = []
        for n, on, sid in names_flags:
            if not on:
                continue
            if sid is None and not keep_shape:
                continue
            names.append(n)
            sids.append(sid)  # maybe none

        if not names:
           
            return jnp.zeros((controls.shape[0] + 1, 0)), []

        def read_row(d: mjx.Data) -> jnp.ndarray:
            cols = []
            for sid in sids:
                if sid is None:  
                    cols.append(jnp.zeros(2))
                else:
                    cols.append(d.site_xpos[sid, :2])
            return jnp.concatenate(cols, axis=0)  # (2*k,)

        def body_fn(data, action):
            data_new, _ = self.train_step(data, action)   # use train_step
            return data_new, read_row(data_new)

        row0 = read_row(data0)                               # (2*k,)
        _, rows = jax.lax.scan(body_fn, data0, controls)     # (T, 2*k)
        traj = jnp.vstack([row0[None, :], rows])             # (T+1, 2*k)
        return traj, names
        

    def rollout_obs_debug(
        self,
        controls: jnp.ndarray,                 # (T, nu) or (nu, T)
        rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Debug/Rendering only: uses env.step (side effects), NOT JIT-friendly.
        Returns: obs_traj (T+1, state_dim).
        """
        controls = controls if (controls.ndim == 2 and controls.shape[-1] == self.nu) else controls.T
        assert controls.ndim == 2 and controls.shape[1] == self.nu, "controls must be (T, nu) or (nu, T)"

        obs_list = [self._get_observation()]
        for a in controls:
            obs, _, _, _ = self.step(a, rng)
            obs_list.append(obs)
        return jnp.stack(obs_list, axis=0)  # (T+1, state_dim)    