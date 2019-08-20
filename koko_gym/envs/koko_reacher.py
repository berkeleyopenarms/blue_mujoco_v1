from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from mujoco_py.generated import const
import os
import random
import time
import numpy as np

class KokoReacherEnv(utils.EzPickle, mujoco_env.MujocoEnv):
    def __init__(self):
        self.init_done = False
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "koko_reacher.xml"), 2)
        self.viewer = self._get_viewer('human')
        # adjust the actuation space
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        low, high = low[:-4], high[:-4] # four joints for finger are dependants of the finger inertial joint
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.gripper_action = self.sim.data.qpos[-4:]
        self.init_done = True

    def step(self, a):
        vec = self.get_body_com("robotleftfingertip") - self.get_body_com("target")
        reward_dist = -np.square(2.0*np.linalg.norm(vec))
        reward_vel = -np.sqrt(np.square(self.sim.data.qvel).mean())
        reward_ctrl = -np.square(a).sum()/len(self.sim.data.ctrl)
        reward = reward_dist + reward_ctrl

        if self.init_done:
            self.gripper_action = np.ones(4) * a[-1]
            self.gripper_action[1] *= -1
            self.gripper_action[3] *= -1
            a = np.concatenate((a,self.gripper_action))

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        info = {'reward_dist':reward_dist,
                'reward_vel':reward_vel}
        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos,
            self.sim.data.qvel,
            self.get_body_com("robotleftfingertip") - self.get_body_com("target")
        ])

    def viewer_setup(self, camera_type='global_cam', camera_select=0):
        if camera_type == 'fixed_cam':
            cam_type = const.CAMERA_FIXED
            camera_select = camera_select
        elif camera_type == 'global_cam':
            cam_type = 0
        DEFAULT_CAMERA_CONFIG = {
            'distance': 6.0,
            'azimuth': 140.0,
            'elevation': -30.0,
            'type': cam_type,
            'fixedcamid': camera_select
        }

        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
