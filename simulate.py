from koko_gym import KokoReacherEnv
from glfw import get_framebuffer_size
import random
import numpy as np

#Make reacher env instance
reacher = KokoReacherEnv()
reacher.reset_model()

#Set the viewer
width, height = get_framebuffer_size(reacher.viewer.window)
reacher.viewer_setup(camera_type='global_cam', camera_select=0)

# Sample propotional controller should be replaced with your policy function
Kp = 1.0
target_state = reacher.sim.get_state()

for i in range(5000):
    #Get the current state info
    current_state = reacher.sim.get_state()

    # Sample controller (Pseudo Policy Function)
    target_state.qpos[0] = 0.5*np.sin(i/500)        # base_roll_joint
    target_state.qpos[1] = 0.5*np.sin(i/500)        # shoulder_lift_joint
    target_state.qpos[2] = 0.5*np.sin(i/500)        # shoulder_roll_joint
    target_state.qpos[3] = 0.5*np.sin(i/500)        # elbow_lift_joint
    target_state.qpos[4] = 0.5*np.sin(i/500)        # elbow_roll_joint
    target_state.qpos[5] = 0.5*np.sin(i/500)        # wrist_lift_joint
    target_state.qpos[6] = 0.5*np.sin(i/500)        # wrist_roll_joint
    target_state.qpos[7] = 1.0*np.sin(i/500)        # robotfinger_actuator_joint       
    feedback_cmd = Kp * (target_state.qpos - current_state.qpos)
    
    #Adding Step to model
    ob, _, _, _ = reacher.step(a=feedback_cmd[:8]) #ob = qpos numpy.ndarray len=8
    reacher.render(mode='human', width=width, height=height)


