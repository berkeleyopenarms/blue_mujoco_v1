# koko-mujoco

## Requirement
* Mujoco1.55
* OpenAI Gym 
* OpenAI Mujoco-py

## Repo structure
```bash
├── README.md
├── simulate.py
├── train.py
├── assets
│   ├── koko_full.xml
│   └── STL files
└── koko_gym
    └── envs
        ├── assets
        │   ├── koko_reacher.xml
        │   └── STL files
        ├── __init__.py
        └──  koko_reacher.py
```

## Explaination for each file 
* koko_full.xml


MJCF file for the Blue robot. Actuated gripper installed. Having following actuators (joints).
```xml
    <actuator>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="2.0" joint="base_roll_joint" />
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="2.0" joint="shoulder_lift_joint" />
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="2.0" joint="shoulder_roll_joint" />
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="2.0" joint="elbow_lift_joint" />
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="2.0" joint="elbow_roll_joint" />
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="2.0" joint="wrist_lift_joint" />
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="2.0" joint="wrist_roll_joint" />
        <position ctrllimited="true" ctrlrange="0 1.05" gear="1.0" joint="robotfinger_actuator_joint" />
        <position ctrllimited="true" kp="1.0" ctrlrange="0 1.4" joint="right_fingerlimb_joint" />
        <position ctrllimited="true" kp="1.0" ctrlrange="-1.4 0" joint="right_fingertip_joint" />
        <position ctrllimited="true" kp="1.0" ctrlrange="0 1.4" joint="left_fingerlimb_joint" />
        <position ctrllimited="true" kp="1.0" ctrlrange="-1.4 0" joint="left_fingertip_joint" />
    </actuator>
```
Since no URDF `<mimic>` tag equivalent exists in MJCF, the grippers (last four actuators) are actuated by a position controller that takes the current `robotfinger_actuator_joint` angle as an input (`fingerlimb_joint` moves positive and `fingertip_joint` goes negative to make the tips parallel to each other).

* koko_reacher.py


OpenAI Gym environment for Blue. `reacher.step` takes 1x8 size action array. The actuator of the gripper joints cannot be controlled respectively but will be controlled at once using `robotfinger_actuator_joint`'s angle as the position input. You can also set your favorite reward signal in a step function.

* koko_reacher.xml


`koko_full.xml` with target object.

* train.py


Training loop using random controller. Add your favorite algorithm to train the policy.

* simulate.py


Runs the Mujoco-py viewer simulator for 5000 time steps. Use this for test run your trained policy. 

## Reference

* [https://github.com/openai/gym](https://github.com/openai/gym)
* [https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)
* [http://www.mujoco.org/](http://www.mujoco.org/)
