"""Bipedal humanoid MJCF generator (self-contained, arena included)."""
from __future__ import annotations
import mujoco

_ARENA = """
    <light name="key_light" pos="0 -2 3" dir="0 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="exo_cam" pos="2.8 -2.8 2.0" xyaxes="1 1 0 -0.4 0.4 1" fovy="55"/>
    <camera name="overhead_cam" pos="0 0 4.5" xyaxes="1 0 0 0 1 0" fovy="60"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.18 0.2 0.24 1" friction="1.2 0.05 0.01"/>
    <body name="ball" pos="0.6 0.3 0.11">
      <freejoint name="ball_free"/><inertial pos="0 0 0" mass="0.3" diaginertia="0.001 0.001 0.001"/>
      <geom name="ball_geom" type="sphere" size="0.11" rgba="1 0.1 0.1 1" friction="0.8 0.05 0.01"/>
      <site name="ball_site" pos="0 0 0" size="0.02" rgba="1 0 0 1"/>
    </body>
    <body name="person_target" pos="2.0 0 0">
      <inertial pos="0 0 0" mass="5" diaginertia="0.1 0.1 0.1"/>
      <geom name="person_torso" type="capsule" fromto="0 0 0.55 0 0 1.35" size="0.16" rgba="0.2 0.5 0.9 1"/>
      <geom name="person_head" type="sphere" pos="0 0 1.55" size="0.13" rgba="0.9 0.75 0.6 1"/>
      <geom name="person_leg_l" type="capsule" fromto="-0.09 0 0.05 -0.09 0 0.55" size="0.06" rgba="0.2 0.2 0.3 1"/>
      <geom name="person_leg_r" type="capsule" fromto="0.09 0 0.05 0.09 0 0.55" size="0.06" rgba="0.2 0.2 0.3 1"/>
      <site name="person_site" pos="0 0 1.1" size="0.04" rgba="0 0.5 1 1"/>
    </body>
    <body name="obstacle_1" pos="1.0 0.7 0.15"><inertial pos="0 0 0" mass="2" diaginertia="0.01 0.01 0.01"/>
      <geom name="obstacle_1_geom" type="box" size="0.15 0.15 0.15" rgba="0.9 0.6 0.1 1"/></body>
    <body name="obstacle_2" pos="1.2 -0.6 0.22"><inertial pos="0 0 0" mass="2" diaginertia="0.01 0.01 0.01"/>
      <geom name="obstacle_2_geom" type="cylinder" size="0.14 0.22" rgba="0.5 0.5 0.6 1"/></body>
    <body name="target_ring" pos="2.0 0 1.1"><inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom name="target_ring_geom" type="cylinder" size="0.28 0.03" rgba="0.1 0.9 0.3 0.6"/>
      <site name="target_ring_site" pos="0 0 0" size="0.03"/></body>
"""

def build_mjcf(include_arena: bool = True) -> str:
    arena = _ARENA if include_arena else '<geom name="floor" type="plane" size="5 5 0.1" rgba="0.2 0.2 0.2 1"/><camera name="exo_cam" pos="2.8 -2.8 2.0" xyaxes="1 1 0 -0.4 0.4 1"/>'
    return f"""<mujoco model="humanoid">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default><joint damping="0.6" armature="0.02"/><geom friction="1.0 0.05 0.01" condim="3"/></default>
  <worldbody>
{arena}
    <body name="pelvis" pos="0 0 0.95">
      <freejoint name="root_free"/>
      <inertial pos="0 0 0" mass="6" diaginertia="0.08 0.08 0.05"/>
      <geom name="pelvis_geom" type="box" size="0.13 0.10 0.07" rgba="0.3 0.35 0.4 1"/>
      <body name="torso" pos="0 0 0.12">
        <joint name="torso_pitch" type="hinge" axis="0 1 0" range="-0.5 0.7"/>
        <inertial pos="0 0 0" mass="8" diaginertia="0.15 0.15 0.08"/>
        <geom name="torso_geom" type="box" pos="0 0 0.18" size="0.15 0.10 0.22" rgba="0.25 0.5 0.8 1"/>
        <body name="head" pos="0 0 0.48">
          <joint name="neck_yaw" type="hinge" axis="0 0 1" range="-1.2 1.2"/>
          <joint name="neck_pitch" type="hinge" axis="0 1 0" range="-0.6 0.6"/>
          <inertial pos="0 0 0" mass="1.2" diaginertia="0.01 0.01 0.01"/>
          <geom name="head_geom" type="sphere" pos="0 0 0.08" size="0.11" rgba="0.9 0.8 0.7 1"/>
          <camera name="ego_cam" pos="0.10 0 0.10" xyaxes="0 -1 0 0 0 1" fovy="70"/>
          <site name="head_site" pos="0.11 0 0.08" size="0.015"/>
        </body>
        <body name="left_upper_arm" pos="0 0.16 0.32">
          <joint name="left_shoulder_pitch" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
          <joint name="left_shoulder_roll" type="hinge" axis="1 0 0" range="-1.2 1.2"/>
          <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.005"/>
          <geom name="left_upper_arm_geom" type="capsule" fromto="0 0 0 0 0.03 -0.28" size="0.04" rgba="0.25 0.5 0.8 1"/>
          <body name="left_forearm" pos="0 0.03 -0.28">
            <joint name="left_elbow" type="hinge" axis="0 1 0" range="-2.2 0"/>
            <inertial pos="0 0 0" mass="0.7" diaginertia="0.008 0.008 0.004"/>
            <geom name="left_forearm_geom" type="capsule" fromto="0 0 0 0 0 -0.26" size="0.035" rgba="0.4 0.65 0.9 1"/>
            <geom name="left_hand_geom" type="sphere" pos="0 0 -0.28" size="0.05" rgba="0.9 0.8 0.7 1"/>
            <site name="left_end_effector" pos="0 0 -0.30" size="0.02" rgba="0 1 0 1"/>
            <site name="left_gripper_site" pos="0 0 -0.32" size="0.015"/>
          </body>
        </body>
        <body name="right_upper_arm" pos="0 -0.16 0.32">
          <joint name="right_shoulder_pitch" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
          <joint name="right_shoulder_roll" type="hinge" axis="1 0 0" range="-1.2 1.2"/>
          <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.005"/>
          <geom name="right_upper_arm_geom" type="capsule" fromto="0 0 0 0 -0.03 -0.28" size="0.04" rgba="0.25 0.5 0.8 1"/>
          <body name="right_forearm" pos="0 -0.03 -0.28">
            <joint name="right_elbow" type="hinge" axis="0 1 0" range="-2.2 0"/>
            <inertial pos="0 0 0" mass="0.7" diaginertia="0.008 0.008 0.004"/>
            <geom name="right_forearm_geom" type="capsule" fromto="0 0 0 0 0 -0.26" size="0.035" rgba="0.4 0.65 0.9 1"/>
            <geom name="right_hand_geom" type="sphere" pos="0 0 -0.28" size="0.05" rgba="0.9 0.8 0.7 1"/>
            <site name="right_end_effector" pos="0 0 -0.30" size="0.02" rgba="0 1 0 1"/>
            <site name="right_gripper_site" pos="0 0 -0.32" size="0.015"/>
          </body>
        </body>
      </body>
      <body name="left_thigh" pos="0 0.10 -0.05">
        <joint name="left_hip_pitch" type="hinge" axis="0 1 0" range="-1.6 1.6"/>
        <joint name="left_hip_roll" type="hinge" axis="1 0 0" range="-0.6 0.6"/>
        <inertial pos="0 0 0" mass="2.0" diaginertia="0.02 0.02 0.01"/>
        <geom name="left_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.055" rgba="0.3 0.3 0.35 1"/>
        <body name="left_shin" pos="0 0 -0.38">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="0 2.0"/>
          <inertial pos="0 0 0" mass="1.2" diaginertia="0.012 0.012 0.006"/>
          <geom name="left_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.045" rgba="0.35 0.35 0.4 1"/>
          <body name="left_foot" pos="0 0 -0.38">
            <joint name="left_ankle" type="hinge" axis="0 1 0" range="-0.7 0.7"/>
            <inertial pos="0 0 0" mass="0.4" diaginertia="0.003 0.003 0.002"/>
            <geom name="left_foot_geom" type="box" pos="0.05 0 -0.03" size="0.11 0.05 0.03" rgba="0.15 0.15 0.15 1"/>
            <site name="left_foot_site" pos="0.05 0 -0.06" size="0.015"/>
          </body>
        </body>
      </body>
      <body name="right_thigh" pos="0 -0.10 -0.05">
        <joint name="right_hip_pitch" type="hinge" axis="0 1 0" range="-1.6 1.6"/>
        <joint name="right_hip_roll" type="hinge" axis="1 0 0" range="-0.6 0.6"/>
        <inertial pos="0 0 0" mass="2.0" diaginertia="0.02 0.02 0.01"/>
        <geom name="right_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.055" rgba="0.3 0.3 0.35 1"/>
        <body name="right_shin" pos="0 0 -0.38">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="0 2.0"/>
          <inertial pos="0 0 0" mass="1.2" diaginertia="0.012 0.012 0.006"/>
          <geom name="right_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.045" rgba="0.35 0.35 0.4 1"/>
          <body name="right_foot" pos="0 0 -0.38">
            <joint name="right_ankle" type="hinge" axis="0 1 0" range="-0.7 0.7"/>
            <inertial pos="0 0 0" mass="0.4" diaginertia="0.003 0.003 0.002"/>
            <geom name="right_foot_geom" type="box" pos="0.05 0 -0.03" size="0.11 0.05 0.03" rgba="0.15 0.15 0.15 1"/>
            <site name="right_foot_site" pos="0.05 0 -0.06" size="0.015"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="torso_pitch_act" joint="torso_pitch" kp="200" ctrlrange="-0.5 0.7"/>
    <position name="neck_yaw_act" joint="neck_yaw" kp="40" ctrlrange="-1.2 1.2"/>
    <position name="neck_pitch_act" joint="neck_pitch" kp="40" ctrlrange="-0.6 0.6"/>
    <position name="left_shoulder_pitch_act" joint="left_shoulder_pitch" kp="60" ctrlrange="-2.2 2.2"/>
    <position name="left_shoulder_roll_act" joint="left_shoulder_roll" kp="60" ctrlrange="-1.2 1.2"/>
    <position name="left_elbow_act" joint="left_elbow" kp="50" ctrlrange="-2.2 0"/>
    <position name="right_shoulder_pitch_act" joint="right_shoulder_pitch" kp="60" ctrlrange="-2.2 2.2"/>
    <position name="right_shoulder_roll_act" joint="right_shoulder_roll" kp="60" ctrlrange="-1.2 1.2"/>
    <position name="right_elbow_act" joint="right_elbow" kp="50" ctrlrange="-2.2 0"/>
    <position name="left_hip_pitch_act" joint="left_hip_pitch" kp="200" ctrlrange="-1.6 1.6"/>
    <position name="left_hip_roll_act" joint="left_hip_roll" kp="150" ctrlrange="-0.6 0.6"/>
    <position name="left_knee_act" joint="left_knee" kp="180" ctrlrange="0 2.0"/>
    <position name="left_ankle_act" joint="left_ankle" kp="100" ctrlrange="-0.7 0.7"/>
    <position name="right_hip_pitch_act" joint="right_hip_pitch" kp="200" ctrlrange="-1.6 1.6"/>
    <position name="right_hip_roll_act" joint="right_hip_roll" kp="150" ctrlrange="-0.6 0.6"/>
    <position name="right_knee_act" joint="right_knee" kp="180" ctrlrange="0 2.0"/>
    <position name="right_ankle_act" joint="right_ankle" kp="100" ctrlrange="-0.7 0.7"/>
  </actuator>
  <sensor>
    <jointpos name="left_knee_pos" joint="left_knee"/><jointpos name="right_knee_pos" joint="right_knee"/>
    <gyro name="pelvis_gyro" site="head_site"/>
  </sensor>
</mujoco>"""

# Aliases for flexible callers
get_mjcf = build_mjcf
mjcf_xml = build_mjcf
MJCF_XML = None  # populated lazily below to avoid divergence
def create_model():
    return mujoco.MjModel.from_xml_string(build_mjcf())
def build_model():
    return create_model()
def load_model():
    return create_model()
def make_data(model=None):
    return mujoco.MjData(model or create_model())
MJCF_XML = build_mjcf()
XML = MJCF_XML
MODEL_NAME = "humanoid"
NAME = "humanoid"
