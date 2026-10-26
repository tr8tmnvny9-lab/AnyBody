"""Quadruped dog with dorsal manipulator arm MJCF generator (self-contained)."""
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

def _leg(prefix, x, y):
    side = "left" if y > 0 else "right"
    return f"""
      <body name="{prefix}_upper_leg" pos="{x} {y} -0.05">
        <joint name="{prefix}_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <joint name="{prefix}_hip_abduction" type="hinge" axis="1 0 0" range="-0.5 0.5"/>
        <inertial pos="0 0 0" mass="0.8" diaginertia="0.008 0.008 0.004"/>
        <geom name="{prefix}_upper_leg_geom" type="capsule" fromto="0 0 0 0 0 -0.24" size="0.04" rgba="0.35 0.35 0.4 1"/>
        <body name="{prefix}_lower_leg" pos="0 0 -0.24">
          <joint name="{prefix}_knee" type="hinge" axis="0 1 0" range="-2.0 0"/>
          <inertial pos="0 0 0" mass="0.5" diaginertia="0.005 0.005 0.003"/>
          <geom name="{prefix}_lower_leg_geom" type="capsule" fromto="0 0 0 0 0 -0.24" size="0.03" rgba="0.4 0.4 0.45 1"/>
          <geom name="{prefix}_foot_geom" type="sphere" pos="0 0 -0.26" size="0.04" rgba="0.1 0.1 0.1 1"/>
          <site name="{prefix}_foot_site" pos="0 0 -0.28" size="0.012"/>
        </body>
      </body>"""

def build_mjcf(include_arena: bool = True) -> str:
    arena = _ARENA if include_arena else '<geom name="floor" type="plane" size="5 5 0.1" rgba="0.2 0.2 0.2 1"/><camera name="exo_cam" pos="2.8 -2.8 2.0" xyaxes="1 1 0 -0.4 0.4 1"/>'
    legs = "".join([
        _leg("front_left", 0.24, 0.15), _leg("front_right", 0.24, -0.15),
        _leg("rear_left", -0.24, 0.15), _leg("rear_right", -0.24, -0.15)])
    leg_acts = ""
    for p in ["front_left", "front_right", "rear_left", "rear_right"]:
        leg_acts += f'<position name="{p}_hip_act" joint="{p}_hip" kp="120" ctrlrange="-1.2 1.2"/>'
        leg_acts += f'<position name="{p}_hip_abduction_act" joint="{p}_hip_abduction" kp="80" ctrlrange="-0.5 0.5"/>'
        leg_acts += f'<position name="{p}_knee_act" joint="{p}_knee" kp="120" ctrlrange="-2.0 0"/>'
    return f"""<mujoco model="quadruped_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default><joint damping="0.5" armature="0.02"/><geom friction="1.0 0.05 0.01" condim="3"/></default>
  <worldbody>
{arena}
    <body name="trunk" pos="0 0 0.56">
      <freejoint name="root_free"/>
      <inertial pos="0 0 0" mass="7" diaginertia="0.12 0.18 0.08"/>
      <geom name="trunk_geom" type="box" size="0.34 0.15 0.11" rgba="0.45 0.45 0.5 1"/>
      <geom name="head_geom" type="box" pos="0.42 0 0.06" size="0.11 0.09 0.09" rgba="0.5 0.5 0.55 1"/>
      <camera name="ego_cam" pos="0.52 0 0.08" xyaxes="0 -1 0 0 0 1" fovy="70"/>
      <site name="head_site" pos="0.53 0 0.06" size="0.015"/>
      {legs}
      <body name="dorsal_base" pos="-0.05 0 0.13">
        <joint name="dorsal_yaw" type="hinge" axis="0 0 1" range="-2.5 2.5"/>
        <inertial pos="0 0 0" mass="0.6" diaginertia="0.005 0.005 0.005"/>
        <geom name="dorsal_base_geom" type="cylinder" size="0.07 0.05" rgba="0.2 0.6 0.9 1"/>
        <body name="dorsal_upper" pos="0 0 0.06">
          <joint name="dorsal_shoulder" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
          <inertial pos="0 0 0" mass="0.5" diaginertia="0.004 0.004 0.002"/>
          <geom name="dorsal_upper_geom" type="capsule" fromto="0 0 0 0.22 0 0.08" size="0.03" rgba="0.2 0.6 0.9 1"/>
          <body name="dorsal_forearm" pos="0.22 0 0.08">
            <joint name="dorsal_elbow" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
            <inertial pos="0 0 0" mass="0.35" diaginertia="0.003 0.003 0.002"/>
            <geom name="dorsal_forearm_geom" type="capsule" fromto="0 0 0 0.20 0 -0.04" size="0.025" rgba="0.4 0.7 0.95 1"/>
            <body name="dorsal_wrist_body" pos="0.20 0 -0.04">
              <joint name="dorsal_wrist" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
              <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
              <geom name="dorsal_gripper_geom" type="box" pos="0.04 0 0" size="0.05 0.04 0.02" rgba="0.9 0.7 0.2 1"/>
              <site name="dorsal_end_effector" pos="0.09 0 0" size="0.02" rgba="0 1 0 1"/>
              <site name="dorsal_gripper_site" pos="0.10 0 0" size="0.015"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    {leg_acts}
    <position name="dorsal_yaw_act" joint="dorsal_yaw" kp="40" ctrlrange="-2.5 2.5"/>
    <position name="dorsal_shoulder_act" joint="dorsal_shoulder" kp="40" ctrlrange="-1.8 1.8"/>
    <position name="dorsal_elbow_act" joint="dorsal_elbow" kp="35" ctrlrange="-2.2 2.2"/>
    <position name="dorsal_wrist_act" joint="dorsal_wrist" kp="25" ctrlrange="-1.8 1.8"/>
  </actuator>
  <sensor><gyro name="trunk_gyro" site="head_site"/></sensor>
</mujoco>"""

get_mjcf = build_mjcf
mjcf_xml = build_mjcf
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
MODEL_NAME = "quadruped_arm"
NAME = "quadruped_arm"
