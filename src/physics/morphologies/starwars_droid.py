"""Compact wheeled/bipedal Star Wars-style droid MJCF generator (self-contained)."""
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
    return f"""<mujoco model="starwars_droid">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default><joint damping="0.4" armature="0.02"/><geom friction="1.2 0.05 0.01" condim="3"/></default>
  <worldbody>
{arena}
    <body name="droid_body" pos="0 0 0.52">
      <freejoint name="root_free"/>
      <inertial pos="0 0 0" mass="4" diaginertia="0.06 0.06 0.04"/>
      <geom name="droid_body_geom" type="box" size="0.16 0.13 0.14" rgba="0.85 0.85 0.88 1"/>
      <geom name="droid_belly_geom" type="sphere" pos="0.05 0 -0.08" size="0.12" rgba="0.9 0.55 0.15 1"/>
      <body name="left_wheel_leg" pos="0 0.16 -0.18">
        <joint name="left_leg_pitch" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
        <inertial pos="0 0 0" mass="0.3" diaginertia="0.002 0.002 0.002"/>
        <geom name="left_leg_geom" type="capsule" fromto="0 0 0 0 0.02 -0.16" size="0.03" rgba="0.7 0.7 0.75 1"/>
        <body name="left_wheel" pos="0 0.03 -0.16">
          <joint name="left_wheel_joint" type="hinge" axis="0 1 0"/>
          <inertial pos="0 0 0" mass="0.5" diaginertia="0.005 0.005 0.005"/>
          <geom name="left_wheel_geom" type="cylinder" pos="0 0 0" euler="1.5708 0 0" size="0.17 0.04" rgba="0.12 0.12 0.12 1"/>
          <site name="left_wheel_site" pos="0 0 0" size="0.012"/>
        </body>
      </body>
      <body name="right_wheel_leg" pos="0 -0.16 -0.18">
        <joint name="right_leg_pitch" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
        <inertial pos="0 0 0" mass="0.3" diaginertia="0.002 0.002 0.002"/>
        <geom name="right_leg_geom" type="capsule" fromto="0 0 0 0 -0.02 -0.16" size="0.03" rgba="0.7 0.7 0.75 1"/>
        <body name="right_wheel" pos="0 -0.03 -0.16">
          <joint name="right_wheel_joint" type="hinge" axis="0 1 0"/>
          <inertial pos="0 0 0" mass="0.5" diaginertia="0.005 0.005 0.005"/>
          <geom name="right_wheel_geom" type="cylinder" pos="0 0 0" euler="1.5708 0 0" size="0.17 0.04" rgba="0.12 0.12 0.12 1"/>
          <site name="right_wheel_site" pos="0 0 0" size="0.012"/>
        </body>
      </body>
      <body name="droid_head" pos="0 0 0.20">
        <joint name="head_yaw" type="hinge" axis="0 0 1" range="-2.5 2.5"/>
        <joint name="head_pitch" type="hinge" axis="0 1 0" range="-0.8 0.8"/>
        <inertial pos="0 0 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>
        <geom name="droid_head_geom" type="box" pos="0 0 0.09" size="0.13 0.11 0.09" rgba="0.9 0.9 0.92 1"/>
        <geom name="eye_geom" type="sphere" pos="0.12 0 0.11" size="0.045" rgba="0.1 0.2 0.9 1"/>
        <geom name="antenna_geom" type="capsule" fromto="0 -0.06 0.16 0 -0.06 0.32" size="0.012" rgba="0.3 0.3 0.3 1"/>
        <camera name="ego_cam" pos="0.13 0 0.11" xyaxes="0 -1 0 0 0 1" fovy="75"/>
        <site name="head_site" pos="0.13 0 0.11" size="0.015"/>
      </body>
      <body name="scoop_arm" pos="0.16 -0.10 -0.02">
        <joint name="scoop_pitch" type="hinge" axis="0 1 0" range="-1.8 1.2"/>
        <joint name="scoop_yaw" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
        <inertial pos="0 0 0" mass="0.4" diaginertia="0.003 0.003 0.002"/>
        <geom name="scoop_arm_geom" type="capsule" fromto="0 0 0 0.24 0 -0.06" size="0.03" rgba="0.9 0.55 0.15 1"/>
        <body name="launcher" pos="0.24 0 -0.06">
          <joint name="launcher_pitch" type="hinge" axis="0 1 0" range="-1.2 1.8"/>
          <inertial pos="0 0 0" mass="0.25" diaginertia="0.002 0.002 0.001"/>
          <geom name="launcher_geom" type="box" pos="0.05 0 0" size="0.07 0.06 0.03" rgba="0.9 0.3 0.2 1"/>
          <site name="launcher_end_effector" pos="0.12 0 0" size="0.02" rgba="0 1 0 1"/>
          <site name="scoop_site" pos="0.12 0 0" size="0.015"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_wheel_act" joint="left_wheel_joint" kv="20" ctrlrange="-12 12"/>
    <velocity name="right_wheel_act" joint="right_wheel_joint" kv="20" ctrlrange="-12 12"/>
    <position name="left_leg_pitch_act" joint="left_leg_pitch" kp="60" ctrlrange="-0.5 0.5"/>
    <position name="right_leg_pitch_act" joint="right_leg_pitch" kp="60" ctrlrange="-0.5 0.5"/>
    <position name="head_yaw_act" joint="head_yaw" kp="30" ctrlrange="-2.5 2.5"/>
    <position name="head_pitch_act" joint="head_pitch" kp="30" ctrlrange="-0.8 0.8"/>
    <position name="scoop_pitch_act" joint="scoop_pitch" kp="40" ctrlrange="-1.8 1.2"/>
    <position name="scoop_yaw_act" joint="scoop_yaw" kp="30" ctrlrange="-1.5 1.5"/>
    <position name="launcher_pitch_act" joint="launcher_pitch" kp="40" ctrlrange="-1.2 1.8"/>
  </actuator>
  <sensor><gyro name="droid_gyro" site="head_site"/></sensor>
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
MODEL_NAME = "starwars_droid"
NAME = "starwars_droid"
