"""Fixed-base 6-DoF table arm MJCF generator (self-contained)."""
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
    return f"""<mujoco model="table_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default><joint damping="0.8" armature="0.03"/><geom friction="1.0 0.05 0.01" condim="3"/></default>
  <worldbody>
{arena}
    <geom name="table_top" type="box" pos="0 0 0.38" size="0.55 0.55 0.04" rgba="0.45 0.35 0.25 1"/>
    <body name="arm_base" pos="0 0 0.42">
      <inertial pos="0 0 0" mass="3" diaginertia="0.03 0.03 0.03"/>
      <geom name="arm_base_geom" type="cylinder" size="0.11 0.06" rgba="0.25 0.3 0.35 1"/>
      <camera name="ego_cam" pos="0.15 0 0.35" xyaxes="0 -1 0 0.3 0 1" fovy="65"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="base_yaw" type="hinge" axis="0 0 1" range="-3.0 3.0"/>
        <inertial pos="0 0 0" mass="1.2" diaginertia="0.01 0.01 0.01"/>
        <geom name="link1_geom" type="cylinder" pos="0 0 0.08" size="0.07 0.09" rgba="0.2 0.55 0.85 1"/>
        <body name="link2" pos="0 0 0.17">
          <joint name="shoulder_pitch" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
          <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.008 0.005"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0.34 0 0.10" size="0.045" rgba="0.2 0.55 0.85 1"/>
          <body name="link3" pos="0.34 0 0.10">
            <joint name="elbow_pitch" type="hinge" axis="0 1 0" range="-2.4 2.4"/>
            <inertial pos="0 0 0" mass="0.8" diaginertia="0.008 0.006 0.004"/>
            <geom name="link3_geom" type="capsule" fromto="0 0 0 0.30 0 -0.04" size="0.04" rgba="0.4 0.7 0.95 1"/>
            <body name="link4" pos="0.30 0 -0.04">
              <joint name="wrist_roll" type="hinge" axis="1 0 0" range="-3.0 3.0"/>
              <joint name="wrist_pitch" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
              <inertial pos="0 0 0" mass="0.4" diaginertia="0.003 0.003 0.002"/>
              <geom name="link4_geom" type="capsule" fromto="0 0 0 0.16 0 0" size="0.03" rgba="0.5 0.75 0.95 1"/>
              <body name="tool" pos="0.16 0 0">
                <joint name="tool_roll" type="hinge" axis="1 0 0" range="-3.0 3.0"/>
                <inertial pos="0 0 0" mass="0.25" diaginertia="0.002 0.002 0.001"/>
                <geom name="rake_geom" type="box" pos="0.08 0 -0.02" size="0.09 0.07 0.015" rgba="0.9 0.65 0.15 1"/>
                <geom name="catapult_geom" type="box" pos="0.03 0 0.03" size="0.05 0.04 0.03" rgba="0.9 0.3 0.2 1"/>
                <site name="tool_end_effector" pos="0.16 0 0" size="0.02" rgba="0 1 0 1"/>
                <site name="rake_site" pos="0.17 0 -0.02" size="0.015"/>
                <site name="gripper_site" pos="0.16 0 0" size="0.015"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="base_yaw_act" joint="base_yaw" kp="120" ctrlrange="-3.0 3.0"/>
    <position name="shoulder_pitch_act" joint="shoulder_pitch" kp="120" ctrlrange="-2.2 2.2"/>
    <position name="elbow_pitch_act" joint="elbow_pitch" kp="100" ctrlrange="-2.4 2.4"/>
    <position name="wrist_roll_act" joint="wrist_roll" kp="60" ctrlrange="-3.0 3.0"/>
    <position name="wrist_pitch_act" joint="wrist_pitch" kp="60" ctrlrange="-2.0 2.0"/>
    <position name="tool_roll_act" joint="tool_roll" kp="40" ctrlrange="-3.0 3.0"/>
  </actuator>
  <sensor><jointpos name="shoulder_pos" joint="shoulder_pitch"/></sensor>
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
MODEL_NAME = "table_arm"
NAME = "table_arm"
