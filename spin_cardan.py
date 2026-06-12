from pathlib import Path
import math
import time

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).with_name("Leg_assembly.xml")
THIGH_KNEE_MIN_ANGLE = 0.0
THIGH_KNEE_MAX_ANGLE = 0.0
THIGH_YAW_ROLL_MIN_ANGLE = 0.0
THIGH_YAW_ROLL_MAX_ANGLE = 0.0
CARDAN_INITIAL_ANGLE = 0.052999999999999936
SHANK_INITIAL_ANGLE = 0.0
CRANK_RIGHT_IN_INITIAL_ANGLE = -0.11
CRANK_RIGHT_OUT_INITIAL_ANGLE = 0.188
CRANK_SWEEP_AMPLITUDE = 0.5
CRANK_ROTATION_SPEED_RPS = 0.1
THIGH_YAW_ROLL_SWEEP_SPEED_RPS = 0.03
CRANK_KP = 8.0
CRANK_KD = 1.5
CRANK_TORQUE_LIMIT = 3.0
THIGH_KNEE_TARGET_ANGLE = 0.0
THIGH_KNEE_KP = 60.0
THIGH_KNEE_KD = 8.0
THIGH_KNEE_TORQUE_LIMIT = 16.0
THIGH_YAW_ROLL_KP = 60.0
THIGH_YAW_ROLL_KD = 8.0
THIGH_YAW_ROLL_TORQUE_LIMIT = 16.0
SIM_STEPS_PER_VIEWER_SYNC = 4


def opposite_crank_offset(elapsed: float, amplitude: float, speed_rps: float) -> float:
    angular_speed = math.tau * speed_rps
    travel = (angular_speed * elapsed) % (4.0 * amplitude)
    if travel <= amplitude:
        return travel
    if travel <= 3.0 * amplitude:
        return 2.0 * amplitude - travel
    return travel - 4.0 * amplitude


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def sweep_between(elapsed: float, minimum: float, maximum: float, speed_rps: float) -> float:
    if maximum <= minimum:
        return minimum
    phase = (elapsed * speed_rps) % 1.0
    triangle = 1.0 - abs(2.0 * phase - 1.0)
    return minimum + (maximum - minimum) * triangle


def pd_torque(
    target: float,
    position: float,
    velocity: float,
    kp: float,
    kd: float,
    torque_limit: float,
) -> float:
    return clamp(kp * (target - position) - kd * velocity, torque_limit)


def apply_crank_torques(
    data: mujoco.MjData,
    crank_in_qpos_addr: int,
    crank_in_dof_addr: int,
    crank_in_target: float,
    crank_out_qpos_addr: int,
    crank_out_dof_addr: int,
    crank_out_target: float,
    thigh_knee_qpos_addr: int,
    thigh_knee_dof_addr: int,
    thigh_knee_target: float,
    thigh_yaw_roll_qpos_addr: int,
    thigh_yaw_roll_dof_addr: int,
    thigh_yaw_roll_target: float,
) -> None:
    clamped_thigh_knee_target = max(
        THIGH_KNEE_MIN_ANGLE,
        min(THIGH_KNEE_MAX_ANGLE, thigh_knee_target),
    )
    clamped_thigh_yaw_roll_target = max(
        THIGH_YAW_ROLL_MIN_ANGLE,
        min(THIGH_YAW_ROLL_MAX_ANGLE, thigh_yaw_roll_target),
    )
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[crank_in_dof_addr] = pd_torque(
        crank_in_target,
        float(data.qpos[crank_in_qpos_addr]),
        float(data.qvel[crank_in_dof_addr]),
        CRANK_KP,
        CRANK_KD,
        CRANK_TORQUE_LIMIT,
    )
    data.qfrc_applied[crank_out_dof_addr] = pd_torque(
        crank_out_target,
        float(data.qpos[crank_out_qpos_addr]),
        float(data.qvel[crank_out_dof_addr]),
        CRANK_KP,
        CRANK_KD,
        CRANK_TORQUE_LIMIT,
    )
    data.qfrc_applied[thigh_knee_dof_addr] = pd_torque(
        clamped_thigh_knee_target,
        float(data.qpos[thigh_knee_qpos_addr]),
        float(data.qvel[thigh_knee_dof_addr]),
        THIGH_KNEE_KP,
        THIGH_KNEE_KD,
        THIGH_KNEE_TORQUE_LIMIT,
    )
    data.qfrc_applied[thigh_yaw_roll_dof_addr] = pd_torque(
        clamped_thigh_yaw_roll_target,
        float(data.qpos[thigh_yaw_roll_qpos_addr]),
        float(data.qvel[thigh_yaw_roll_dof_addr]),
        THIGH_YAW_ROLL_KP,
        THIGH_YAW_ROLL_KD,
        THIGH_YAW_ROLL_TORQUE_LIMIT,
    )


def require_joint(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        raise RuntimeError(f"Joint '{name}' was not found in the model.")
    return model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]


def require_joint_range(model: mujoco.MjModel, name: str) -> tuple[float, float]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        raise RuntimeError(f"Joint '{name}' was not found in the model.")
    if not model.jnt_limited[joint_id]:
        raise RuntimeError(f"Joint '{name}' does not have limits in the model.")
    joint_min, joint_max = model.jnt_range[joint_id]
    return float(joint_min), float(joint_max)


def require_site(model: mujoco.MjModel, name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id == -1:
        raise RuntimeError(f"Site '{name}' was not found in the model.")
    return site_id


def require_body(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        raise RuntimeError(f"Body '{name}' was not found in the model.")
    return body_id


def align_rod_to_site(
    data: mujoco.MjData,
    rod_body_id: int,
    crank_site_id: int,
    rod_x_qpos_addr: int,
    rod_y_qpos_addr: int,
) -> None:
    rod_base = data.xpos[rod_body_id]
    crank_anchor = data.site_xpos[crank_site_id]
    delta_x = float(crank_anchor[0] - rod_base[0])
    delta_y = float(crank_anchor[1] - rod_base[1])
    delta_z = float(crank_anchor[2] - rod_base[2])
    distance = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
    if distance <= 1e-9:
        raise RuntimeError("Cannot align rod: crank anchor coincides with rod base.")

    direction_x = delta_x / distance
    direction_y = delta_y / distance
    direction_z = delta_z / distance
    data.qpos[rod_y_qpos_addr] = math.asin(max(-1.0, min(1.0, direction_x)))
    data.qpos[rod_x_qpos_addr] = math.atan2(-direction_y, direction_z)


def main() -> None:
    global THIGH_KNEE_MIN_ANGLE, THIGH_KNEE_MAX_ANGLE
    global THIGH_YAW_ROLL_MIN_ANGLE, THIGH_YAW_ROLL_MAX_ANGLE

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    cardan_qpos_addr, _ = require_joint(model, "cardan_y")
    shank_qpos_addr, _ = require_joint(model, "shank_right_y")
    thigh_knee_qpos_addr, thigh_knee_dof_addr = require_joint(model, "thigh_knee_yaw_y")
    THIGH_KNEE_MIN_ANGLE, THIGH_KNEE_MAX_ANGLE = require_joint_range(model, "thigh_knee_yaw_y")
    thigh_yaw_roll_qpos_addr, thigh_yaw_roll_dof_addr = require_joint(model, "thigh_yaw_roll_y")
    THIGH_YAW_ROLL_MIN_ANGLE, THIGH_YAW_ROLL_MAX_ANGLE = require_joint_range(model, "thigh_yaw_roll_y")
    crank_in_qpos_addr, crank_in_dof_addr = require_joint(model, "crank_right_in_z")
    crank_out_qpos_addr, crank_out_dof_addr = require_joint(model, "crank_right_out_z")
    rod1_x_qpos_addr, _ = require_joint(model, "rod1_x")
    rod1_y_qpos_addr, _ = require_joint(model, "rod1_y")
    rod2_x_qpos_addr, _ = require_joint(model, "rod2_x")
    rod2_y_qpos_addr, _ = require_joint(model, "rod2_y")
    crank_in_site_id = require_site(model, "crank_right_in_rod2_site")
    crank_out_site_id = require_site(model, "crank_right_out_rod1_site")
    rod1_body_id = require_body(model, "rod1_body")
    rod2_body_id = require_body(model, "rod2_body")
    period = model.opt.timestep

    data.qpos[cardan_qpos_addr] = CARDAN_INITIAL_ANGLE
    data.qpos[shank_qpos_addr] = SHANK_INITIAL_ANGLE
    data.qpos[thigh_knee_qpos_addr] = THIGH_KNEE_TARGET_ANGLE
    data.qpos[thigh_yaw_roll_qpos_addr] = THIGH_YAW_ROLL_MIN_ANGLE
    data.qpos[crank_in_qpos_addr] = CRANK_RIGHT_IN_INITIAL_ANGLE
    data.qpos[crank_out_qpos_addr] = CRANK_RIGHT_OUT_INITIAL_ANGLE
    mujoco.mj_forward(model, data)
    align_rod_to_site(
        data,
        rod1_body_id,
        crank_out_site_id,
        rod1_x_qpos_addr,
        rod1_y_qpos_addr,
    )
    align_rod_to_site(
        data,
        rod2_body_id,
        crank_in_site_id,
        rod2_x_qpos_addr,
        rod2_y_qpos_addr,
    )
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        start_time = time.perf_counter()
        while viewer.is_running():
            elapsed = time.perf_counter() - start_time
            crank_offset = opposite_crank_offset(
                elapsed,
                CRANK_SWEEP_AMPLITUDE,
                CRANK_ROTATION_SPEED_RPS,
            )
            thigh_yaw_roll_target = sweep_between(
                elapsed,
                THIGH_YAW_ROLL_MIN_ANGLE,
                THIGH_YAW_ROLL_MAX_ANGLE,
                THIGH_YAW_ROLL_SWEEP_SPEED_RPS,
            )
            crank_in_target = CRANK_RIGHT_IN_INITIAL_ANGLE + crank_offset
            crank_out_target = CRANK_RIGHT_OUT_INITIAL_ANGLE - crank_offset

            for _ in range(SIM_STEPS_PER_VIEWER_SYNC):
                apply_crank_torques(
                    data,
                    crank_in_qpos_addr,
                    crank_in_dof_addr,
                    crank_in_target,
                    crank_out_qpos_addr,
                    crank_out_dof_addr,
                    crank_out_target,
                    thigh_knee_qpos_addr,
                    thigh_knee_dof_addr,
                    THIGH_KNEE_TARGET_ANGLE,
                    thigh_yaw_roll_qpos_addr,
                    thigh_yaw_roll_dof_addr,
                    thigh_yaw_roll_target,
                )
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(period * SIM_STEPS_PER_VIEWER_SYNC)


if __name__ == "__main__":
    main()