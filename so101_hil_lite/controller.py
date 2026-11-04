"""Control ownership and continuous leader/follower transitions."""
import numpy as np
import logging

LOG = logging.getLogger(__name__)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KEYS = tuple(f"{joint}.pos" for joint in JOINTS)


class AlignmentError(ValueError):
    """Leader cannot yet be enabled at the follower's current target."""


def vector(action):
    value = np.array([action[key] for key in KEYS], dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite joint positions")
    return value


def action_dict(value):
    return dict(zip(KEYS, map(float, value), strict=True))


class Controller:
    def __init__(self, follower, leader, policy, max_arm_step=6.51, max_gripper_step=20.0):
        self.follower, self.leader, self.policy = follower, leader, policy
        self.limits = np.array([max_arm_step] * 5 + [max_gripper_step])
        self.manual = True
        self.previous = None
        self.limited_steps = 0
        self.pending_takeover = False
        self.hold_target = None
        self.leader_hold = None

    def start(self, observation):
        current = action_dict(vector(observation))
        # Program a current-position hold before enabling leader torque.
        leader_position = vector(self.leader.get_action())
        follower_position = vector(observation)
        difference = leader_position - follower_position
        if np.any(np.abs(difference) > self.limits):
            rows = ["Align leader to follower (arm: degrees; gripper: 0-100):",
                    "joint                 follower    leader     delta     limit"]
            for name, follower, leader, delta, limit in zip(
                    JOINTS, follower_position, leader_position, difference, self.limits, strict=True):
                mark = " *" if abs(delta) > limit else ""
                rows.append(f"{name:20} {follower:9.2f} {leader:9.2f} {delta:9.2f} {limit:9.2f}{mark}")
            raise AlignmentError("\n".join(rows))
        self.leader.send_feedback(current)
        self.leader.enable_torque()
        self.policy.reset()
        self.previous = vector(observation)
        self.manual = False
        self.pending_takeover = False

    def toggle(self, observation):
        self.policy.reset()
        if self.pending_takeover:
            return
        if not self.manual:
            self.hold_target = vector(observation)
            self.leader_hold = vector(self.leader.get_action())
            self.previous = self.hold_target.copy()
            self.pending_takeover = True
        else:
            self.start(observation)

    def confirm_takeover(self, observation):
        if not self.pending_takeover:
            return
        self.leader.disable_torque()
        # Read fresh positions after releasing; do not reuse a policy chunk.
        vector(self.leader.get_action())
        self.previous = vector(observation)
        self.manual = True
        self.pending_takeover = False

    def step(self, observation):
        policy_action = None
        if self.pending_takeover:
            selected = self.hold_target.copy()
        elif self.manual:
            selected = vector(self.leader.get_action())
        else:
            policy_action = self.policy.select_action(observation)
            selected = np.asarray(policy_action, dtype=np.float32)
        if selected.shape != (6,) or not np.isfinite(selected).all():
            raise ValueError("Expected six finite targets")
        reference = vector(observation) if self.previous is None else self.previous
        current = vector(observation)
        lower = np.maximum(reference - self.limits, current - self.limits)
        upper = np.minimum(reference + self.limits, current + self.limits)
        if np.any(lower > upper):
            raise ValueError(f"Follower tracking gap exceeds limits: measured-minus-target={current - reference}")
        command = np.clip(selected, lower, upper)
        if np.any(np.abs(command - selected) > 1e-5):
            self.limited_steps += 1
            if self.limited_steps == 1 or self.limited_steps % 30 == 0:
                limited = {name: (round(float(raw), 2), round(float(sent), 2))
                           for name, raw, sent in zip(JOINTS, selected, command, strict=True)
                           if abs(raw - sent) > 1e-5}
                LOG.warning("Rate-limited step %d; raw -> command: %s", self.limited_steps, limited)
        sent = self.follower.send_action(action_dict(command))
        accepted = vector(sent)
        if self.pending_takeover:
            self.leader.send_feedback(action_dict(self.leader_hold))
        elif not self.manual:
            self.leader.send_feedback(action_dict(accepted))
        self.previous = accepted
        return selected, accepted, policy_action, self.manual

    def stop(self):
        self.policy.reset()
        self.leader.disable_torque()
        self.manual = True
        self.pending_takeover = False
