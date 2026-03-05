import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import random


class SpiderEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        super().__init__()

        self.render_mode = render_mode
        self.max_lidar_distance = 5.0
        self.num_rays = 360
        self.position = [0, 0, 0]

        # Action smoothing memory
        self.prev_action = np.zeros(12, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0,
            high=self.max_lidar_distance,
            shape=(372,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        if render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.robot_id = None
        self.obstacles = []

        self.reset()

    # -------------------------
    # WORLD CREATION
    # -------------------------

    def _create_robot(self):
        box_size = [0.2, 0.2, 0.1]

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=box_size
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=box_size,
            rgbaColor=[0, 0, 1, 1]
        )

        robot_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, 0.2]
        )

        # Damping prevents endless sliding/spinning
        p.changeDynamics(robot_id, -1,
                         linearDamping=0.2,
                         angularDamping=0.2)

        return robot_id

    def _create_wall(self, center_position, length=5, thickness=0.1, height=1.0):
        half_extents = [length / 2, thickness / 2, height / 2]

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.6, 0.6, 0.6, 1]
        )

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center_position
        )

    def _create_wall90(self, center_position, length=5, thickness=0.1, height=1.0):
        half_extents = [length / 2, thickness / 2, height / 2]
        orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.6, 0.6, 0.6, 1]
        )

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center_position,
            baseOrientation=orientation
        )

    def _create_obstacle(self, position):
        half_extents = [0.2, 0.2, 0.5]

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[1, 0, 0, 1]
        )

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position
        )

    # -------------------------
    # LIDAR
    # -------------------------

    def _get_lidar(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)

        rays_from = []
        rays_to = []

        for angle in np.linspace(0, 2*np.pi, self.num_rays, endpoint=False):
            dx = self.max_lidar_distance * np.cos(angle)
            dy = self.max_lidar_distance * np.sin(angle)

            rays_from.append(base_pos)
            rays_to.append([
                base_pos[0] + dx,
                base_pos[1] + dy,
                base_pos[2]
            ])

        results = p.rayTestBatch(rays_from, rays_to)

        distances = []
        for r in results:
            hit_fraction = r[2]
            distances.append(hit_fraction * self.max_lidar_distance)

        return np.array(distances, dtype=np.float32)

    # -------------------------
    # GYM API
    # -------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.robot_id = self._create_robot()
        self.prev_action = np.zeros(12, dtype=np.float32)
        
        self.obstacles = []
        self.obstacles.append(self._create_wall([2.5, 3, 0.5], length=7))
        self.obstacles.append(self._create_wall90([-1, 0, 0.5], length=6))
        self.obstacles.append(self._create_wall([2.5, -3, 0.5], length=7))
        self.obstacles.append(self._create_wall90([6, 0, 0.5], length=6))

        
        for _ in range(5):
            x = random.uniform(1, 5)
            y = random.uniform(-3, 3)
            self.obstacles.append(self._create_obstacle([x, y, 0.5]))

        self.position = p.getBasePositionAndOrientation(self.robot_id)[0]

        return self._get_observation(), {}

    def _get_observation(self):
        lidar = self._get_lidar()
        joint_positions = np.zeros(12, dtype=np.float32)
        return np.concatenate([lidar, joint_positions])
    def step(self, action):

        terminated = False

        # --- Smooth actions
        alpha = 0.2
        action = alpha * self.prev_action + (1 - alpha) * action
        self.prev_action = action

        forward_signal = np.mean(action[0:4])
        lateral_signal = np.mean(action[4:8])
        rotation_signal = np.mean(action[8:12])

        max_linear_speed = 2.0
        max_angular_speed = 1.0

        vx = forward_signal * max_linear_speed
        vy = lateral_signal * max_linear_speed
        wz = np.clip(rotation_signal * max_angular_speed, -1.0, 1.0)

        # Direct velocity control
        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[vx, vy, 0],
            angularVelocity=[0, 0, wz]
        )

        p.stepSimulation()

        # --- Collision check
        collision = False
        for obs_id in self.obstacles:
            if len(p.getContactPoints(self.robot_id, obs_id)) > 0:
                collision = True
                break

        new_position = p.getBasePositionAndOrientation(self.robot_id)[0]
        progress = new_position[0] - self.position[0]

        # ---------- Reward computation ----------

        # Forward progress reward (stronger)
        reward = progress * 8.0

        # Confidence forward reward when path is clear
        lidar = self._get_lidar()
        front_lidar = lidar[:30]

        if np.min(front_lidar) > 1.5:
            reward += 0.5 * max(0, vx)

        # Goal reward
        if new_position[0] >= 5.0:
            reward += 30.0
            terminated = True

        # Collision penalty
        elif collision:
            reward -= 5.0
            terminated = True

        else:
            # Proximity safety shaping
            min_distance = np.min(lidar)

            safe_distance = 0.5
            if min_distance < safe_distance:
                reward -= (safe_distance - min_distance) * 2.0

            # Time penalty (small)
            reward -= 0.01

        self.position = new_position

        return self._get_observation(), reward, terminated, False, {}
    def render(self):
        pass

    def close(self):
        p.disconnect()