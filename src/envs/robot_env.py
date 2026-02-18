import pybullet as p
import time
import pybullet_data
import numpy as np
import math
from env_randomizer import PoissonDisc

# TODOs
# Get max lidar distance
# Add friction to robot parts

class RobotEnv:
    def __init__(self, grid_size: int, num_of_obsticles: int, robot_file: str="spider.urdf", gui: bool=False, obstacle_distance: float=1.3, min_dist_to_goal: float=4.0):

        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        
        self.grid_size_half = grid_size / 2
        self.grid_size_full = grid_size
        self.obstacle_distance = obstacle_distance
        self.num_of_obsticles = num_of_obsticles
        self.min_dist_to_goal = min_dist_to_goal
        self.robot_file = robot_file

        self.joint_indicies = []
        self.foot_indicies = []
        self.obstacle_ids = set()
        self.action_repeat = 8

    def _set_goal_and_start(self) -> None:
        """Intializes robot starting pos and goal pos"""

        x1 = np.random.randint(-self.grid_size_half, self.grid_size_half)
        y1 = np.random.randint(-self.grid_size_half, self.grid_size_half)

        self.start_pos = [x1, y1, 0.3]

        distance = 0
        while (distance < self.min_dist_to_goal):
            x2 = np.random.randint(-self.grid_size_half, self.grid_size_half)
            y2 = np.random.randint(-self.grid_size_half, self.grid_size_half)
            distance = math.hypot(x1 - x2, y1 - y2)
        
        self.goal_pos = [x2, y2, 0.3]

    def _spawn_goal_start(self):
        """Creates beams displaying start and goal positions"""
        
        gx, gy, gz = self.goal_pos
        sx, sy, sz = self.start_pos
        height = 2.0
        radius = 0.05

        visual_id_s = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[1, 0, 0, 0.4]
        )

        visual_id_g = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0, 1, 0, 0.4]
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id_g,
            baseCollisionShapeIndex=-1,
            basePosition=[gx, gy, gz + height/2]
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id_s,
            baseCollisionShapeIndex=-1,
            basePosition=[sx, sy, sz + height/2]
        )

    def _spawn_robot(self) -> None:
        """Loads the robot into the world facing random direction"""
        
        self.joint_indicies.clear()

        random_yaw = np.random.uniform(-np.pi,np.pi)
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(self.robot_file, self.start_pos, startOrientation)

        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)

            joint_index = info[0]
            joint_type = info[2]
            link_name = info[12].decode("utf-8")

            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indicies.append(joint_index)

            if "foot" in link_name.lower():
                self.foot_indicies.append(joint_index)

    def _build_static_world(self) -> None:
        """Creates world that exists between each episode"""
        
        self.planeId = p.loadURDF("plane.urdf")

    def _spawn_obstacles(self, num_of_obsticles):
        self.spawned_obsticled = set()

        dis = PoissonDisc(self.grid_size_full, self.grid_size_full, self.obstacle_distance, num_of_obsticles)
        points = dis.generate()
        gx, gy = self.goal_pos[0], self.goal_pos[1]
        sx, sy = self.start_pos[0], self.start_pos[1]

        for point in points:
            
            x, y = point

            x = x - self.grid_size_half
            y = y - self.grid_size_half

            if math.hypot(gx-x, gy-y) < 1.0 or math.hypot(sx-x, sy-y) < 1.0:
                continue

            visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.4,0.4,0.4])
            collisionShapeID = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.4,0.4,0.4])
            body_id = p.createMultiBody(baseCollisionShapeIndex=collisionShapeID,
                              baseVisualShapeIndex=visualShapeId,
                              basePosition=[x,y,0.5],
                              baseMass=0)
            
            if body_id != -1:
                self.obstacle_ids.add(body_id)
    
    def _get_lidar_points(self, num_of_rays=360, ray_range=5, height=.02):
        """Calculates 360 degree vector of distances to object
        
        Casts the num of rays evenly spaced in 360 degree range around
        the robot slighly above robots height position. Calculates the
        distance to the object hit by ray if any and is returned in
        hit vector
        """

        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        x, y, z = pos

        start = np.array([x,y,z+height])

        angles = np.linspace(0, 2*np.pi, num_of_rays, endpoint=False)

        ray_from = []
        ray_to = []

        for angle in angles:
            nx = np.sin(angle)
            ny = np.cos(angle)

            end = start + (ray_range * np.array([nx, ny, 0.0]))

            ray_from.append(start.tolist())
            ray_to.append(end)
        
        results = p.rayTestBatch(ray_from,ray_to)
        
        hit_vector = np.zeros(num_of_rays)
        for i, (object_id, link_id, hit_fraction, hit_position, hit_normal) in enumerate(results):
            
            if object_id == -1:
                hit_vector[i] = ray_range
            else:
                hit_vector[i] = ray_range * hit_fraction
        
        return hit_vector
    
    def _set_servo_positions(self, actions):
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indicies,
            p.POSITION_CONTROL,
            targetPositions = actions,
            forces=[4.0]*12
        )

    def _get_servo_positions(self):
        """Gets the current joint angles (radians) for the 12 servos"""
        
        states = p.getJointStates(self.robot_id, self.joint_indicies)
        positions = np.array([s[0] for s in states])
        return positions

    def _has_fallen(self) -> bool:
        """Returns boolean stating if robot has fallen or not"""
        
        contact_points = p.getContactPoints(self.robot_id)
        for point in contact_points:
            contact_b = point[2]
            link_a_id = point[3]
            if contact_b == self.planeId:
                if link_a_id == -1:
                    return True
                # if link_a_id not in self.foot_indicies:
                #     return True
        
        return False
    
    def _get_contact(self) -> bool:
        """Return true if part of the robot hit any obstacle"""

        contact_points = p.getContactPoints(self.robot_id)
        for point in contact_points:
            contact_b = point[2]
            if contact_b in self.obstacle_ids:
                return True
        
        return False

    def _reward(self):
        """Calculates the reward for current simulation step
        
        Calculates reward by calculating distance away from
        goal node, if the robot hit any object, if the robot
        has fallen.
        """
        
        reward = 0.0
        done = False
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        x, y, z = pos
        gx, gy, gz = self.goal_pos

        distance = math.hypot(gx-x, gy-y)
        reward += -distance

        if distance < 0.5 :
            reward += 50
            done = True

        if self._get_contact():
            reward -= 2

        if self._has_fallen():
            reward -= 50
            done = True
        
        return reward, done
    
    def obeservations(self):
        return np.concatenate([self._get_lidar_points(), self._get_servo_positions()])

    def _set_up_sim(self):
        self._set_goal_and_start()
        self._build_static_world()
        self._spawn_goal_start()
        self._spawn_robot()
        self._spawn_obstacles(self.num_of_obsticles)

    def reset(self):
        """Resets simulation world

        Args:
        None

        Returns:
        observations: np.array of shape (372,) with lidars data points
            and the servo angles
        """
        
        p.resetSimulation()

        self.joint_indicies.clear()
        self.obstacle_ids.clear()
        self.foot_indicies.clear()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1/240)
        self._set_up_sim()

        return self.obeservations()

    def step(self, actions):
        """Step forward the simulation, given the action.

        Args:
        actions: A list of desired motor angles for the 12 servos.

        Returns:
        observations: The 360 degree lidar data + the 12 servo angles
        reward: The reward for the current state-action pair.
        done: Whether the episode has ended.
        """
        
        self._set_servo_positions(actions)

        for _ in range(self.action_repeat):
            p.stepSimulation()

        reward, done = self._reward()
        obs = self.obeservations()

        return obs, reward, done

    def close(self):
        p.disconnect()