"""
Entorno de Almacén para RL - Gymnasium Environment
Basado en almacen_alu_v1.py
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, just_pick: bool = True, random_objects: bool = False, render_mode: str = None):
        super().__init__()

        self.render_mode = render_mode
        self.width = 10.0
        self.height = 10.0

        self.just_pick = just_pick
        self.random_objects = random_objects

        # Define action and observation spaces
        n_actions = 5 if self.just_pick else 6
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=10, shape=(11,), dtype=np.float32)

        # Warehouse layout
        self.shelves = [(1.9, 1.0, 0.2, 5.0), (4.9, 1.0, 0.2, 5.0), (7.9, 1.0, 0.2, 5.0)]
        self.delivery_area = (2.5, 9, 5.0, 2.0)

        # Agent properties
        self.agent_radius = 0.2
        self.agent_velocity = 0.25
        self.pickup_distance = 0.6

        self.fig = None
        self.ax = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = self._get_random_empty_position()

        if self.random_objects:
            self.object_positions = [self._get_random_position_on_shelf(s) for s in self.shelves]
        else:
            self.object_positions = [(2, 3.0), (8, 4.0), (5, 2.0)]

        self.agent_has_object = False
        self.delivery = False
        self.collision = False

        self.steps = 0
        self.max_steps = 200

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        # --- Initialize state variables ---
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False

        # --- Action handling ---
        if action < 4:  # Movement
            new_pos = self._get_new_position(action)
            if not self._is_collision(new_pos):
                self.agent_pos = new_pos
                reward = -1  # Pequeña penalización por paso
            else:
                self.collision = True
                terminated = True
                reward = -100  # Penalización por colisión

        elif action == 4:  # Pick
            if not self.agent_has_object:
                picked = False
                for i, obj_pos in enumerate(self.object_positions):
                    if obj_pos is not None and self._distance(self.agent_pos, obj_pos) <= self.pickup_distance + self.agent_radius:
                        self.agent_has_object = True
                        self.object_positions[i] = None
                        reward = 100  # Recompensa por recoger objeto
                        picked = True
                        if self.just_pick:
                            terminated = True
                        break
                if not picked:
                    reward = -1  # Penalización si intenta coger sin estar cerca

        elif action == 5:  # Drop
            if self.agent_has_object:
                if self._is_in_area(self.agent_pos, self.delivery_area):
                    reward = 200  # Gran recompensa por entrega exitosa
                    self.delivery = True
                else:
                    reward = -50  # Penalización por soltar fuera de zona
                    self.object_positions.append(self.agent_pos)
                self.agent_has_object = False
                terminated = True
            else:
                reward = -1  # Penalización si intenta soltar sin objeto

        if self.steps >= self.max_steps:
            truncated = True

        # --- Prepare outputs ---
        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.zeros(11, dtype=np.float32)
        obs[0:2] = self.agent_pos
        for i, obj in enumerate(self.object_positions):
            if obj is not None:
                obs[2 + 2 * i: 4 + 2 * i] = obj
            else:
                obs[2 + 2 * i: 4 + 2 * i] = self.agent_pos
        obs[8] = float(self.agent_has_object)
        obs[9] = float(self.collision)
        obs[10] = float(self.delivery)
        return obs

    def _get_new_position(self, action):
        x, y = self.agent_pos
        if action == 0:  # Up
            y = min(self.height - self.agent_radius, y + self.agent_velocity)
        elif action == 1:  # Down
            y = max(self.agent_radius, y - self.agent_velocity)
        elif action == 2:  # Left
            x = max(self.agent_radius, x - self.agent_velocity)
        elif action == 3:  # Right
            x = min(self.width - self.agent_radius, x + self.agent_velocity)
        return (x, y)

    def _is_collision(self, pos):
        # Check for collisions with walls
        if (
            pos[0] <= self.agent_radius or
            pos[0] >= self.width - self.agent_radius or
            pos[1] <= self.agent_radius or
            pos[1] >= self.height - self.agent_radius
        ):
            return True

        # Check for collisions with shelves
        for shelf in self.shelves:
            if self._is_in_area(pos, shelf, self.agent_radius):
                return True

        return False

    def _get_random_empty_position(self):
        while True:
            pos = (
                np.random.uniform(self.agent_radius, self.width - self.agent_radius),
                np.random.uniform(self.agent_radius, self.height - self.agent_radius),
            )
            if not self._is_collision(pos):
                return pos

    def _get_random_position_on_shelf(self, shelf):
        aux = np.random.uniform(0, 1)
        x = shelf[0] + (0.25 if aux < 0.5 else 0.75) * shelf[2]
        y = np.random.uniform(shelf[1] + 0.5, shelf[1] + shelf[3] - 0.5)
        return (x, y)

    @staticmethod
    def _distance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @staticmethod
    def _is_in_area(pos, area, margin=0):
        return (
            area[0] - margin <= pos[0] <= area[0] + area[2] + margin and
            area[1] - margin <= pos[1] <= area[1] + area[3] + margin
        )

    def render(self):
        if self.render_mode is None:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect("equal")

        # Shelves
        for s in self.shelves:
            self.ax.add_patch(Rectangle(s[:2], s[2], s[3], fill=False, edgecolor="brown"))

        # Delivery area
        self.ax.add_patch(Rectangle(self.delivery_area[:2], self.delivery_area[2], self.delivery_area[3],
                                    fill=True, facecolor="lightgreen", edgecolor="green", alpha=0.5))

        # Objects
        for obj in self.object_positions:
            if obj is not None:
                self.ax.add_patch(Circle(obj, radius=0.2, color="blue"))

        # Agent
        color = "red" if self.agent_has_object else "orange"
        self.ax.add_patch(Circle(self.agent_pos, radius=self.agent_radius, color=color))

        plt.title("WarehouseEnv")
        plt.draw()
        plt.pause(0.05)

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None


if __name__ == "__main__":
    env = WarehouseEnv(just_pick=True, random_objects=False, render_mode="human")
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
