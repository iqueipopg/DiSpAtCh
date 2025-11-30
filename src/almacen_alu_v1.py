import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        just_pick: bool = True,
        random_objects: bool = False,
        render_mode: str = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.width = 10.0
        self.height = 10.0

        self.just_pick = just_pick
        self.random_objects = random_objects

        # Define action and observation spaces
        n_actions = 5 if self.just_pick else 6
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(11,), dtype=np.float32
        )

        # Warehouse layout
        self.shelves = [
            (1.9, 1.0, 0.2, 5.0),
            (4.9, 1.0, 0.2, 5.0),
            (7.9, 1.0, 0.2, 5.0),
        ]
        self.delivery_area = (2.5, 9, 5.0, 2.0)

        # Agent properties
        self.agent_radius = 0.2
        self.agent_velocity = 0.25
        self.pickup_distance = 0.6

        # === RECOMPENSAS DISEÑADAS ===
        self.r_step = -0.02
        self.r_collision = -200.0  # ✅ AUMENTADO OTRA VEZ (era -100.0)
        self.r_pick = 100.0
        self.r_delivery = 1000.0
        self.r_wrong_drop = -2.0

        self.fig = None
        self.ax = None

        # Para tracking
        self.previous_distance_to_nearest_object = None
        self.previous_distance_to_delivery = None

        self.reset()

    def reset(self, options=None):
        super().reset()

        self.agent_pos = self._get_random_empty_position()

        if self.random_objects:
            self.object_positions = [
                self._get_random_position_on_shelf(s) for s in self.shelves
            ]
        else:
            self.object_positions = [(2, 3.0), (8, 4.0), (5, 2.0)]

        self.agent_has_object = False
        self.delivery = False
        self.collision = False

        self.steps = 0
        self.max_steps = 400  # ✅ REDUCIDO (era 1000)

        # Reset tracking
        self.previous_distance_to_nearest_object = (
            self._get_distance_to_nearest_object()
        )
        delivery_center = (
            self.delivery_area[0] + self.delivery_area[2] / 2,
            self.delivery_area[1] + self.delivery_area[3] / 2,
        )
        self.previous_distance_to_delivery = self._distance(
            self.agent_pos, delivery_center
        )

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        self.steps += 1
        reward = self.r_step
        terminated = False
        truncated = False

        # --- Action handling ---
        if action < 4:  # Movement (0: Up, 1: Down, 2: Left, 3: Right)
            new_pos = self._get_new_position(action)

            if not self._is_collision(new_pos):
                self.agent_pos = new_pos

                # ✅ SHAPED REWARD CORREGIDO
                if not self.agent_has_object:
                    # Sin objeto: reward ligero por acercarse
                    current_distance = self._get_distance_to_nearest_object()
                    if (
                        current_distance != float("inf")
                        and current_distance < self.previous_distance_to_nearest_object
                    ):
                        reward += 0.05
                    if current_distance != float("inf"):
                        self.previous_distance_to_nearest_object = current_distance
                else:
                    # ✅ CON OBJETO: Shaped reward FUERTE para guiar al área
                    delivery_center = (
                        self.delivery_area[0] + self.delivery_area[2] / 2,
                        self.delivery_area[1] + self.delivery_area[3] / 2,
                    )
                    current_distance = self._distance(self.agent_pos, delivery_center)

                    # Reward escalado por proximidad
                    if current_distance < self.previous_distance_to_delivery:
                        if current_distance < 2.0:
                            reward += 5.0  # MUY cerca
                        elif current_distance < 4.0:
                            reward += 2.0  # Cerca
                        elif current_distance < 6.0:
                            reward += 1.0  # Medio
                        else:
                            reward += 0.3  # Lejos
                    else:
                        reward -= 0.5  # Penalización por alejarse

                    # Gran bonus si está DENTRO del área
                    if self._is_in_area(self.agent_pos, self.delivery_area):
                        reward += 50.0  # ENORME bonus

                    self.previous_distance_to_delivery = current_distance
            else:
                # COLISIÓN
                self.collision = True
                terminated = True
                reward = self.r_collision

        elif action == 4:  # Pick
            picked = False
            if not self.agent_has_object:
                for i, obj_pos in enumerate(self.object_positions):
                    if (
                        obj_pos is not None
                        and self._distance(self.agent_pos, obj_pos)
                        <= self.pickup_distance + self.agent_radius
                    ):
                        self.agent_has_object = True
                        self.object_positions[i] = None
                        reward = self.r_pick
                        picked = True

                        # Resetear tracking para la nueva fase
                        delivery_center = (
                            self.delivery_area[0] + self.delivery_area[2] / 2,
                            self.delivery_area[1] + self.delivery_area[3] / 2,
                        )
                        self.previous_distance_to_delivery = self._distance(
                            self.agent_pos, delivery_center
                        )

                        if self.just_pick:
                            terminated = True  # Éxito en Entorno 1
                        break

                if not picked:
                    reward = -2.0

        elif action == 5:  # Drop (solo en Entorno 2 y 3)
            if self.agent_has_object:
                if self._is_in_area(self.agent_pos, self.delivery_area):
                    # ✅ ENTREGA EXITOSA
                    reward = self.r_delivery
                    self.delivery = True
                    terminated = True
                else:
                    # ✅ ENTREGA FALLIDA: Reward parcial si está cerca
                    delivery_center = (
                        self.delivery_area[0] + self.delivery_area[2] / 2,
                        self.delivery_area[1] + self.delivery_area[3] / 2,
                    )
                    distance = self._distance(self.agent_pos, delivery_center)

                    if distance < 1.5:
                        reward = 100.0  # Muy cerca
                    elif distance < 3.0:
                        reward = 30.0  # Cerca
                    else:
                        reward = self.r_wrong_drop

                    self.object_positions.append(self.agent_pos)
                    terminated = True

                self.agent_has_object = False
            else:
                reward = -2.0

        # Truncamiento por timeout
        if self.steps >= self.max_steps:
            truncated = True
            reward += -10.0

        # --- Prepare outputs ---
        obs = self._get_obs()
        info = {
            "has_object": self.agent_has_object,
            "collision": self.collision,
            "delivery": self.delivery,
            "steps": self.steps,
        }

        return obs, reward, terminated, truncated, info

    def _get_distance_to_nearest_object(self):
        """Distancia al objeto más cercano disponible"""
        distances = []
        for obj_pos in self.object_positions:
            if obj_pos is not None:
                distances.append(self._distance(self.agent_pos, obj_pos))
        return min(distances) if distances else float("inf")

    def _get_obs(self):
        obs = np.zeros(11, dtype=np.float32)
        obs[0:2] = self.agent_pos

        # Objetos (si no existen, poner fuera del mapa)
        for i in range(3):
            if i < len(self.object_positions) and self.object_positions[i] is not None:
                obs[2 + 2 * i : 4 + 2 * i] = self.object_positions[i]
            else:
                obs[2 + 2 * i : 4 + 2 * i] = [10, 10]

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
        # Check walls
        if (
            pos[0] <= self.agent_radius
            or pos[0] >= self.width - self.agent_radius
            or pos[1] <= self.agent_radius
            or pos[1] >= self.height - self.agent_radius
        ):
            return True

        # Check shelves
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
            area[0] - margin <= pos[0] <= area[0] + area[2] + margin
            and area[1] - margin <= pos[1] <= area[1] + area[3] + margin
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
            self.ax.add_patch(
                Rectangle(
                    s[:2],
                    s[2],
                    s[3],
                    fill=True,
                    facecolor="saddlebrown",
                    edgecolor="brown",
                    linewidth=2,
                )
            )

        # Delivery area
        self.ax.add_patch(
            Rectangle(
                self.delivery_area[:2],
                self.delivery_area[2],
                self.delivery_area[3],
                fill=True,
                facecolor="lightgreen",
                edgecolor="green",
                alpha=0.6,
                linewidth=2,
            )
        )

        # Objects
        for obj in self.object_positions:
            if obj is not None:
                self.ax.add_patch(Circle(obj, radius=0.2, color="blue"))

        # Agent
        color = "red" if self.agent_has_object else "orange"
        self.ax.add_patch(
            Circle(self.agent_pos, radius=self.agent_radius, color=color, zorder=10)
        )

        title = f"Steps: {self.steps}"
        if self.agent_has_object:
            title += " | Carrying object 📦"
        plt.title(title)
        plt.draw()
        plt.pause(0.01)

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
    print("=== Testing Entorno 1 (just_pick=True) ===")
    env = WarehouseEnv(just_pick=True, random_objects=False, render_mode="human")
    obs, info = env.reset()

    for episode in range(2):
        obs, info = env.reset()
        total_reward = 0
        print(f"\n--- Episode {episode+1} ---")

        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()

            if terminated or truncated:
                print(f"Finished: Total reward: {total_reward:.2f}, Steps: {step+1}")
                if info["has_object"]:
                    print("✅ Success: Object picked!")
                elif info["collision"]:
                    print("❌ Failed: Collision")
                break

    env.close()
