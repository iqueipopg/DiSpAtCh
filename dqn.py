"""
DQN Agent para WarehouseEnv - Entornos 1, 2 y 3
================================================

MEJORAS IMPLEMENTADAS:
1. Red más pequeña y adecuada para 32 features densas
2. Epsilon decay lineal (más predecible)
3. Target network update más frecuente
4. Normalización de rewards para estabilidad
5. Exploración semi-guiada opcional
6. Mejor manejo del éxito según el entorno
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from almacen_alu_v1 import WarehouseEnv
from representacion_wharehouse import WarehouseFeedback


# =============================================================================
# RED NEURONAL
# =============================================================================


class DQN(nn.Module):
    """
    Red neuronal para DQN.
    Arquitectura ajustada para 32 features densas.
    """

    def __init__(self, input_size, num_actions, hidden_sizes=[128, 128, 64]):
        super(DQN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_actions))

        self.network = nn.Sequential(*layers)

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# REPLAY BUFFER
# =============================================================================


class ReplayBuffer:
    """Buffer de experiencia simple pero eficiente."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# AGENTE DQN
# =============================================================================


class DQNAgent:
    def __init__(
        self,
        env,
        feedback,
        input_size,
        num_actions,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20000,
        buffer_size=100000,
        batch_size=64,
        target_update=500,
        device="cuda" if torch.cuda.is_available() else "cpu",
        env_type="Entorno_1",
    ):
        self.env = env
        self.feedback = feedback
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.env_type = env_type

        # Epsilon scheduling lineal
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Redes Q
        self.q_network = DQN(input_size, num_actions).to(self.device)
        self.target_network = DQN(input_size, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience Replay
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Contadores
        self.total_steps = 0
        self.episodes_trained = 0

        # Mejor modelo
        self.best_eval_success = 0.0
        self.best_eval_reward = -float("inf")

        # Métricas
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.losses = []

        # Stats para debugging
        self.collision_count = 0
        self.pick_success_count = 0
        self.pick_attempt_count = 0
        self.delivery_count = 0

    def get_epsilon(self):
        """Epsilon decay lineal."""
        progress = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def get_action(self, obs, epsilon=None):
        """
        Selecciona acción usando epsilon-greedy PURO.
        """
        if epsilon is None:
            epsilon = self.get_epsilon()

        if np.random.random() < epsilon:
            # Exploración: acción completamente aleatoria
            return self.env.action_space.sample()

        # Explotación: usar la red Q
        features = self.feedback.process_observation(obs)
        state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.argmax().item()

    def train_step(self):
        """Realiza un paso de entrenamiento."""
        if len(self.replay_buffer) < self.batch_size * 2:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # NO normalizar rewards - queremos que la red vea la diferencia
        # entre +100 (pick exitoso) y -15 (colisión)

        # Q actual
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Q target (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * ~dones.unsqueeze(1)

        # Loss (Huber loss para robustez)
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Actualizar target network
        if self.total_steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def _check_success(self, info, terminated, reward=None):
        """
        Verifica si el episodio fue exitoso según el tipo de entorno.

        Para Entorno 1: Éxito si el episodio terminó Y el agente tiene objeto
                        (pick exitoso termina el episodio con has_object=True)
        Para Entornos 2,3: Éxito si hubo delivery
        """
        if self.env_type == "Entorno_1":
            # Entorno 1: Éxito si terminó con objeto (pick exitoso)
            # También verificamos reward alto como backup
            if terminated and info.get("has_object", False):
                return True
            # Backup: si reward es muy alto (~100), fue pick exitoso
            if reward is not None and reward > 50:
                return True
            return False
        else:
            # Entornos 2 y 3: Éxito si entregó objeto
            return info.get("delivery", False)

    def train(self, num_episodes, evaluate_every=200, num_eval_episodes=20):
        """
        Entrena el agente por num_episodes episodios.
        """
        pbar = tqdm(total=num_episodes, desc=f"Training DQN ({self.env_type})")
        recent_rewards = deque(maxlen=100)
        recent_success = deque(maxlen=100)

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            episode_losses = []

            for step in range(1000):  # Max steps por episodio
                # Seleccionar acción
                action = self.get_action(obs)

                # Ejecutar acción
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Estadísticas
                if info.get("collision", False):
                    self.collision_count += 1
                if action == 4:  # Pick
                    self.pick_attempt_count += 1
                    if info.get("has_object", False):
                        self.pick_success_count += 1
                if info.get("delivery", False):
                    self.delivery_count += 1

                # Procesar features
                features = self.feedback.process_observation(obs)
                next_features = self.feedback.process_observation(next_obs)

                # Guardar en buffer
                self.replay_buffer.push(features, action, reward, next_features, done)

                self.total_steps += 1

                # Entrenar
                loss = self.train_step()
                if loss is not None:
                    episode_losses.append(loss)

                obs = next_obs
                total_reward += reward

                if done:
                    break

            # Métricas del episodio - detectar éxito correctamente
            if self.env_type == "Entorno_1":
                success = terminated and info.get("has_object", False)
            else:
                success = info.get("delivery", False)

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step + 1)
            recent_rewards.append(total_reward)
            recent_success.append(1.0 if success else 0.0)

            if episode_losses:
                self.losses.append(np.mean(episode_losses))

            self.episodes_trained += 1

            # Actualizar progress bar
            pbar.set_postfix(
                {
                    "Reward": f"{np.mean(recent_rewards):.1f}",
                    "Success%": f"{np.mean(recent_success)*100:.0f}",
                    "ε": f"{self.get_epsilon():.3f}",
                    "Steps": step + 1,
                }
            )
            pbar.update(1)

            # Evaluación periódica
            if (episode + 1) % evaluate_every == 0:
                eval_reward, eval_success = self.evaluate(num_eval_episodes)
                self.success_rate.append((episode + 1, eval_success))

                # ===== GUARDAR MEJOR MODELO =====
                if eval_success > self.best_eval_success or (
                    eval_success == self.best_eval_success
                    and eval_reward > self.best_eval_reward
                ):
                    self.best_eval_success = eval_success
                    self.best_eval_reward = eval_reward
                    self.save(f"best_model_{self.env_type}.pth")
                    print(
                        f"\n   💾 Nuevo mejor modelo guardado! Success={eval_success*100:.0f}%"
                    )

                # Stats de TRAINING (últimos N episodios)
                train_pick_rate = (
                    self.pick_success_count / max(1, self.pick_attempt_count) * 100
                )

                # Stats de EVALUACIÓN
                eval_stats = getattr(self, "last_eval_stats", {})
                eval_pick_rate = (
                    eval_stats.get("picks", 0)
                    / max(1, eval_stats.get("pick_attempts", 1))
                    * 100
                )
                eval_drop_rate = (
                    eval_stats.get("drops", 0)
                    / max(1, eval_stats.get("drop_attempts", 1))
                    * 100
                )
                action_dist = eval_stats.get("action_distribution", [0] * 6)

                print(f"\n📊 Ep {episode + 1}:")
                print(
                    f"   Eval: Reward={eval_reward:.2f}, Success={eval_success*100:.0f}%"
                )
                print(
                    f"   Eval Stats: Collisions={eval_stats.get('collisions', 0)}, PickRate={eval_pick_rate:.1f}%",
                    end="",
                )
                if self.env_type != "Entorno_1":
                    print(
                        f", DropRate={eval_drop_rate:.1f}% ({eval_stats.get('drops', 0)}/{eval_stats.get('drop_attempts', 0)})"
                    )
                else:
                    print()
                print(
                    f"   Eval Actions: ↑={action_dist[0]}, ↓={action_dist[1]}, ←={action_dist[2]}, →={action_dist[3]}, pick={action_dist[4]}",
                    end="",
                )
                if self.env_type != "Entorno_1":
                    print(f", drop={action_dist[5]}")
                else:
                    print()
                print(
                    f"   Train Stats: Collisions={self.collision_count}, PickRate={train_pick_rate:.1f}%",
                    end="",
                )
                if self.env_type != "Entorno_1":
                    print(f", Deliveries={self.delivery_count}")
                else:
                    print()

                # Reset stats de training
                self.collision_count = 0
                self.pick_success_count = 0
                self.pick_attempt_count = 0
                self.delivery_count = 0

        pbar.close()
        self.plot_training()

    def evaluate(self, num_episodes):
        """Evalúa el agente con epsilon mínimo (0.05)."""
        rewards = []
        successes = []
        eval_collisions = 0
        eval_picks = 0
        eval_pick_attempts = 0
        eval_drops = 0
        eval_drop_attempts = 0
        action_counts = [0, 0, 0, 0, 0, 0]  # Contar qué acciones elige

        eval_epsilon = 0.05  # Pequeña exploración para evitar loops

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            success = False
            had_object_before = False

            for step in range(500):
                # Usar epsilon pequeño
                action = self.get_action(obs, epsilon=eval_epsilon)
                action_counts[action] += 1

                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward

                # Stats de evaluación
                if info.get("collision", False):
                    eval_collisions += 1
                if action == 4:  # Pick
                    eval_pick_attempts += 1
                    if info.get("has_object", False) and not had_object_before:
                        eval_picks += 1
                if action == 5:  # Drop
                    eval_drop_attempts += 1
                    if info.get("delivery", False):
                        eval_drops += 1

                had_object_before = info.get("has_object", False)

                # Detectar éxito
                if self.env_type == "Entorno_1":
                    if info.get("has_object", False) and terminated:
                        success = True
                else:
                    if info.get("delivery", False):
                        success = True

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            successes.append(1.0 if success else 0.0)

        # Guardar stats
        self.last_eval_stats = {
            "collisions": eval_collisions,
            "picks": eval_picks,
            "pick_attempts": eval_pick_attempts,
            "drops": eval_drops,
            "drop_attempts": eval_drop_attempts,
            "action_distribution": action_counts,
        }

        return np.mean(rewards), np.mean(successes)

    def plot_training(self):
        """Visualiza las métricas de entrenamiento."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, color="blue")
        if len(self.episode_rewards) > 50:
            smooth = np.convolve(self.episode_rewards, np.ones(50) / 50, mode="valid")
            axes[0, 0].plot(
                range(49, len(self.episode_rewards)), smooth, "b-", linewidth=2
            )
        axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        axes[0, 0].set_title(f"Episode Rewards ({self.env_type})", fontweight="bold")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].grid(True, alpha=0.3)

        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.3, color="green")
        if len(self.episode_lengths) > 50:
            smooth = np.convolve(self.episode_lengths, np.ones(50) / 50, mode="valid")
            axes[0, 1].plot(
                range(49, len(self.episode_lengths)), smooth, "g-", linewidth=2
            )
        axes[0, 1].set_title("Episode Lengths", fontweight="bold")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(True, alpha=0.3)

        # Loss
        if self.losses:
            axes[1, 0].plot(self.losses, alpha=0.5, color="red")
            if len(self.losses) > 50:
                smooth = np.convolve(self.losses, np.ones(50) / 50, mode="valid")
                axes[1, 0].plot(
                    range(49, len(self.losses)), smooth, "darkred", linewidth=2
                )
            axes[1, 0].set_title("Training Loss", fontweight="bold")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_yscale("log")
            axes[1, 0].grid(True, alpha=0.3)

        # Success rate
        if self.success_rate:
            eps, rates = zip(*self.success_rate)
            axes[1, 1].plot(
                eps,
                [r * 100 for r in rates],
                "o-",
                color="purple",
                linewidth=2,
                markersize=6,
            )
            axes[1, 1].set_title("Evaluation Success Rate", fontweight="bold")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Success Rate (%)")
            axes[1, 1].set_ylim(0, 105)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"training_{self.env_type}_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"📈 Gráficas guardadas: {filename}")

    def save(self, filename):
        """Guarda el modelo."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "episodes_trained": self.episodes_trained,
                "episode_rewards": self.episode_rewards,
                "success_rate": self.success_rate,
                "env_type": self.env_type,
            },
            filename,
        )
        print(f"✅ Modelo guardado: {filename}")

    def load(self, filename):
        """Carga el modelo."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.episodes_trained = checkpoint.get("episodes_trained", 0)
        print(f"✅ Modelo cargado: {filename}")


# =============================================================================
# FUNCIÓN DE ENTRENAMIENTO
# =============================================================================


def train_environment(env_name, just_pick, random_objects, num_episodes=2000):
    """
    Entrena un agente para un entorno específico.

    Args:
        env_name: "Entorno_1", "Entorno_2", o "Entorno_3"
        just_pick: True para Entorno 1, False para 2 y 3
        random_objects: False para 1 y 2, True para 3
        num_episodes: Número de episodios de entrenamiento
    """
    print(f"\n{'='*60}")
    print(f"🎯 Training {env_name}")
    print(f"   just_pick={just_pick}, random_objects={random_objects}")
    print(f"{'='*60}")

    # Crear entorno
    env = WarehouseEnv(
        just_pick=just_pick, random_objects=random_objects, render_mode=None
    )

    # Crear feedback (representación de estado)
    feedback = WarehouseFeedback(dims=(10.0, 10.0), delivery_area=(2.5, 9, 5.0, 2.0))

    # Configuración
    input_size = feedback.feature_size
    num_actions = 5 if just_pick else 6

    print(f"📐 Input size: {input_size}, Actions: {num_actions}")
    print(f"🔧 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Hiperparámetros según entorno
    if env_name == "Entorno_1":
        config = {
            "lr": 1e-4,  # Más bajo porque rewards son grandes
            "epsilon_decay_steps": 40000,  # Más exploración
            "target_update": 1000,  # Más estable
            "num_episodes": 4000,
        }
    elif env_name == "Entorno_2":
        config = {
            "lr": 5e-5,
            "epsilon_decay_steps": 60000,
            "target_update": 1000,
            "num_episodes": 6000,
        }
    else:  # Entorno_3
        config = {
            "lr": 3e-5,
            "epsilon_decay_steps": 100000,
            "target_update": 1000,
            "num_episodes": 10000,
        }

    print(
        f"⚙️  Config: lr={config['lr']}, decay_steps={config['epsilon_decay_steps']}, target_update={config['target_update']}"
    )

    # Crear agente
    agent = DQNAgent(
        env=env,
        feedback=feedback,
        input_size=input_size,
        num_actions=num_actions,
        lr=config["lr"],
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=config["epsilon_decay_steps"],
        buffer_size=100000,
        batch_size=64,
        target_update=config["target_update"],
        env_type=env_name,
    )

    # Entrenar
    agent.train(
        num_episodes=config.get("num_episodes", num_episodes),
        evaluate_every=200,
        num_eval_episodes=20,
    )

    # Evaluación final
    print(f"\n{'='*60}")
    print(f"🎯 EVALUACIÓN FINAL - {env_name}")
    print(f"{'='*60}")

    final_reward, final_success = agent.evaluate(num_episodes=100)
    print(f"   Reward promedio: {final_reward:.2f}")
    print(f"   Tasa de éxito: {final_success*100:.1f}%")

    # Guardar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"dqn_{env_name.lower()}_{timestamp}.pth"
    agent.save(filename)

    env.close()
    return agent


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("🚀 DQN Warehouse Training Script")
    print("=" * 60)
    print("\nSelect environment to train:")
    print("1. Entorno 1: Fixed objects, pick only")
    print("2. Entorno 2: Fixed objects, pick + delivery")
    print("3. Entorno 3: Random objects, pick + delivery")
    print("4. Train all (sequentially)")

    choice = input("\nEnter choice (1-4) [default=1]: ").strip() or "1"

    if choice == "1":
        train_environment(
            "Entorno_1", just_pick=True, random_objects=False, num_episodes=2000
        )
    elif choice == "2":
        train_environment(
            "Entorno_2", just_pick=False, random_objects=False, num_episodes=3000
        )
    elif choice == "3":
        train_environment(
            "Entorno_3", just_pick=False, random_objects=True, num_episodes=5000
        )
    elif choice == "4":
        train_environment(
            "Entorno_1", just_pick=True, random_objects=False, num_episodes=2000
        )
        train_environment(
            "Entorno_2", just_pick=False, random_objects=False, num_episodes=3000
        )
        train_environment(
            "Entorno_3", just_pick=False, random_objects=True, num_episodes=5000
        )
    else:
        print("Opción inválida. Entrenando Entorno 1 por defecto...")
        train_environment(
            "Entorno_1", just_pick=True, random_objects=False, num_episodes=2000
        )

    print("\n✅ ¡Entrenamiento completado!")
