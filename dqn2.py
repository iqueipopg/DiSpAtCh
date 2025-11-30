"""
DQN con Transfer Learning - VERSIÓN FINAL CORREGIDA
====================================================

Cambios aplicados:
- Learning rate aumentado a 1e-4
- Epsilon inicial reducido a 0.5
- Forzado de Drop solo cuando está DENTRO del área
- Colisión penalizada -100 (en entorno)
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
from pathlib import Path
import glob

from almacen_alu_v1 import WarehouseEnv
from representacion_wharehouse import WarehouseFeedback
from tqdm import tqdm


class DQN(nn.Module):
    """Red neuronal para DQN"""

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

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Buffer de experiencia"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        env,
        feedback,
        num_actions=6,
        lr=1e-4,  # ✅ AUMENTADO (era 3e-5)
        gamma=0.99,
        epsilon_start=0.5,  # ✅ REDUCIDO (era 0.7)
        epsilon_end=0.05,
        epsilon_decay_steps=150000,
        buffer_size=50000,
        batch_size=64,
        target_update=1000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.feedback = feedback
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        input_size = feedback.feature_size

        # Redes Q
        self.q_network = DQN(input_size, num_actions).to(self.device)
        self.target_network = DQN(input_size, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Epsilon-greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Contadores
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.losses = []

        # ✅ AÑADIDO: Tracking del mejor modelo
        self.best_eval_success = 0.0
        self.best_eval_reward = float("-inf")

        # Tracking
        self.collision_count = 0
        self.pick_success_count = 0
        self.pick_attempt_count = 0
        self.drop_success_count = 0
        self.drop_attempt_count = 0

    def get_epsilon(self):
        """Epsilon decay lineal"""
        fraction = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def get_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.get_epsilon()

        # ✅ FORZAR Drop solo si está DENTRO del área de delivery
        has_object = state[8] > 0.5

        if has_object:
            agent_x, agent_y = state[0], state[1]

            # Verificar si está DENTRO del área de delivery
            delivery_area = (2.5, 9, 5.0, 2.0)  # (x, y, width, height)
            in_delivery_area = (
                delivery_area[0] <= agent_x <= delivery_area[0] + delivery_area[2]
                and delivery_area[1] <= agent_y <= delivery_area[1] + delivery_area[3]
            )

            if in_delivery_area:
                # Está dentro → 80% probabilidad de forzar Drop
                if np.random.random() < 0.8:
                    return 5
            else:
                # Está fuera pero cerca del centro
                delivery_center_x = 5.0
                delivery_center_y = 10.0
                distance = np.sqrt(
                    (agent_x - delivery_center_x) ** 2
                    + (agent_y - delivery_center_y) ** 2
                )

                # Solo si está MUY cerca (<1.5m del centro)
                if distance < 1.5 and np.random.random() < 0.3:
                    return 5

        # Epsilon-greedy normal
        if np.random.random() < epsilon:
            return self.env.action_space.sample()

        state_features = self.feedback.process_observation(state)
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def train_step(self, state, action, reward, next_state, done):
        state_features = self.feedback.process_observation(state)
        next_state_features = self.feedback.process_observation(next_state)
        self.replay_buffer.push(
            state_features, action, reward, next_state_features, done
        )

        self.total_steps += 1

        if len(self.replay_buffer) < self.batch_size * 4:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Q actual
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Q target (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (
                self.gamma * next_q_values * ~dones.unsqueeze(1)
            )

        # Loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Actualizar target network
        if self.total_steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def train(self, num_episodes, evaluate_every=250, num_eval_episodes=20):
        pbar = tqdm(total=num_episodes, desc="🔄 Transfer Learning")
        recent_rewards = deque(maxlen=100)
        recent_success = deque(maxlen=100)

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            episode_length = 0
            episode_losses = []
            success = False

            while True:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Tracking
                if info.get("collision", False):
                    self.collision_count += 1
                if action == 4:
                    self.pick_attempt_count += 1
                    if info.get("has_object", False):
                        self.pick_success_count += 1
                if action == 5:
                    self.drop_attempt_count += 1
                    if info.get("delivery", False):
                        self.drop_success_count += 1

                loss = self.train_step(state, action, reward, next_state, terminated)
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward
                episode_length += 1

                if terminated and info.get("delivery", False):
                    success = True

                if terminated or truncated:
                    break

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            recent_rewards.append(total_reward)
            recent_success.append(1.0 if success else 0.0)

            if episode_losses:
                self.losses.append(np.mean(episode_losses))

            # Actualizar barra
            pbar.set_postfix(
                {
                    "Reward": f"{np.mean(recent_rewards):.1f}",
                    "Success%": f"{np.mean(recent_success)*100:.0f}",
                    "ε": f"{self.get_epsilon():.3f}",
                    "Steps": f"{episode_length}",
                }
            )
            pbar.update(1)

            # Evaluación
            if episode % evaluate_every == 0 and episode > 0:
                eval_reward, eval_success = self.evaluate(num_eval_episodes)
                pick_rate = (
                    self.pick_success_count / max(1, self.pick_attempt_count)
                ) * 100
                drop_rate = (
                    self.drop_success_count / max(1, self.drop_attempt_count)
                ) * 100

                print(f"\n📊 Ep {episode}:")
                print(
                    f"   Eval: Reward={eval_reward:.2f}, Success={eval_success*100:.0f}%"
                )
                print(
                    f"   Stats: Pick={pick_rate:.1f}%, Drop={drop_rate:.1f}%, Collisions={self.collision_count}"
                )
                print(
                    f"   📊 Debug: PickAttempts={self.pick_attempt_count}, "
                    f"DropAttempts={self.drop_attempt_count}, DropSuccess={self.drop_success_count}"
                )

                self.success_rate.append((episode, eval_success))

                # ✅ AÑADIDO: Guardar si es el mejor modelo
                if eval_success > self.best_eval_success or (
                    eval_success == self.best_eval_success
                    and eval_reward > self.best_eval_reward
                ):
                    self.best_eval_success = eval_success
                    self.best_eval_reward = eval_reward

                    save_dir = Path("models")
                    save_dir.mkdir(exist_ok=True)
                    best_path = save_dir / "best_model_Entorno_2.pth"

                    self.save(str(best_path))
                    print(f"   💾 NEW BEST! Saved to {best_path}")
                    print(
                        f"   🏆 Best Success: {eval_success*100:.1f}% | Reward: {eval_reward:.2f}"
                    )

                # ✅ AÑADIDO: Guardar checkpoint cada 500 episodios
                if episode % 500 == 0:
                    save_dir = Path("models")
                    save_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    checkpoint_path = (
                        save_dir / f"checkpoint_ep{episode}_{timestamp}.pth"
                    )
                    self.save(str(checkpoint_path))
                    print(f"   📦 Checkpoint saved: {checkpoint_path}")

                # Reset stats
                self.collision_count = 0
                self.pick_success_count = 0
                self.pick_attempt_count = 0
                self.drop_success_count = 0
                self.drop_attempt_count = 0

        pbar.close()
        self.plot_training()

    def evaluate(self, num_episodes):
        total_rewards = []
        successes = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            success = False

            while not done:
                action = self.get_action(state, epsilon=0.0)
                state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated

                if terminated and info.get("delivery", False):
                    success = True

            total_rewards.append(total_reward)
            successes.append(1.0 if success else 0.0)

        return np.mean(total_rewards), np.mean(successes)

    def plot_training(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Recompensas
        axes[0, 0].plot(self.episode_rewards, alpha=0.2, color="blue")
        if len(self.episode_rewards) > 100:
            smooth = np.convolve(self.episode_rewards, np.ones(100) / 100, mode="valid")
            axes[0, 0].plot(smooth, color="blue", linewidth=2, label="Smoothed")
        axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        axes[0, 0].set_title(
            "Episode Rewards (Transfer Learning)", fontsize=12, fontweight="bold"
        )
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Longitud episodios
        axes[0, 1].plot(self.episode_lengths, alpha=0.2, color="green")
        if len(self.episode_lengths) > 100:
            smooth = np.convolve(self.episode_lengths, np.ones(100) / 100, mode="valid")
            axes[0, 1].plot(smooth, color="green", linewidth=2)
        axes[0, 1].set_title("Episode Lengths", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(True, alpha=0.3)

        # Loss
        if self.losses:
            axes[1, 0].plot(self.losses, alpha=0.5, color="red")
            if len(self.losses) > 50:
                smooth = np.convolve(self.losses, np.ones(50) / 50, mode="valid")
                axes[1, 0].plot(smooth, color="darkred", linewidth=2)
            axes[1, 0].set_title("Training Loss", fontsize=12, fontweight="bold")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_yscale("log")
            axes[1, 0].grid(True, alpha=0.3)

        # Success rate
        if self.success_rate:
            episodes, rates = zip(*self.success_rate)
            axes[1, 1].plot(
                episodes,
                [r * 100 for r in rates],
                "o-",
                color="purple",
                linewidth=2,
                markersize=8,
            )
            axes[1, 1].set_title(
                "Success Rate (Evaluation)", fontsize=12, fontweight="bold"
            )
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Success Rate (%)")
            axes[1, 1].set_ylim(0, 105)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig(f"transfer_learning_{timestamp}.png", dpi=150, bbox_inches="tight")
        plt.show()

    def save(self, filename):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "episode_rewards": self.episode_rewards,
            },
            filename,
        )
        print(f"✅ Model saved: {filename}")

    def load_pretrained(self, model_path):
        """Carga pesos pre-entrenados de Entorno 1"""
        print(f"📥 Loading pretrained model: {model_path}")

        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            pretrained_state = checkpoint["q_network"]

            # Cargar solo capas ocultas
            hidden_layers = {
                k: v
                for k, v in pretrained_state.items()
                if "network.0" in k or "network.2" in k or "network.4" in k
            }

            current_state = self.q_network.state_dict()
            current_state.update(hidden_layers)
            self.q_network.load_state_dict(current_state, strict=False)

            current_state_target = self.target_network.state_dict()
            current_state_target.update(hidden_layers)
            self.target_network.load_state_dict(current_state_target, strict=False)

            print("✅ Pretrained weights loaded successfully!")
            print("   📌 Hidden layers: Transferred from Entorno 1")
            print("   📌 Output layer: Randomly initialized for 6 actions")
            print(f"   📌 Starting epsilon: {self.get_epsilon():.3f}")

            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Training from scratch...")
            return False


def find_best_model_entorno1():
    """Busca el mejor modelo de Entorno 1"""
    model_dir = Path("models")

    if not model_dir.exists():
        model_dir = Path(".")

    patterns = ["*Entorno_1*.pth", "*entorno_1*.pth", "best_model_Entorno_1.pth"]

    models = []
    for pattern in patterns:
        models.extend(glob.glob(str(model_dir / pattern)))

    if not models:
        return None

    latest = max(models, key=lambda x: Path(x).stat().st_mtime)
    return latest


def main():
    print("=" * 60)
    print("🔄 TRANSFER LEARNING: Entorno 1 → Entorno 2 [CORREGIDO]")
    print("=" * 60)

    # Buscar modelo de Entorno 1
    model_path = find_best_model_entorno1()

    if model_path:
        print(f"\n✅ Found Entorno 1 model: {model_path}")
        use_transfer = (
            input("Use this model for transfer learning? (y/n) [default=y]: ")
            .strip()
            .lower()
        )
        if use_transfer == "n":
            model_path = None
    else:
        print("\n⚠️  No Entorno 1 model found")
        model_path = input(
            "Enter model path manually (or press Enter to skip): "
        ).strip()
        if not model_path:
            model_path = None

    # Crear entorno 2
    env = WarehouseEnv(just_pick=False, random_objects=False, render_mode=None)
    feedback = WarehouseFeedback(dims=(10.0, 10.0), delivery_area=(2.5, 9, 5.0, 2.0))

    print(f"\n📐 Input size: {feedback.feature_size}")
    print(f"🎮 Actions: 6")
    print(f"🔧 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Crear agente con hiperparámetros corregidos
    agent = DQNAgent(
        env=env,
        feedback=feedback,
        num_actions=6,
        lr=1e-4,  # ✅ Aumentado
        epsilon_start=0.5,  # ✅ Reducido
        epsilon_decay_steps=150000,
    )

    # Cargar modelo pre-entrenado
    if model_path:
        success = agent.load_pretrained(model_path)
        if success:
            print("\n🔬 Evaluating pretrained model on Entorno 2...")
            eval_reward, eval_success = agent.evaluate(20)
            print(f"   Initial performance:")
            print(f"   - Reward: {eval_reward:.2f}")
            print(f"   - Success: {eval_success*100:.0f}%")

    # Entrenar
    print(f"\n🚀 Starting training...")
    num_episodes = 2500 if model_path else 4000
    print(f"   Episodes: {num_episodes}")
    print(f"   Estimated time: {num_episodes * 0.25 / 60:.0f} minutes\n")

    agent.train(num_episodes=num_episodes, evaluate_every=250, num_eval_episodes=20)

    # Evaluación final
    final_reward, final_success = agent.evaluate(50)
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Average Reward: {final_reward:.2f}")
    print(f"   Success Rate: {final_success*100:.1f}%")

    # Guardar modelo final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    filename = save_dir / f"transfer_entorno2_final_{timestamp}.pth"
    agent.save(str(filename))

    # Verificar si es mejor que el best guardado
    best_path = save_dir / "best_model_Entorno_2.pth"

    if final_success > agent.best_eval_success:
        agent.save(str(best_path))
        print(f"💾 Final model is the best! Saved to {best_path}")
    else:
        print(f"💾 Final model saved to {filename}")
        print(
            f"🏆 Best model remains: Success={agent.best_eval_success*100:.1f}% at {best_path}"
        )

    env.close()
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
