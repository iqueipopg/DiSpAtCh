"""
Agente DQN (Deep Q-Network) para resolver los entornos de almacén
Implementación con Double DQN, Experience Replay Priorizado, y Checkpointing por Success Rate

Características:
- Double DQN para reducir sobreestimación
- Experience Replay con priorización de experiencias positivas
- Checkpointing que guarda SOLO el modelo con mejor success rate
- Early stopping cuando se alcanza el objetivo
- Soporte para GPU (CUDA)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from datetime import datetime


class DQNNetwork(nn.Module):
    """
    Red neuronal para aproximar la función Q
    Arquitectura: MLP con capas configurables
    """
    def __init__(self, state_size, action_size, hidden_sizes=[128, 64]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Capas ocultas
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Buffer de experiencia para Experience Replay con priorización simple
    SOLO experiencias con recompensas POSITIVAS tienen mayor probabilidad de ser sampleadas
    Esto evita catastrophic forgetting
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        # Prioridad: boost SOLO para éxitos, no para fracasos
        if reward >= 100:  # Éxito grande (recoger o entregar)
            priority = 10.0
        elif reward > 0:
            priority = 3.0
        else:
            priority = 1.0  # Experiencias negativas tienen prioridad base
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        n = len(self.buffer)
        if n < batch_size:
            batch_size = n
        
        # Convertir prioridades a probabilidades
        priorities_array = np.array(self.priorities)
        probabilities = priorities_array / priorities_array.sum()
        
        # Samplear con probabilidades ponderadas
        replace = n < batch_size * 2
        indices = np.random.choice(n, batch_size, p=probabilities, replace=replace)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agente DQN con Experience Replay, Target Network, y Checkpointing por Success Rate
    
    IMPORTANTE: El modelo guardado siempre es el que tiene MEJOR SUCCESS RATE durante
    el entrenamiento, no el último entrenado.
    """
    def __init__(self, state_size, action_size, feedback=None,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64,
                 target_update_freq=10, hidden_sizes=[128, 64],
                 models_dir="../models"):
        
        self.state_size = state_size
        self.action_size = action_size
        self.feedback = feedback
        self.models_dir = models_dir
        
        # Hiperparámetros
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device (GPU si está disponible)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Usando dispositivo: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Redes Q (policy y target)
        self.policy_net = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizador y función de pérdida
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss - más estable que MSE
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Métricas de entrenamiento
        self.training_rewards = []
        self.training_lengths = []
        self.losses = []
        self.success_rates_history = []
        self.update_count = 0
        
        # Checkpointing - guardar el mejor modelo por SUCCESS RATE
        self.best_success_rate = 0.0
        self.best_episode = 0
        self._best_weights = None
        
    def get_action(self, state, epsilon=None):
        """
        Selecciona acción usando política epsilon-greedy
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        
        # Procesar estado si hay feedback
        if self.feedback is not None:
            state = self.feedback.process_observation(state)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """
        Almacena experiencia en el replay buffer
        """
        if self.feedback is not None:
            state = self.feedback.process_observation(state)
            next_state = self.feedback.process_observation(next_state)
        
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """
        Entrena la red usando un batch del replay buffer con Double DQN
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Samplear batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Escalar recompensas para estabilidad
        rewards = rewards / 100.0
        
        # Q-values actuales
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q-values objetivo usando DOUBLE DQN
        # La policy net selecciona la mejor acción, la target net evalúa
        with torch.no_grad():
            # Policy net selecciona las mejores acciones
            best_actions = self.policy_net(next_states).argmax(1)
            # Target net evalúa esas acciones
            next_q_values = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calcular pérdida
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """
        Hard update de la target network (copia completa)
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _save_best_weights(self, episode, success_rate):
        """Guarda los mejores pesos internamente"""
        self._best_weights = {
            'policy_net': {k: v.clone().cpu() for k, v in self.policy_net.state_dict().items()},
            'target_net': {k: v.clone().cpu() for k, v in self.target_net.state_dict().items()},
            'optimizer': {k: v if not isinstance(v, torch.Tensor) else v.clone().cpu() 
                         for k, v in self.optimizer.state_dict().items()},
            'episode': episode,
            'success_rate': success_rate,
            'epsilon': self.epsilon
        }
        self.best_success_rate = success_rate
        self.best_episode = episode
    
    def _load_best_weights(self):
        """Carga los mejores pesos guardados"""
        if self._best_weights is not None:
            self.policy_net.load_state_dict(
                {k: v.to(self.device) for k, v in self._best_weights['policy_net'].items()}
            )
            self.target_net.load_state_dict(
                {k: v.to(self.device) for k, v in self._best_weights['target_net'].items()}
            )
    
    def _quick_evaluate(self, env, num_episodes=100):
        """Evaluación rápida para checkpointing y early stopping"""
        success_count = 0
        collision_count = 0
        total_rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.get_action(state, epsilon=0.01)  # Casi greedy
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            if env.delivery or (env.just_pick and env.agent_has_object):
                success_count += 1
            elif env.collision:
                collision_count += 1
        
        return {
            'success_rate': (success_count / num_episodes) * 100,
            'collision_rate': (collision_count / num_episodes) * 100,
            'avg_reward': np.mean(total_rewards)
        }
    
    def train(self, env, num_episodes=1000, verbose=True, early_stopping=False, 
              target_success_rate=None, patience=500, eval_freq=100,
              save_path=None, env_name="entorno"):
        """
        Entrena el agente con checkpointing basado en SUCCESS RATE
        
        Args:
            env: Entorno de gymnasium
            num_episodes: Número máximo de episodios
            verbose: Mostrar progreso
            early_stopping: Si True, para cuando alcanza el objetivo
            target_success_rate: Tasa de éxito objetivo (ej: 0.95 para 95%)
            patience: Episodios sin mejora antes de parar
            eval_freq: Frecuencia de evaluación para checkpointing
            save_path: Ruta donde guardar el mejor modelo
            env_name: Nombre del entorno para logging
        """
        episodes_without_improvement = 0
        start_time = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"ENTRENAMIENTO - {env_name.upper()}")
        print(f"{'='*70}")
        print(f"Episodios máximos: {num_episodes}")
        print(f"Target success rate: {target_success_rate*100 if target_success_rate else 'N/A'}%")
        print(f"Evaluación cada: {eval_freq} episodios")
        print(f"{'='*70}\n")
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Seleccionar acción
                action = self.get_action(state)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Almacenar experiencia
                self.remember(state, action, reward, next_state, done)
                
                # Entrenar
                loss = self.replay()
                if loss is not None:
                    self.losses.append(loss)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Actualizar target network periódicamente
            if episode % self.target_update_freq == 0:
                self.update_target_network()
            
            # Decrecer epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Guardar métricas
            self.training_rewards.append(total_reward)
            self.training_lengths.append(steps)
            
            # Mostrar progreso
            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.training_rewards[-100:]) if len(self.training_rewards) >= 100 else np.mean(self.training_rewards)
                avg_loss = np.mean(self.losses[-100:]) if len(self.losses) >= 100 else (np.mean(self.losses) if self.losses else 0)
                print(f"Episode {episode:5d}/{num_episodes} | "
                      f"Reward: {avg_reward:7.2f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Buffer: {len(self.memory):5d}")
            
            # Evaluación y checkpointing basado en SUCCESS RATE
            if episode > 0 and episode % eval_freq == 0:
                eval_results = self._quick_evaluate(env, num_episodes=100)
                current_success_rate = eval_results['success_rate']
                
                self.success_rates_history.append({
                    'episode': episode,
                    'success_rate': current_success_rate,
                    'collision_rate': eval_results['collision_rate'],
                    'avg_reward': eval_results['avg_reward']
                })
                
                if verbose:
                    print(f"  → Evaluación: {current_success_rate:.1f}% éxito | "
                          f"{eval_results['collision_rate']:.1f}% colisión | "
                          f"reward: {eval_results['avg_reward']:.1f}")
                
                # CHECKPOINTING: Guardar SOLO si mejora el success rate
                if current_success_rate > self.best_success_rate:
                    old_best = self.best_success_rate
                    self._save_best_weights(episode, current_success_rate)
                    episodes_without_improvement = 0
                    
                    if verbose:
                        print(f"  ✅ NUEVO MEJOR MODELO: {current_success_rate:.1f}% "
                              f"(anterior: {old_best:.1f}%) - Episodio {episode}")
                    
                    # Guardar a disco inmediatamente
                    if save_path:
                        self._save_checkpoint_to_disk(save_path, episode, current_success_rate)
                else:
                    episodes_without_improvement += eval_freq
                
                # Early stopping por objetivo alcanzado
                if early_stopping and target_success_rate:
                    if current_success_rate / 100.0 >= target_success_rate:
                        print(f"\n✅ EARLY STOPPING: Objetivo alcanzado!")
                        print(f"   Success rate: {current_success_rate:.1f}% >= {target_success_rate*100:.1f}%")
                        print(f"   Episodio: {episode}")
                        break
                
                # Early stopping por estabilización
                if early_stopping and episodes_without_improvement >= patience:
                    print(f"\n⏹️  EARLY STOPPING: Sin mejora en {patience} episodios")
                    print(f"   Mejor success rate: {self.best_success_rate:.1f}% (episodio {self.best_episode})")
                    break
        
        # Fin del entrenamiento
        elapsed = datetime.now() - start_time
        print(f"\n{'='*70}")
        print(f"ENTRENAMIENTO COMPLETADO")
        print(f"{'='*70}")
        print(f"Tiempo total: {elapsed}")
        print(f"Episodios entrenados: {episode + 1}")
        print(f"Mejor success rate: {self.best_success_rate:.1f}% (episodio {self.best_episode})")
        print(f"{'='*70}")
        
        # Cargar los mejores pesos antes de terminar
        if self._best_weights is not None:
            self._load_best_weights()
            print(f"✅ Cargados los pesos del mejor modelo (episodio {self.best_episode})")
    
    def _save_checkpoint_to_disk(self, filepath, episode, success_rate):
        """Guarda el checkpoint actual a disco"""
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_episode': episode,
            'best_success_rate': success_rate,
            'training_rewards': self.training_rewards,
            'training_lengths': self.training_lengths,
            'losses': self.losses,
            'success_rates_history': self.success_rates_history
        }
        
        torch.save(checkpoint, filepath)
    
    def save(self, filepath, use_best=True):
        """
        Guarda el agente a disco
        
        Args:
            filepath: Ruta del archivo
            use_best: Si True, guarda el mejor modelo (por defecto True)
        """
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        if use_best and self._best_weights is not None:
            self._load_best_weights()
            print(f"✅ Guardando el mejor modelo (episodio {self.best_episode}, "
                  f"success rate: {self.best_success_rate:.1f}%)")
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_episode': self.best_episode,
            'best_success_rate': self.best_success_rate,
            'training_rewards': self.training_rewards,
            'training_lengths': self.training_lengths,
            'losses': self.losses,
            'success_rates_history': self.success_rates_history,
            'state_size': self.state_size,
            'action_size': self.action_size
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Agente guardado en: {filepath}")
    
    def load(self, filepath):
        """
        Carga el agente desde disco
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.best_episode = checkpoint.get('best_episode', 0)
        self.best_success_rate = checkpoint.get('best_success_rate', 0.0)
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.training_lengths = checkpoint.get('training_lengths', [])
        self.losses = checkpoint.get('losses', [])
        self.success_rates_history = checkpoint.get('success_rates_history', [])
        
        print(f"📂 Agente cargado desde: {filepath}")
        print(f"   Mejor episodio: {self.best_episode}")
        print(f"   Mejor success rate: {self.best_success_rate:.1f}%")
    
    def evaluate(self, env, num_episodes=100, render=False):
        """
        Evalúa el agente
        """
        total_rewards = []
        success_count = 0
        collision_count = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state, epsilon=0.01)  # Casi greedy
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if render:
                    env.render()
                
                state = next_state
                total_reward += reward
                
                if done:
                    if env.delivery or (env.just_pick and env.agent_has_object):
                        success_count += 1
                    elif env.collision:
                        collision_count += 1
            
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        success_rate = (success_count / num_episodes) * 100
        collision_rate = (collision_count / num_episodes) * 100
        
        print(f"\n{'='*60}")
        print(f"EVALUACIÓN: {num_episodes} episodios")
        print(f"{'='*60}")
        print(f"Recompensa promedio: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Tasa de éxito: {success_rate:.2f}%")
        print(f"Tasa de colisión: {collision_rate:.2f}%")
        print(f"{'='*60}\n")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'success_rate': success_rate,
            'collision_rate': collision_rate
        }
    
    def evaluate_robust(self, env, num_episodes=500, num_rounds=3):
        """
        Evaluación robusta: ejecuta múltiples rondas y calcula estadísticas.
        
        Args:
            env: Entorno
            num_episodes: Episodios por ronda
            num_rounds: Número de rondas de evaluación
        
        Returns:
            dict con media y desviación de las métricas
        """
        all_success_rates = []
        all_collision_rates = []
        all_rewards = []
        
        print(f"\n{'='*60}")
        print(f"EVALUACIÓN ROBUSTA: {num_rounds} rondas x {num_episodes} episodios")
        print(f"{'='*60}")
        
        for round_num in range(num_rounds):
            success_count = 0
            collision_count = 0
            round_rewards = []
            
            for _ in range(num_episodes):
                state, _ = env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    action = self.get_action(state, epsilon=0.01)
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                
                round_rewards.append(total_reward)
                if env.delivery or (env.just_pick and env.agent_has_object):
                    success_count += 1
                elif env.collision:
                    collision_count += 1
            
            success_rate = (success_count / num_episodes) * 100
            collision_rate = (collision_count / num_episodes) * 100
            avg_reward = np.mean(round_rewards)
            
            all_success_rates.append(success_rate)
            all_collision_rates.append(collision_rate)
            all_rewards.append(avg_reward)
            
            print(f"  Ronda {round_num + 1}: {success_rate:.1f}% éxito | "
                  f"{collision_rate:.1f}% colisión | "
                  f"reward: {avg_reward:.1f}")
        
        # Calcular estadísticas finales
        final_success = np.mean(all_success_rates)
        final_success_std = np.std(all_success_rates)
        final_collision = np.mean(all_collision_rates)
        final_reward = np.mean(all_rewards)
        final_reward_std = np.std(all_rewards)
        
        print(f"{'='*60}")
        print(f"RESULTADO FINAL (media de {num_rounds} rondas):")
        print(f"  Tasa de éxito: {final_success:.2f}% ± {final_success_std:.2f}%")
        print(f"  Tasa de colisión: {final_collision:.2f}%")
        print(f"  Recompensa: {final_reward:.2f} ± {final_reward_std:.2f}")
        print(f"{'='*60}\n")
        
        return {
            'success_rate': final_success,
            'success_rate_std': final_success_std,
            'collision_rate': final_collision,
            'avg_reward': final_reward,
            'std_reward': final_reward_std,
            'all_success_rates': all_success_rates
        }


if __name__ == "__main__":
    print("="*60)
    print("MÓDULO DQN AGENT")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print("="*60)
