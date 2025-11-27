"""
Visualizar modelo DQN entrenado
================================

Uso:
    python test_model.py best_model_Entorno_1.pth
    python test_model.py best_model_Entorno_2.pth --env 2
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
import time

from almacen_alu_v1 import WarehouseEnv
from representacion_wharehouse import WarehouseFeedback


class DQN(nn.Module):
    """Red neuronal (igual que en dqn.py)"""

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


def test_model(model_path, env_type=1, num_episodes=5, delay=0.1):
    """
    Carga un modelo y lo visualiza.

    Args:
        model_path: Ruta al .pth
        env_type: 1, 2 o 3
        num_episodes: Cuántos episodios mostrar
        delay: Segundos entre pasos (más alto = más lento)
    """

    # Configurar entorno
    if env_type == 1:
        env = WarehouseEnv(just_pick=True, random_objects=False, render_mode="human")
        num_actions = 5
    elif env_type == 2:
        env = WarehouseEnv(just_pick=False, random_objects=False, render_mode="human")
        num_actions = 6
    else:
        env = WarehouseEnv(just_pick=False, random_objects=True, render_mode="human")
        num_actions = 6

    # Feedback
    feedback = WarehouseFeedback(dims=(10.0, 10.0), delivery_area=(2.5, 9, 5.0, 2.0))

    # Cargar modelo
    q_network = DQN(feedback.feature_size, num_actions)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    q_network.load_state_dict(checkpoint["q_network"])
    q_network.eval()

    print(f"✅ Modelo cargado: {model_path}")
    print(f"🎮 Entorno: {env_type}")
    print(f"▶️  Mostrando {num_episodes} episodios...\n")

    # Ejecutar episodios
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        print(f"--- Episodio {ep + 1} ---")

        while True:
            # Obtener acción del modelo
            features = feedback.process_observation(obs)
            state_tensor = torch.FloatTensor(features).unsqueeze(0)

            with torch.no_grad():
                q_values = q_network(state_tensor)
                action = q_values.argmax().item()

            # Ejecutar
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Renderizar
            env.render()
            time.sleep(delay)

            if terminated or truncated:
                break

        # Resultado
        if env_type == 1:
            success = "✅ PICK!" if info.get("has_object") else "❌ Falló"
        else:
            success = "✅ DELIVERY!" if info.get("delivery") else "❌ Falló"

        print(f"   {success} | Pasos: {steps} | Reward: {total_reward:.1f}\n")
        time.sleep(0.5)

    env.close()
    print("🏁 Fin de la demostración")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizar modelo DQN")
    parser.add_argument(
        "model",
        nargs="?",
        default="best_model_Entorno_1.pth",
        help="Archivo .pth del modelo",
    )
    parser.add_argument(
        "--env",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Tipo de entorno (1, 2 o 3)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Número de episodios")
    parser.add_argument(
        "--delay", type=float, default=0.1, help="Delay entre pasos (segundos)"
    )

    args = parser.parse_args()

    test_model(args.model, args.env, args.episodes, args.delay)
