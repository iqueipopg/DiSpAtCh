"""
Validador de modelos para los 3 entornos
========================================
Evalúa un modelo en cualquier entorno con número configurable de episodios.

Uso:
    python validate_model.py modelo.pth --env 1 --episodes 100
    python validate_model.py modelo.pth --env 2 --episodes 50
    python validate_model.py modelo.pth --env 3 --episodes 200
    python validate_model.py modelo.pth --env 3 --episodes 10 --verbose
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
import time

from almacen_alu_v1 import WarehouseEnv
from representacion_wharehouse import WarehouseFeedback


class DQN(nn.Module):
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


def validate_model(model_path, env_type, num_episodes, verbose=False, force_drop=True):
    # Configurar entorno
    if env_type == 1:
        env = WarehouseEnv(
            just_pick=True,  # Equivale a drop=False
            random_objects=False,
            render_mode="human" if verbose else None,
        )
    elif env_type == 2:
        env = WarehouseEnv(
            just_pick=False,
            random_objects=False,
            render_mode="human" if verbose else None,
        )
    else:  # env_type == 3
        env = WarehouseEnv(
            just_pick=False,
            random_objects=True,
            render_mode="human" if verbose else None,
        )

    feedback = WarehouseFeedback(dims=(10.0, 10.0), delivery_area=(2.5, 9, 5.0, 2.0))

    # 🔧 CARGAR MODELO Y DETECTAR NUMERO DE ACCIONES AUTOMÁTICAMENTE
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    saved_num_actions = checkpoint["q_network"]["network.6.bias"].shape[0]

    q_network = DQN(feedback.feature_size, saved_num_actions)
    q_network.load_state_dict(checkpoint["q_network"])
    q_network.eval()

    print(f"   📡 Modelo detectado: {saved_num_actions} acciones")
    print(
        f"   🎯 Entorno {env_type}: {'Pick-only' if env_type==1 else 'Pick+Delivery'}"
    )

    # Métricas
    successes = []
    total_steps = []
    total_rewards = []
    collisions = 0
    timeouts = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        has_object = False

        while True:
            agent_x, agent_y = obs[0], obs[1]
            current_has_object = obs[8] > 0.5

            # 🔧 LÓGICA ADAPTATIVA: Para Entorno 1 termina al recoger
            if env_type == 1:
                # Solo necesitamos recoger objeto (ignora DROP)
                features = feedback.process_observation(obs)
                state_tensor = torch.FloatTensor(features).unsqueeze(0)
                with torch.no_grad():
                    q_values = q_network(state_tensor)
                    # Ignora acción DROP (índice 5) si no tiene objeto
                    if not current_has_object:
                        q_values[0, 5] = float("-inf")  # Deshabilita DROP
                    action = q_values.argmax().item()
            else:
                # Lógica normal para Entorno 2/3
                if force_drop and current_has_object:
                    in_delivery = (2.5 <= agent_x <= 7.5) and (9.0 <= agent_y <= 11.0)
                    if in_delivery:
                        action = 5  # DROP
                    else:
                        features = feedback.process_observation(obs)
                        state_tensor = torch.FloatTensor(features).unsqueeze(0)
                        with torch.no_grad():
                            q_values = q_network(state_tensor)
                        action = q_values.argmax().item()
                else:
                    features = feedback.process_observation(obs)
                    state_tensor = torch.FloatTensor(features).unsqueeze(0)
                    with torch.no_grad():
                        q_values = q_network(state_tensor)
                    action = q_values.argmax().item()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Actualizar estado del objeto
            has_object = obs[8] > 0.5

            if verbose:
                env.render()
                time.sleep(0.05)

            # TERMINAR EN ENTORNO 1 CUANDO RECOGE OBJETO
            if env_type == 1 and has_object and not current_has_object:
                terminated = True  # ¡ÉXITO! Termina al recoger

            if terminated or truncated:
                break

        # Registrar resultado ADAPTADO
        if env_type == 1:
            success = has_object  # Éxito = recogió objeto
        else:
            success = info.get("delivery", False)

        successes.append(1.0 if success else 0.0)
        total_steps.append(steps)
        total_rewards.append(episode_reward)

        if info.get("collision", False):
            collisions += 1
        if truncated:
            timeouts += 1

        # ... resto del código igual (verbose print, etc.)

        if verbose:
            result = "✅" if success else "❌"
            reason = ""
            if info.get("collision"):
                reason = "(colisión)"
            elif truncated:
                reason = "(timeout)"
            print(
                f"Ep {ep+1}: {result} {reason} | Steps: {steps} | Reward: {episode_reward:.1f}"
            )

    env.close()

    # Calcular métricas
    results = {
        "success_rate": np.mean(successes) * 100,
        "success_std": np.std(successes) * 100,
        "avg_steps": np.mean(total_steps),
        "std_steps": np.std(total_steps),
        "avg_reward": np.mean(total_rewards),
        "collisions": collisions,
        "collision_rate": collisions / num_episodes * 100,
        "timeouts": timeouts,
        "timeout_rate": timeouts / num_episodes * 100,
        "num_episodes": num_episodes,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Validar modelo en entornos 1, 2 o 3")
    parser.add_argument("model", help="Archivo .pth del modelo")
    parser.add_argument(
        "--env",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Tipo de entorno (1, 2 o 3)",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Número de episodios para evaluar"
    )
    parser.add_argument("--verbose", action="store_true", help="Mostrar visualización")
    parser.add_argument(
        "--no-force-drop",
        action="store_true",
        help="Desactivar drop forzado en zona delivery",
    )

    args = parser.parse_args()

    env_names = {
        1: "Entorno 1 (Pick)",
        2: "Entorno 2 (Pick+Delivery)",
        3: "Entorno 3 (Random+Delivery)",
    }

    print("=" * 60)
    print(f"🧪 VALIDACIÓN DE MODELO")
    print("=" * 60)
    print(f"   Modelo: {args.model}")
    print(f"   Entorno: {env_names[args.env]}")
    print(f"   Episodios: {args.episodes}")
    print(f"   Drop forzado: {'No' if args.no_force_drop else 'Sí'}")
    print()

    results = validate_model(
        args.model,
        args.env,
        args.episodes,
        verbose=args.verbose,
        force_drop=not args.no_force_drop,
    )

    print("=" * 60)
    print(f"📊 RESULTADOS ({results['num_episodes']} episodios)")
    print("=" * 60)
    print(
        f"   Success Rate: {results['success_rate']:.1f}% ± {results['success_std']:.1f}%"
    )
    print(f"   Avg Steps: {results['avg_steps']:.1f} ± {results['std_steps']:.1f}")
    print(f"   Avg Reward: {results['avg_reward']:.1f}")
    print()
    print(f"   Colisiones: {results['collisions']} ({results['collision_rate']:.1f}%)")
    print(f"   Timeouts: {results['timeouts']} ({results['timeout_rate']:.1f}%)")
    print("=" * 60)

    # Indicador de calidad
    sr = results["success_rate"]
    if sr >= 80:
        print(f"🎉 ¡EXCELENTE! {sr:.1f}% >= 80%")
    elif sr >= 60:
        print(f"👍 Bueno: {sr:.1f}%")
    elif sr >= 40:
        print(f"📈 Aceptable: {sr:.1f}%")
    else:
        print(f"⚠️  Necesita mejora: {sr:.1f}%")


if __name__ == "__main__":
    main()
