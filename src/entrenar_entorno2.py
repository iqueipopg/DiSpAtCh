"""
Entrenamiento del Agente para ENTORNO 2: Objetos fijos, recogida y entrega
==========================================================================
Tarea: Recoger un objeto Y entregarlo en la zona de entrega
Objetivo: >= 90% success rate

Usa Transfer Learning desde el agente del entorno 1.
El modelo se guarda SOLO cuando mejora el success rate.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Añadir directorio src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from almacen_env import WarehouseEnv
from representacion import WarehouseFeedback
from agente_dqn import DQNAgent


def load_pretrained_weights(agent, pretrained_path, verbose=True):
    """
    Carga pesos preentrenados del entorno 1 (5 acciones) al entorno 2 (6 acciones)
    Transfiere todas las capas excepto la última, que se adapta parcialmente.
    """
    if verbose:
        print("\n🔄 TRANSFER LEARNING desde Entorno 1")

    try:
        checkpoint = torch.load(
            pretrained_path, map_location=agent.device, weights_only=False
        )

        if "policy_net_state_dict" in checkpoint:
            pretrained_state = checkpoint["policy_net_state_dict"]
        else:
            pretrained_state = checkpoint

        current_state = agent.policy_net.state_dict()

        layers_loaded = 0
        layers_partial = 0

        for name, param in pretrained_state.items():
            if name in current_state:
                if param.shape == current_state[name].shape:
                    # Capa compatible - copiar completa
                    current_state[name] = param.clone()
                    layers_loaded += 1
                    if verbose:
                        print(f"   ✅ Capa completa: {name}")
                else:
                    # Última capa tiene diferente tamaño (5 vs 6 acciones)
                    if "weight" in name and param.shape[0] == 5:
                        current_state[name][:5, :] = param.clone()
                        layers_partial += 1
                        if verbose:
                            print(f"   🔶 Capa parcial: {name} (5 de 6 acciones)")
                    elif "bias" in name and param.shape[0] == 5:
                        current_state[name][:5] = param.clone()
                        layers_partial += 1
                        if verbose:
                            print(f"   🔶 Capa parcial: {name} (5 de 6 acciones)")

        agent.policy_net.load_state_dict(current_state)
        agent.target_net.load_state_dict(current_state)

        if verbose:
            print(f"\n   ✅ Transfer learning completado")
            print(f"      - Capas completas: {layers_loaded}")
            print(f"      - Capas parciales: {layers_partial}")

        return True

    except FileNotFoundError:
        if verbose:
            print(f"   ⚠️  No se encontró {pretrained_path}")
            print(f"      Entrenando desde cero...")
        return False
    except Exception as e:
        if verbose:
            print(f"   ⚠️  Error al cargar: {e}")
            print(f"      Entrenando desde cero...")
        return False


def main():
    print("=" * 70)
    print("ENTRENAMIENTO ENTORNO 2: OBJETOS FIJOS, RECOGIDA Y ENTREGA")
    print("=" * 70)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Rutas
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "entorno2_mejor.pth")
    pretrained_path = os.path.join(models_dir, "entorno1_mejor.pth")

    # Crear entorno
    env = WarehouseEnv(just_pick=False, random_objects=False, render_mode=None)
    print(f"\n✅ Entorno creado")
    print(f"   - Tarea: Recoger un objeto Y entregarlo")
    print(f"   - Objetos: Fijos en posiciones predefinidas")
    print(f"   - Acciones: {env.action_space.n} (4 movimiento + 1 coger + 1 soltar)")

    # Usar representación RICA
    feedback = WarehouseFeedback()
    state_size = feedback.get_feature_size()
    action_size = env.action_space.n
    print(f"\n✅ Representación RICA: {state_size} features")

    # Crear agente DQN con hiperparámetros optimizados para transfer learning
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        feedback=feedback,
        learning_rate=0.0005,  # LR más bajo para fine-tuning
        gamma=0.99,
        epsilon=1.0,  # EMPEZAR ALTO - necesita explorar acción 5 (soltar)
        epsilon_min=0.05,  # Mínimo más alto para seguir explorando
        epsilon_decay=0.9997,  # Decay MUY lento - mantener exploración
        buffer_size=20000,  # Buffer más grande
        batch_size=64,
        target_update_freq=10,
        hidden_sizes=[128, 64],
        models_dir=models_dir,
    )

    print(f"\n✅ Agente DQN creado")
    print(f"   - Learning rate: 0.0005")
    print(f"   - Gamma: 0.99")
    print(f"   - Epsilon: 1.0 → 0.05 (decay: 0.9997)")
    print(f"   - Buffer size: 20,000")
    print(f"   - Batch size: 64")
    print(f"   - Arquitectura: [128, 64]")

    # Transfer Learning desde Entorno 1
    load_pretrained_weights(agent, pretrained_path, verbose=True)

    # Entrenar
    num_episodes = 10000  # Máximo

    print("\n" + "=" * 70)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    print("NOTA: El modelo se guarda SOLO cuando mejora el SUCCESS RATE")
    print("=" * 70)

    agent.train(
        env,
        num_episodes=num_episodes,
        verbose=True,
        early_stopping=True,
        target_success_rate=0.92,  # 92% objetivo
        patience=1500,
        eval_freq=200,
        save_path=model_path,
        env_name="Entorno 2",
    )

    # Guardar modelo final
    agent.save(model_path, use_best=True)

    # Evaluación robusta final
    print("\n" + "=" * 70)
    print("EVALUACIÓN ROBUSTA FINAL")
    print("=" * 70)
    results = agent.evaluate_robust(env, num_episodes=500, num_rounds=3)

    # Guardar gráficas
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Entorno 2: Objetos Fijos, Recogida y Entrega", fontsize=14, fontweight="bold"
    )

    # Recompensas
    axes[0, 0].plot(agent.training_rewards, alpha=0.3, color="blue")
    window = 100
    if len(agent.training_rewards) >= window:
        moving_avg = np.convolve(
            agent.training_rewards, np.ones(window) / window, mode="valid"
        )
        axes[0, 0].plot(
            range(window - 1, len(agent.training_rewards)),
            moving_avg,
            color="red",
            linewidth=2,
        )
    axes[0, 0].set_xlabel("Episodio")
    axes[0, 0].set_ylabel("Recompensa Total")
    axes[0, 0].set_title("Progreso de Entrenamiento")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(["Recompensa", f"Media móvil ({window} ep)"])

    # Pérdida
    if agent.losses:
        axes[0, 1].plot(agent.losses, alpha=0.5, color="green")
        window_loss = min(1000, len(agent.losses) // 10)
        if len(agent.losses) > window_loss:
            moving_avg_loss = np.convolve(
                agent.losses, np.ones(window_loss) / window_loss, mode="valid"
            )
            axes[0, 1].plot(
                range(window_loss - 1, len(agent.losses)),
                moving_avg_loss,
                color="darkgreen",
                linewidth=2,
            )
        axes[0, 1].set_xlabel("Update Step")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Pérdida Durante el Entrenamiento")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")

    # Success rate durante entrenamiento
    if agent.success_rates_history:
        episodes = [h["episode"] for h in agent.success_rates_history]
        success_rates = [h["success_rate"] for h in agent.success_rates_history]
        axes[1, 0].plot(episodes, success_rates, "b-o", markersize=3)
        axes[1, 0].axhline(y=90, color="r", linestyle="--", label="Objetivo (90%)")
        axes[1, 0].axhline(
            y=agent.best_success_rate,
            color="g",
            linestyle="--",
            label=f"Mejor ({agent.best_success_rate:.1f}%)",
        )
        axes[1, 0].set_xlabel("Episodio")
        axes[1, 0].set_ylabel("Success Rate (%)")
        axes[1, 0].set_title("Evolución del Success Rate")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 105])

    # Información del modelo
    axes[1, 1].axis("off")
    info_text = f"""
    RESUMEN DEL ENTRENAMIENTO
    ─────────────────────────────
    
    Modelo guardado: entorno2_mejor.pth
    Transfer Learning: Desde entorno1_mejor.pth
    
    Mejor episodio: {agent.best_episode}
    Mejor success rate: {agent.best_success_rate:.2f}%
    
    Evaluación final (3x500 eps):
    - Success rate: {results['success_rate']:.2f}% ± {results['success_rate_std']:.2f}%
    - Collision rate: {results['collision_rate']:.2f}%
    - Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}
    
    Objetivo: ≥ 90%
    Estado: {'✅ CUMPLIDO' if results['success_rate'] >= 90 else '❌ NO CUMPLIDO'}
    """
    axes[1, 1].text(
        0.1,
        0.5,
        info_text,
        transform=axes[1, 1].transAxes,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    graph_path = os.path.join(outputs_dir, "entorno2_training.png")
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Gráficas guardadas en: {graph_path}")

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN ENTORNO 2")
    print("=" * 70)
    print(f"Mejor episodio: {agent.best_episode}")
    print(f"Mejor success rate durante entrenamiento: {agent.best_success_rate:.2f}%")
    print(f"Tasa de éxito final (evaluación): {results['success_rate']:.2f}%")
    print(f"Recompensa promedio final: {results['avg_reward']:.2f}")
    print(f"Modelo guardado: {model_path}")
    print("=" * 70)

    env.close()
    plt.close("all")

    return results


if __name__ == "__main__":
    main()
