"""
Entrenamiento del Agente para ENTORNO 1: Objetos fijos, solo recogida
======================================================================
Tarea: Aproximarse a cualquier objeto y recogerlo exitosamente
Objetivo: >= 95% success rate

El modelo se guarda SOLO cuando mejora el success rate, por lo que 
si el episodio 1000 tiene mejor rendimiento que el 1500, se mantiene
el modelo del episodio 1000.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Añadir directorio src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from almacen_env import WarehouseEnv
from representacion import WarehouseFeedback
from agente_dqn import DQNAgent


def main():
    print("=" * 70)
    print("ENTRENAMIENTO ENTORNO 1: OBJETOS FIJOS, SOLO RECOGIDA")
    print("=" * 70)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Rutas
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'entorno1_mejor.pth')
    
    # Crear entorno
    env = WarehouseEnv(just_pick=True, random_objects=False, render_mode=None)
    print(f"\n✅ Entorno creado")
    print(f"   - Tarea: Recoger un objeto")
    print(f"   - Objetos: Fijos en posiciones predefinidas")
    print(f"   - Acciones: {env.action_space.n} (4 movimiento + 1 coger)")
    
    # Usar representación RICA con features engineered
    feedback = WarehouseFeedback()
    state_size = feedback.get_feature_size()
    action_size = env.action_space.n
    print(f"\n✅ Representación RICA: {state_size} features")
    print(f"   - Incluye: posición, distancias, direcciones, obstáculos")
    
    # Crear agente DQN
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        feedback=feedback,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10,
        hidden_sizes=[128, 64],
        models_dir=models_dir
    )
    
    print(f"\n✅ Agente DQN (Double DQN) creado")
    print(f"   - Learning rate: 0.001")
    print(f"   - Gamma: 0.99")
    print(f"   - Epsilon: 1.0 → 0.01 (decay: 0.995)")
    print(f"   - Buffer size: 10,000")
    print(f"   - Batch size: 64")
    print(f"   - Arquitectura: [128, 64]")
    print(f"   - Target update: cada 10 episodios")
    
    # Entrenar con early stopping
    num_episodes = 5000  # Máximo, pero para antes si alcanza objetivo
    
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
        target_success_rate=0.96,  # 96% objetivo
        patience=600,
        eval_freq=100,
        save_path=model_path,
        env_name="Entorno 1"
    )
    
    # Guardar modelo final (que es el mejor por success rate)
    agent.save(model_path, use_best=True)
    
    # Evaluación robusta final
    print("\n" + "=" * 70)
    print("EVALUACIÓN ROBUSTA FINAL")
    print("=" * 70)
    results = agent.evaluate_robust(env, num_episodes=500, num_rounds=3)
    
    # Guardar gráficas
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Entorno 1: Objetos Fijos, Solo Recogida', fontsize=14, fontweight='bold')
    
    # Recompensas
    axes[0, 0].plot(agent.training_rewards, alpha=0.3, color='blue')
    window = 100
    if len(agent.training_rewards) >= window:
        moving_avg = np.convolve(agent.training_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(agent.training_rewards)), moving_avg, color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Recompensa Total')
    axes[0, 0].set_title('Progreso de Entrenamiento')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(['Recompensa', f'Media móvil ({window} ep)'])
    
    # Pérdida
    if agent.losses:
        axes[0, 1].plot(agent.losses, alpha=0.5, color='green')
        window_loss = min(1000, len(agent.losses) // 10)
        if len(agent.losses) > window_loss:
            moving_avg_loss = np.convolve(agent.losses, np.ones(window_loss)/window_loss, mode='valid')
            axes[0, 1].plot(range(window_loss-1, len(agent.losses)), moving_avg_loss, color='darkgreen', linewidth=2)
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Pérdida Durante el Entrenamiento')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    
    # Success rate durante entrenamiento
    if agent.success_rates_history:
        episodes = [h['episode'] for h in agent.success_rates_history]
        success_rates = [h['success_rate'] for h in agent.success_rates_history]
        axes[1, 0].plot(episodes, success_rates, 'b-o', markersize=3)
        axes[1, 0].axhline(y=95, color='r', linestyle='--', label='Objetivo (95%)')
        axes[1, 0].axhline(y=agent.best_success_rate, color='g', linestyle='--', 
                          label=f'Mejor ({agent.best_success_rate:.1f}%)')
        axes[1, 0].set_xlabel('Episodio')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_title('Evolución del Success Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 105])
    
    # Información del modelo
    axes[1, 1].axis('off')
    info_text = f"""
    RESUMEN DEL ENTRENAMIENTO
    ─────────────────────────────
    
    Modelo guardado: entorno1_mejor.pth
    
    Mejor episodio: {agent.best_episode}
    Mejor success rate: {agent.best_success_rate:.2f}%
    
    Evaluación final (3x500 eps):
    - Success rate: {results['success_rate']:.2f}% ± {results['success_rate_std']:.2f}%
    - Collision rate: {results['collision_rate']:.2f}%
    - Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}
    
    Objetivo: ≥ 95%
    Estado: {'✅ CUMPLIDO' if results['success_rate'] >= 95 else '❌ NO CUMPLIDO'}
    """
    axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    graph_path = os.path.join(outputs_dir, 'entorno1_training.png')
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráficas guardadas en: {graph_path}")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN ENTORNO 1")
    print("=" * 70)
    print(f"Mejor episodio: {agent.best_episode}")
    print(f"Mejor success rate durante entrenamiento: {agent.best_success_rate:.2f}%")
    print(f"Tasa de éxito final (evaluación): {results['success_rate']:.2f}%")
    print(f"Recompensa promedio final: {results['avg_reward']:.2f}")
    print(f"Modelo guardado: {model_path}")
    print("=" * 70)
    
    env.close()
    plt.close('all')
    
    return results


if __name__ == "__main__":
    main()
