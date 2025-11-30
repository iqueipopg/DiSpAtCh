"""
Continuar training desde el mejor modelo guardado.
Uso: python continue_training.py
"""

import torch
from dqn import DQNAgent, train_environment
from almacen_alu_v1 import WarehouseEnv
from representacion_wharehouse import WarehouseFeedback


def continue_from_best():
    # Configuración para Entorno 2
    env = WarehouseEnv(just_pick=False, random_objects=False)
    feedback = WarehouseFeedback(dims=(10.0, 10.0), delivery_area=(2.5, 9, 5.0, 2.0))

    # Crear agente con mismos parámetros
    agent = DQNAgent(
        env=env,
        feedback=feedback,
        input_size=feedback.feature_size,  # 32 features
        num_actions=6,  # 6 acciones en Entorno 2
        lr=5e-5,  # Learning rate bajo para no olvidar
        epsilon_decay_steps=100000,  # Decay más corto, ya tenemos buen modelo
        target_update=1000,
        env_type="Entorno_2",
    )

    # Cargar el mejor modelo
    checkpoint = torch.load(
        "best_model_Entorno_2.pth", map_location=agent.device, weights_only=False
    )
    agent.q_network.load_state_dict(checkpoint["q_network"])
    agent.target_network.load_state_dict(checkpoint["target_network"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])

    # IMPORTANTE: Empezar con epsilon bajo porque ya tenemos un modelo que funciona
    agent.total_steps = 200000  # Esto hará que epsilon empiece en ~0.33

    print("✅ Modelo cargado: best_model_Entorno_2.pth")
    print(f"   Empezando con epsilon ≈ {agent.get_epsilon():.3f}")

    # Evaluar primero para ver cómo está
    print("\n📊 Evaluación inicial:")
    eval_reward, eval_success = agent.evaluate(20)
    print(f"   Reward: {eval_reward:.2f}, Success: {eval_success*100:.0f}%")

    # Continuar training
    print("\n🚀 Continuando training...")
    agent.train(
        num_episodes=3000,  # 3000 episodios más
        evaluate_every=200,
        num_eval_episodes=20,
    )

    # Guardar modelo final
    agent.save("continued_model_Entorno_2.pth")
    agent.plot_training()


if __name__ == "__main__":
    continue_from_best()
