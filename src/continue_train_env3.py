"""
Continuar training Entorno 3 desde el mejor modelo
"""

import torch
from pathlib import Path
from almacen_alu_v3 import WarehouseEnv
from representacion_wharehouse import WarehouseFeedback
from dqn3 import DQNAgent


def main():
    print("=" * 60)
    print("🔧 FINE-TUNING Entorno 3")
    print("=" * 60)

    model_path = Path("models/best_model_Entorno_3_52.pth")

    if not model_path.exists():
        print(f"❌ No encontrado: {model_path}")
        return

    env = WarehouseEnv(render_mode=None)
    feedback = WarehouseFeedback(dims=(10.0, 10.0), delivery_area=(2.5, 9, 5.0, 2.0))

    agent = DQNAgent(
        env=env,
        feedback=feedback,
        num_actions=6,
        lr=3e-5,
        epsilon_start=0.15,
        epsilon_end=0.05,
        epsilon_decay_steps=50000,
    )

    agent.load_pretrained(str(model_path))

    print("\n📊 Evaluación inicial:")
    eval_reward, eval_success = agent.evaluate(30)
    print(f"   Reward: {eval_reward:.2f}")
    print(f"   Success: {eval_success*100:.0f}%")

    agent.best_eval_success = eval_success
    agent.best_eval_reward = eval_reward

    print(f"\n🚀 Fine-tuning con lr=3e-5")
    agent.train(num_episodes=2000, evaluate_every=200, num_eval_episodes=30)

    final_reward, final_success = agent.evaluate(50)
    print(f"\n🎯 FINAL: Success={final_success*100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
