"""
Visualiza un agente DQN entrenado en el entorno de almacén.
============================================================
Requiere haber ejecutado antes entrenar_entornoX.py

Uso: 
    python visualizar_agente.py 1        # Visualizar entorno 1
    python visualizar_agente.py 2 5      # Visualizar entorno 2, 5 episodios
    python visualizar_agente.py 3 10     # Visualizar entorno 3, 10 episodios
"""
import os
import sys

# Añadir directorio src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from almacen_env import WarehouseEnv
from representacion import WarehouseFeedback
from agente_dqn import DQNAgent


def main():
    # Configuración por defecto
    entorno = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    # Rutas
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    configs = {
        1: (True, False, 'entorno1_mejor.pth', 'Entorno 1: Objetos fijos, solo recogida'),
        2: (False, False, 'entorno2_mejor.pth', 'Entorno 2: Objetos fijos, recogida y entrega'),
        3: (False, True, 'entorno3_mejor.pth', 'Entorno 3: Objetos aleatorios, recogida y entrega')
    }
    
    if entorno not in configs:
        print("Uso: python visualizar_agente.py [1|2|3] [num_episodios]")
        print("  1 - Entorno 1: Objetos fijos, solo recogida")
        print("  2 - Entorno 2: Objetos fijos, recogida y entrega")
        print("  3 - Entorno 3: Objetos aleatorios, recogida y entrega")
        sys.exit(1)
    
    just_pick, random_obj, archivo, nombre = configs[entorno]
    model_path = os.path.join(models_dir, archivo)
    
    print("=" * 70)
    print(f"VISUALIZACIÓN: {nombre}")
    print("=" * 70)
    print(f"Modelo: {archivo}")
    print(f"Episodios: {num_eps}")
    print("=" * 70)
    
    # Cargar agente
    print(f"\nCargando {archivo}...")
    env = WarehouseEnv(just_pick=just_pick, random_objects=random_obj)
    feedback = WarehouseFeedback()
    agent = DQNAgent(
        feedback.get_feature_size(), 
        env.action_space.n, 
        feedback, 
        hidden_sizes=[128, 64]
    )
    
    try:
        agent.load(model_path)
    except FileNotFoundError:
        print(f"\n❌ Error: No se encontró {model_path}")
        print(f"   Por favor, entrena primero el agente ejecutando:")
        print(f"   python entrenar_entorno{entorno}.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error al cargar modelo: {e}")
        sys.exit(1)
    
    env.close()
    
    # Crear entorno con visualización
    env = WarehouseEnv(just_pick=just_pick, random_objects=random_obj, render_mode='human')
    
    # Estadísticas
    successes = 0
    collisions = 0
    timeouts = 0
    total_rewards = []
    total_steps = []
    
    print("\n" + "=" * 70)
    print("INICIANDO VISUALIZACIÓN")
    print("=" * 70)
    print("Presiona Ctrl+C para detener\n")
    
    try:
        for ep in range(num_eps):
            print(f"--- Episodio {ep+1}/{num_eps} ---")
            state, _ = env.reset()
            steps = 0
            episode_reward = 0
            done = False
            
            while steps < 200 and not done:
                action = agent.get_action(state, epsilon=0.01)  # Casi greedy
                state, reward, term, trunc, _ = env.step(action)
                env.render()
                steps += 1
                episode_reward += reward
                done = term or trunc
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
            
            # Determinar resultado
            if env.delivery or (just_pick and env.agent_has_object):
                print(f"✅ Éxito en {steps} pasos | Reward: {episode_reward:.1f}")
                successes += 1
            elif env.collision:
                print(f"❌ Colisión en {steps} pasos | Reward: {episode_reward:.1f}")
                collisions += 1
            else:
                print(f"⏱️ Timeout en {steps} pasos | Reward: {episode_reward:.1f}")
                timeouts += 1
                
    except KeyboardInterrupt:
        print("\n\n⏹️ Visualización interrumpida por el usuario")
    
    env.close()
    
    # Resumen
    total_eps = successes + collisions + timeouts
    if total_eps > 0:
        print("\n" + "=" * 70)
        print("RESUMEN DE VISUALIZACIÓN")
        print("=" * 70)
        print(f"Episodios completados: {total_eps}")
        print(f"Éxitos: {successes} ({100*successes/total_eps:.1f}%)")
        print(f"Colisiones: {collisions} ({100*collisions/total_eps:.1f}%)")
        print(f"Timeouts: {timeouts} ({100*timeouts/total_eps:.1f}%)")
        print(f"Reward promedio: {sum(total_rewards)/len(total_rewards):.2f}")
        print(f"Pasos promedio: {sum(total_steps)/len(total_steps):.1f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
