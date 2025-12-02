"""
Evaluación completa de los 3 agentes con gráficos personalizados
================================================================
Genera gráficos diferentes y adicionales para el análisis.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from almacen_env import WarehouseEnv
from representacion import WarehouseFeedback
from agente_dqn import DQNAgent


def evaluate_detailed(agent, env, num_episodes=500):
    """
    Evaluación detallada que guarda métricas por episodio
    """
    results = {
        "rewards": [],
        "steps": [],
        "success": [],
        "collision": [],
        "timeout": [],
    }

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.get_action(state, epsilon=0.01)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        results["rewards"].append(total_reward)
        results["steps"].append(steps)

        if env.delivery or (env.just_pick and env.agent_has_object):
            results["success"].append(1)
            results["collision"].append(0)
            results["timeout"].append(0)
        elif env.collision:
            results["success"].append(0)
            results["collision"].append(1)
            results["timeout"].append(0)
        else:
            results["success"].append(0)
            results["collision"].append(0)
            results["timeout"].append(1)

    return results


def load_agent(model_path, env):
    """Carga un agente desde archivo"""
    feedback = WarehouseFeedback()
    agent = DQNAgent(
        state_size=feedback.get_feature_size(),
        action_size=env.action_space.n,
        feedback=feedback,
        hidden_sizes=[128, 64],
    )
    agent.load(model_path)
    return agent


def plot_reward_distribution(all_results, names, output_path):
    """
    Gráfico 1: Distribución de recompensas (violin plot)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Paleta personalizada: morado, naranja, turquesa
    colors = ["#9b59b6", "#e67e22", "#1abc9c"]

    data = [r["rewards"] for r in all_results]
    parts = ax.violinplot(
        data, positions=range(len(names)), showmeans=True, showmedians=True
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("white")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Recompensa", fontsize=12)
    ax.set_title(
        "Distribución de Recompensas por Entorno", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)

    # Añadir estadísticas
    for i, rewards in enumerate(data):
        mean = np.mean(rewards)
        std = np.std(rewards)
        ax.annotate(
            f"μ={mean:.0f}\nσ={std:.0f}",
            xy=(i, max(rewards)),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Guardado: {os.path.basename(output_path)}")


def plot_success_pie_charts(all_results, names, output_path):
    """
    Gráfico 2: Distribución de resultados (pie charts)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Distribución de Resultados por Entorno", fontsize=14, fontweight="bold"
    )

    # Colores: éxito=turquesa, colisión=rojo coral, timeout=amarillo mostaza
    colors = ["#1abc9c", "#c0392b", "#d4ac0d"]
    labels = ["Éxito", "Colisión", "Timeout"]

    for i, (results, name) in enumerate(zip(all_results, names)):
        success_rate = np.mean(results["success"]) * 100
        collision_rate = np.mean(results["collision"]) * 100
        timeout_rate = np.mean(results["timeout"]) * 100

        sizes = [success_rate, collision_rate, timeout_rate]

        # Solo mostrar labels si el valor es > 0
        explode = (0.05, 0, 0)

        wedges, texts, autotexts = axes[i].pie(
            sizes,
            explode=explode,
            colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 0.5 else "",
            startangle=90,
            pctdistance=0.6,
        )

        axes[i].set_title(f"{name}\n({success_rate:.1f}% éxito)", fontsize=11)

    # Leyenda común
    fig.legend(
        labels, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Guardado: {os.path.basename(output_path)}")


def plot_steps_histogram(all_results, names, output_path):
    """
    Gráfico 3: Histograma de pasos por episodio
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Distribución de Pasos por Episodio", fontsize=14, fontweight="bold")

    # Paleta: morado, naranja, turquesa
    colors = ["#9b59b6", "#e67e22", "#1abc9c"]

    for i, (results, name, color) in enumerate(zip(all_results, names, colors)):
        steps = results["steps"]

        axes[i].hist(steps, bins=30, color=color, alpha=0.7, edgecolor="black")
        axes[i].axvline(
            np.mean(steps),
            color="red",
            linestyle="--",
            label=f"Media: {np.mean(steps):.0f}",
        )
        axes[i].axvline(
            np.median(steps),
            color="orange",
            linestyle=":",
            label=f"Mediana: {np.median(steps):.0f}",
        )
        axes[i].set_xlabel("Pasos")
        axes[i].set_ylabel("Frecuencia")
        axes[i].set_title(name, fontsize=11)
        axes[i].legend(fontsize=8)
        axes[i].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Guardado: {os.path.basename(output_path)}")


def plot_comparison_bars(all_results, names, targets, output_path):
    """
    Gráfico 4: Comparación con objetivos (barras horizontales)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    success_rates = [np.mean(r["success"]) * 100 for r in all_results]
    collision_rates = [np.mean(r["collision"]) * 100 for r in all_results]

    y_pos = np.arange(len(names))

    # Barras de success rate - color turquesa
    bars = ax.barh(
        y_pos, success_rates, height=0.4, color="#1abc9c", alpha=0.8, label="Conseguido"
    )

    # Líneas de objetivo - morado oscuro
    for i, target in enumerate(targets):
        ax.axvline(x=target, color="#8e44ad", linestyle="--", alpha=0.5)
        ax.plot(target, i, "o", color="#8e44ad", markersize=10, markeredgewidth=2)

    # Marcar si cumple o no
    for i, (success, target) in enumerate(zip(success_rates, targets)):
        symbol = "✓" if success >= target else "✗"
        color = "green" if success >= target else "red"
        ax.annotate(
            f"{success:.1f}% {symbol}",
            xy=(success + 1, i),
            va="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Success Rate (%)", fontsize=12)
    ax.set_title("Rendimiento vs Objetivos", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)

    # Leyenda
    objetivo_patch = mpatches.Patch(color="#8e44ad", alpha=0.5, label="Objetivo")
    ax.legend(handles=[bars[0], objetivo_patch], loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Guardado: {os.path.basename(output_path)}")


def plot_reward_boxplot(all_results, names, output_path):
    """
    Gráfico 5: Box plot de recompensas
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [r["rewards"] for r in all_results]

    bp = ax.boxplot(data, labels=names, patch_artist=True)

    # Paleta: morado, naranja, turquesa
    colors = ["#9b59b6", "#e67e22", "#1abc9c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Recompensa", fontsize=12)
    ax.set_title(
        "Distribución de Recompensas (Box Plot)", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)

    # Añadir puntos de media - naranja oscuro
    means = [np.mean(d) for d in data]
    ax.scatter(
        range(1, len(names) + 1),
        means,
        color="#d35400",
        marker="D",
        s=50,
        zorder=5,
        label="Media",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Guardado: {os.path.basename(output_path)}")


def plot_cumulative_success(all_results, names, output_path):
    """
    Gráfico 6: Éxito acumulado a lo largo de los episodios
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Paleta: morado, naranja, turquesa
    colors = ["#9b59b6", "#e67e22", "#1abc9c"]

    for results, name, color in zip(all_results, names, colors):
        cumsum = np.cumsum(results["success"])
        episodes = np.arange(1, len(cumsum) + 1)
        success_rate = cumsum / episodes * 100

        ax.plot(episodes, success_rate, color=color, linewidth=2, label=name)

    ax.set_xlabel("Episodio", fontsize=12)
    ax.set_ylabel("Success Rate Acumulado (%)", fontsize=12)
    ax.set_title(
        "Evolución del Success Rate Durante Evaluación", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Guardado: {os.path.basename(output_path)}")


def plot_summary_table(all_results, names, targets, output_path):
    """
    Gráfico 7: Tabla resumen visual
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    # Calcular métricas
    table_data = []
    for results, name, target in zip(all_results, names, targets):
        success = np.mean(results["success"]) * 100
        success_std = np.std(
            [np.mean(results["success"][i : i + 100]) * 100 for i in range(0, 400, 100)]
        )
        collision = np.mean(results["collision"]) * 100
        avg_reward = np.mean(results["rewards"])
        avg_steps = np.mean(results["steps"])
        status = "✓ CUMPLE" if success >= target else "✗ NO CUMPLE"

        table_data.append(
            [
                name,
                f"{success:.1f}% ± {success_std:.1f}%",
                f"≥ {target}%",
                f"{collision:.1f}%",
                f"{avg_reward:.1f}",
                f"{avg_steps:.0f}",
                status,
            ]
        )

    columns = [
        "Entorno",
        "Success Rate",
        "Objetivo",
        "Colisiones",
        "Reward Medio",
        "Pasos Medios",
        "Estado",
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.18, 0.15, 0.10, 0.12, 0.13, 0.12, 0.14],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Estilo
    for i in range(len(table_data) + 1):
        for j in range(len(columns)):
            cell = table[(i, j)]
            if i == 0:
                # Header: morado oscuro
                cell.set_facecolor("#6c3483")
                cell.set_text_props(weight="bold", color="white")
            else:
                # Filas alternadas: gris claro / blanco
                cell.set_facecolor("#f5f5f5" if i % 2 == 0 else "white")
                # Colorear estado
                if j == len(columns) - 1:
                    if "✓" in table_data[i - 1][j]:
                        cell.set_text_props(color="#1abc9c", weight="bold")
                    else:
                        cell.set_text_props(color="#c0392b", weight="bold")

    ax.set_title("Resumen de Evaluación", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Guardado: {os.path.basename(output_path)}")


def main():
    print("=" * 70)
    print("EVALUACIÓN COMPLETA CON GRÁFICOS PERSONALIZADOS")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    configs = [
        ("Entorno 1", "entorno1_mejor.pth", True, False, 95),
        ("Entorno 2", "entorno2_mejor.pth", False, False, 90),
        ("Entorno 3", "entorno3_mejor.pth", False, True, 85),
    ]

    all_results = []
    names = []
    targets = []

    for name, filename, just_pick, random_obj, target in configs:
        model_path = os.path.join(models_dir, filename)

        if not os.path.exists(model_path):
            print(f"\n⚠️  No encontrado: {filename}")
            continue

        print(f"\n📊 Evaluando {name}...")

        env = WarehouseEnv(
            just_pick=just_pick, random_objects=random_obj, render_mode=None
        )
        agent = load_agent(model_path, env)

        results = evaluate_detailed(agent, env, num_episodes=500)

        success_rate = np.mean(results["success"]) * 100
        print(f"   Success rate: {success_rate:.1f}% (objetivo: ≥{target}%)")

        all_results.append(results)
        names.append(name)
        targets.append(target)

        env.close()

    if len(all_results) == 0:
        print("\n❌ No se encontraron modelos para evaluar")
        return

    # Generar gráficos
    print(f"\n{'='*70}")
    print("GENERANDO GRÁFICOS")
    print("=" * 70)

    plot_reward_distribution(
        all_results,
        names,
        os.path.join(outputs_dir, "grafico_distribucion_rewards.png"),
    )

    plot_success_pie_charts(
        all_results, names, os.path.join(outputs_dir, "grafico_resultados_pie.png")
    )

    plot_steps_histogram(
        all_results, names, os.path.join(outputs_dir, "grafico_histograma_pasos.png")
    )

    plot_comparison_bars(
        all_results,
        names,
        targets,
        os.path.join(outputs_dir, "grafico_comparacion_objetivos.png"),
    )

    plot_reward_boxplot(
        all_results, names, os.path.join(outputs_dir, "grafico_boxplot_rewards.png")
    )

    plot_cumulative_success(
        all_results, names, os.path.join(outputs_dir, "grafico_success_acumulado.png")
    )

    plot_summary_table(
        all_results,
        names,
        targets,
        os.path.join(outputs_dir, "grafico_tabla_resumen.png"),
    )

    # Resumen en consola
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print("=" * 70)

    all_passed = True
    for name, results, target in zip(names, all_results, targets):
        success = np.mean(results["success"]) * 100
        collision = np.mean(results["collision"]) * 100
        passed = success >= target
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
        print(f"{name}: {success:.1f}% éxito | {collision:.1f}% colisión | {status}")

    print("=" * 70)
    if all_passed:
        print("🎉 ¡TODOS LOS ENTORNOS CUMPLEN LOS OBJETIVOS!")
    else:
        print("⚠️  Algunos entornos no cumplen los objetivos")
    print("=" * 70)

    print(f"\n📁 Gráficos guardados en: {outputs_dir}")


if __name__ == "__main__":
    main()
