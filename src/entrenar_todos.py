"""
Entrenamiento SECUENCIAL de los 3 entornos
==========================================
Ejecuta el entrenamiento de los 3 entornos en orden:
1. Entorno 1 (base)
2. Entorno 2 (con transfer learning desde E1)
3. Entorno 3 (con transfer learning desde E2)

Uso: python entrenar_todos.py
"""
import os
import sys
from datetime import datetime

# Añadir directorio src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 70)
    print("ENTRENAMIENTO SECUENCIAL - TODOS LOS ENTORNOS")
    print("=" * 70)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nEste script entrenará los 3 entornos en orden:")
    print("  1. Entorno 1: Objetos fijos, solo recogida (objetivo ≥95%)")
    print("  2. Entorno 2: Objetos fijos, recogida y entrega (objetivo ≥90%)")
    print("  3. Entorno 3: Objetos aleatorios, recogida y entrega (objetivo ≥85%)")
    print("\nCada entorno usa transfer learning del anterior.")
    print("Los modelos se guardan en ../models/")
    print("=" * 70)
    
    results = {}
    
    # Entorno 1
    print("\n" + "🔵" * 35)
    print("FASE 1: ENTRENANDO ENTORNO 1")
    print("🔵" * 35)
    try:
        from entrenar_entorno1 import main as train_e1
        results['entorno1'] = train_e1()
    except Exception as e:
        print(f"❌ Error en Entorno 1: {e}")
        results['entorno1'] = None
    
    # Entorno 2
    print("\n" + "🟢" * 35)
    print("FASE 2: ENTRENANDO ENTORNO 2")
    print("🟢" * 35)
    try:
        from entrenar_entorno2 import main as train_e2
        results['entorno2'] = train_e2()
    except Exception as e:
        print(f"❌ Error en Entorno 2: {e}")
        results['entorno2'] = None
    
    # Entorno 3
    print("\n" + "🔴" * 35)
    print("FASE 3: ENTRENANDO ENTORNO 3")
    print("🔴" * 35)
    try:
        from entrenar_entorno3 import main as train_e3
        results['entorno3'] = train_e3()
    except Exception as e:
        print(f"❌ Error en Entorno 3: {e}")
        results['entorno3'] = None
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL - TODOS LOS ENTORNOS")
    print("=" * 70)
    
    targets = {'entorno1': 95, 'entorno2': 90, 'entorno3': 85}
    all_passed = True
    
    for name, result in results.items():
        target = targets[name]
        if result is not None:
            success = result['success_rate']
            passed = success >= target
            status = '✅ CUMPLE' if passed else '❌ NO CUMPLE'
            if not passed:
                all_passed = False
            print(f"{name.upper()}: {success:.1f}% (objetivo ≥{target}%) - {status}")
        else:
            print(f"{name.upper()}: ❌ ERROR EN ENTRENAMIENTO")
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ¡TODOS LOS ENTORNOS CUMPLEN LOS REQUISITOS!")
    else:
        print("\n⚠️ Algunos entornos no cumplen los requisitos.")
        print("   Considera ajustar hiperparámetros o entrenar más episodios.")
    
    print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
