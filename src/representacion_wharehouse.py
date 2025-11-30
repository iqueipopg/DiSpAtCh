"""
Representación de estado para WarehouseEnv
==========================================

CAMBIO PRINCIPAL: Usamos features DENSAS en lugar de tile coding sparse.

El tile coding sparse (12,808 features con mayoría 0s) funciona bien con
métodos lineales (SARSA, Q-learning tabular/lineal), pero NO funciona bien
con redes neuronales profundas porque:
1. La mayoría de pesos nunca se actualizan
2. Gradientes muy dispersos
3. Difícil generalización

Las features densas (32 valores normalizados) permiten:
1. Todos los pesos se actualizan
2. Mejor propagación de gradientes
3. Generalización natural por proximidad en el espacio de features
"""

import numpy as np


class WarehouseFeedback:
    """
    Extractor de features densas para el entorno Warehouse.

    Genera un vector de 32 features normalizadas que captura:
    - Posición del agente
    - Posición relativa a objetos
    - Posición relativa al área de entrega
    - Estado actual (tiene objeto, cerca de objetivo, etc.)
    """

    def __init__(
        self, dims, ntiles=None, ntilings=None, delivery_area=(2.5, 9, 5.0, 2.0)
    ):
        """
        Args:
            dims: (width, height) del entorno
            ntiles, ntilings: Se ignoran (compatibilidad con código anterior)
            delivery_area: (x, y, width, height) del área de entrega
        """
        self.width, self.height = dims
        self.delivery_area = delivery_area

        # Centro del área de entrega
        self.delivery_center = (
            delivery_area[0] + delivery_area[2] / 2,
            delivery_area[1] + delivery_area[3] / 2,
        )

        # Posiciones fijas de los objetos (para Entorno 1 y 2)
        self.fixed_object_positions = [(2, 3.0), (8, 4.0), (5, 2.0)]

        # Estanterías (obstáculos)
        self.shelves = [
            (1.9, 1.0, 0.2, 5.0),
            (4.9, 1.0, 0.2, 5.0),
            (7.9, 1.0, 0.2, 5.0),
        ]

        # Feature size: 32 features densas
        self.feature_size = 32

        # Máxima distancia diagonal para normalización
        self.max_dist = np.sqrt(self.width**2 + self.height**2)

        print(f"📊 WarehouseFeedback initialized (DENSE features):")
        print(f"   Feature size: {self.feature_size}")
        print(f"   Delivery center: {self.delivery_center}")

    def process_observation(self, obs):
        """
        Procesa la observación del entorno y devuelve features densas.

        Args:
            obs: numpy array de 11 elementos del WarehouseEnv
                obs[0:2]: posición agente (x, y)
                obs[2:4]: posición objeto 1 (x, y) o [10, 10] si no existe
                obs[4:6]: posición objeto 2 (x, y) o [10, 10] si no existe
                obs[6:8]: posición objeto 3 (x, y) o [10, 10] si no existe
                obs[8]: has_object (0 o 1)
                obs[9]: collision (0 o 1)
                obs[10]: delivery (0 o 1)

        Returns:
            numpy array de 32 features normalizadas
        """
        features = np.zeros(self.feature_size, dtype=np.float32)

        # === POSICIÓN DEL AGENTE ===
        agent_x, agent_y = obs[0], obs[1]

        # [0-1] Posición normalizada del agente
        features[0] = agent_x / self.width
        features[1] = agent_y / self.height

        # === OBJETOS ===
        objects = []
        for i in range(3):
            obj_x, obj_y = obs[2 + 2 * i], obs[3 + 2 * i]
            if obj_x < 9.9 and obj_y < 9.9:  # Objeto existe
                objects.append((obj_x, obj_y))
            else:
                objects.append(None)

        # Encontrar objeto más cercano
        min_dist = float("inf")
        nearest_obj = None
        nearest_idx = -1

        for i, obj in enumerate(objects):
            if obj is not None:
                dist = self._distance(agent_x, agent_y, obj[0], obj[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_obj = obj
                    nearest_idx = i

        # [2-7] Features por cada objeto
        for i, obj in enumerate(objects):
            base_idx = 2 + i * 2
            if obj is not None:
                dx = (obj[0] - agent_x) / self.width
                dy = (obj[1] - agent_y) / self.height
                features[base_idx] = dx
                features[base_idx + 1] = dy
            else:
                # Objeto no existe - valor neutro
                features[base_idx] = 0.0
                features[base_idx + 1] = 0.0

        # [8-11] Features del objeto más cercano
        if nearest_obj is not None:
            dx = nearest_obj[0] - agent_x
            dy = nearest_obj[1] - agent_y
            dist = np.sqrt(dx**2 + dy**2)

            features[8] = dx / self.width  # Dirección X normalizada
            features[9] = dy / self.height  # Dirección Y normalizada
            features[10] = 1.0 - min(
                dist / self.max_dist, 1.0
            )  # Proximidad (1=cerca, 0=lejos)
            features[11] = np.arctan2(dy, dx) / np.pi  # Ángulo normalizado [-1, 1]
        else:
            features[8:12] = 0.0

        # [12-15] Features del área de entrega
        dx_del = self.delivery_center[0] - agent_x
        dy_del = self.delivery_center[1] - agent_y
        dist_del = np.sqrt(dx_del**2 + dy_del**2)

        features[12] = dx_del / self.width
        features[13] = dy_del / self.height
        features[14] = 1.0 - min(dist_del / self.max_dist, 1.0)  # Proximidad
        features[15] = np.arctan2(dy_del, dx_del) / np.pi  # Ángulo

        # [16-19] Flags de estado
        features[16] = float(obs[8])  # has_object
        features[17] = (
            float(min_dist < 0.8) if nearest_obj else 0.0
        )  # near_object (pickup distance)
        features[18] = float(
            self._is_in_delivery_area(agent_x, agent_y)
        )  # in_delivery_area
        features[19] = float(dist_del < 1.5)  # near_delivery

        # [20-23] Existencia de objetos (one-hot)
        features[20] = float(objects[0] is not None)
        features[21] = float(objects[1] is not None)
        features[22] = float(objects[2] is not None)
        features[23] = float(
            sum(1 for o in objects if o is not None)
        )  # Cuenta de objetos

        # [24-27] Distancia a paredes/bordes (útil para evitar colisiones)
        features[24] = agent_x / self.width  # Distancia a pared izquierda
        features[25] = (self.width - agent_x) / self.width  # Distancia a pared derecha
        features[26] = agent_y / self.height  # Distancia a pared inferior
        features[27] = (
            self.height - agent_y
        ) / self.height  # Distancia a pared superior

        # [28-31] Proximidad a estanterías (obstáculos)
        min_shelf_dist = float("inf")
        for shelf in self.shelves:
            # Distancia al centro de la estantería
            shelf_center_x = shelf[0] + shelf[2] / 2
            shelf_center_y = shelf[1] + shelf[3] / 2
            dist = self._distance(agent_x, agent_y, shelf_center_x, shelf_center_y)
            min_shelf_dist = min(min_shelf_dist, dist)

        features[28] = 1.0 - min(
            min_shelf_dist / 3.0, 1.0
        )  # Peligro de colisión (1=muy cerca)

        # Dirección segura (hacia área abierta)
        # Calcular gradiente hacia zona central
        center_x, center_y = self.width / 2, self.height / 2
        features[29] = (center_x - agent_x) / self.width
        features[30] = (center_y - agent_y) / self.height

        # Feature compuesta: ¿debería ir a objeto o a delivery?
        if obs[8] > 0.5:  # Tiene objeto -> ir a delivery
            features[31] = 1.0
        else:  # No tiene objeto -> ir a objeto más cercano
            features[31] = -1.0

        return features

    def _distance(self, x1, y1, x2, y2):
        """Distancia euclidiana entre dos puntos."""
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _is_in_delivery_area(self, x, y):
        """Verifica si la posición está dentro del área de entrega."""
        return (
            self.delivery_area[0] <= x <= self.delivery_area[0] + self.delivery_area[2]
            and self.delivery_area[1]
            <= y
            <= self.delivery_area[1] + self.delivery_area[3]
        )


# =============================================================================
# TEST DEL MÓDULO
# =============================================================================
if __name__ == "__main__":
    print("🧪 Testing WarehouseFeedback (Dense Features)...")

    # Crear feedback
    feedback = WarehouseFeedback(dims=(10.0, 10.0), delivery_area=(2.5, 9, 5.0, 2.0))

    # Test 1: Observación con 3 objetos
    print("\n--- Test 1: All objects present ---")
    obs1 = np.array(
        [
            5.0,
            5.0,  # Agente en (5, 5)
            2.0,
            3.0,  # Objeto 1
            8.0,
            4.0,  # Objeto 2
            5.0,
            2.0,  # Objeto 3
            0.0,  # No tiene objeto
            0.0,  # No colisión
            0.0,  # No delivery
        ],
        dtype=np.float32,
    )

    features1 = feedback.process_observation(obs1)
    print(f"Feature shape: {features1.shape}")
    print(f"Feature range: [{features1.min():.3f}, {features1.max():.3f}]")
    print(f"Sample features: {features1[:8]}")

    # Test 2: Agente llevando objeto cerca de delivery
    print("\n--- Test 2: Carrying object near delivery ---")
    obs2 = np.array(
        [
            5.0,
            9.5,  # Agente cerca de delivery
            10.0,
            10.0,  # Objeto 1 recogido
            10.0,
            10.0,  # Objeto 2 recogido
            10.0,
            10.0,  # Objeto 3 recogido
            1.0,  # Tiene objeto
            0.0,  # No colisión
            0.0,  # No delivery aún
        ],
        dtype=np.float32,
    )

    features2 = feedback.process_observation(obs2)
    print(f"has_object flag: {features2[16]}")
    print(f"in_delivery_area: {features2[18]}")
    print(f"near_delivery: {features2[19]}")
    print(f"goal_direction: {features2[31]}")  # Debería ser 1.0 (ir a delivery)

    # Test 3: Verificar que features son diferentes
    print("\n--- Test 3: Feature differentiation ---")
    similarity = np.dot(features1, features2) / (
        np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-8
    )
    print(f"Cosine similarity: {similarity:.3f}")
    print(
        "✅ Features are distinct"
        if abs(similarity) < 0.9
        else "⚠️ Features too similar"
    )

    # Test 4: Features para posiciones cercanas (generalización)
    print("\n--- Test 4: Generalization test ---")
    obs_a = np.array(
        [5.0, 5.0, 2.0, 3.0, 8.0, 4.0, 5.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    obs_b = np.array(
        [5.1, 5.1, 2.0, 3.0, 8.0, 4.0, 5.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32
    )

    features_a = feedback.process_observation(obs_a)
    features_b = feedback.process_observation(obs_b)

    diff = np.linalg.norm(features_a - features_b)
    print(f"Feature difference for nearby positions: {diff:.4f}")
    print("✅ Smooth generalization" if diff < 0.1 else "⚠️ Check features")

    print("\n✅ All tests completed!")
