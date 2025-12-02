"""
Representación del estado para los entornos de almacén
Implementa ingeniería de características para convertir la observación raw 
en features útiles para el aprendizaje
"""
import numpy as np


class WarehouseFeedback:
    """
    Procesa la observación del entorno de almacén y extrae características relevantes
    Genera 23 features optimizadas para el aprendizaje
    """
    def __init__(self, env_width=10.0, env_height=10.0):
        self.width = env_width
        self.height = env_height
        
        # Normalización para que todas las features estén en rangos similares
        self.position_scale = 10.0
        self.distance_scale = 14.14  # Diagonal máxima del entorno
        
        # Estanterías (x, y, width, height)
        self.shelves = [(1.9, 1.0, 0.2, 5.0), (4.9, 1.0, 0.2, 5.0), (7.9, 1.0, 0.2, 5.0)]
        self.agent_radius = 0.2
        
    def process_observation(self, obs):
        """
        Procesa la observación raw del entorno y devuelve características engineered
        
        Args:
            obs: Observación del entorno con 11 dimensiones:
                obs[0:2] = posición agente (x, y)
                obs[2:8] = posiciones objetos (3 objetos x 2 coords)
                obs[8] = agent_has_object
                obs[9] = collision
                obs[10] = delivery
                
        Returns:
            features: Array con 23 características procesadas
        """
        agent_pos = obs[0:2]
        obj1_pos = obs[2:4]
        obj2_pos = obs[4:6]
        obj3_pos = obs[6:8]
        has_object = obs[8]
        
        features = []
        
        # 1. Posición normalizada del agente (2 features)
        features.extend(agent_pos / self.position_scale)
        
        # 2. Flag si tiene objeto (1 feature)
        features.append(has_object)
        
        # 3. Distancias a cada objeto normalizadas (3 features)
        dist_obj1 = self._distance(agent_pos, obj1_pos) / self.distance_scale
        dist_obj2 = self._distance(agent_pos, obj2_pos) / self.distance_scale
        dist_obj3 = self._distance(agent_pos, obj3_pos) / self.distance_scale
        features.extend([dist_obj1, dist_obj2, dist_obj3])
        
        # 4. Distancia al objeto más cercano (1 feature)
        min_dist = min(dist_obj1, dist_obj2, dist_obj3)
        features.append(min_dist)
        
        # 5. Dirección al objeto más cercano - vector unitario (2 features)
        if dist_obj1 <= dist_obj2 and dist_obj1 <= dist_obj3:
            closest_obj = obj1_pos
        elif dist_obj2 <= dist_obj1 and dist_obj2 <= dist_obj3:
            closest_obj = obj2_pos
        else:
            closest_obj = obj3_pos
            
        direction = self._unit_vector(agent_pos, closest_obj)
        features.extend(direction)
        
        # 6. Distancia a zona de entrega normalizada (1 feature)
        delivery_center = np.array([5.0, 9.5])  # Centro aproximado de zona de entrega
        dist_delivery = self._distance(agent_pos, delivery_center) / self.distance_scale
        features.append(dist_delivery)
        
        # 7. Dirección a zona de entrega - vector unitario (2 features)
        direction_delivery = self._unit_vector(agent_pos, delivery_center)
        features.extend(direction_delivery)
        
        # 8. Posiciones relativas de objetos (6 features)
        if has_object == 0:
            # Posiciones relativas de los 3 objetos respecto al agente
            rel_obj1 = (obj1_pos - agent_pos) / self.position_scale
            rel_obj2 = (obj2_pos - agent_pos) / self.position_scale
            rel_obj3 = (obj3_pos - agent_pos) / self.position_scale
            features.extend(rel_obj1)
            features.extend(rel_obj2)
            features.extend(rel_obj3)
        else:
            # Si tiene objeto, posición relativa de zona de entrega
            rel_delivery = (delivery_center - agent_pos) / self.position_scale
            features.extend(rel_delivery)
            features.extend([0, 0, 0, 0])  # Padding
        
        # 9. Proximidad a obstáculos en cada dirección - 4 direcciones (4 features)
        # Esto ayuda al agente a evitar colisiones
        obstacle_proximity = self._get_obstacle_proximity(agent_pos)
        features.extend(obstacle_proximity)
        
        # 10. Flag binario si está cerca de poder recoger algún objeto (1 feature)
        can_pick = 1.0 if min_dist * self.distance_scale < 0.8 else 0.0
        features.append(can_pick)
        
        return np.array(features, dtype=np.float32)
    
    def _get_obstacle_proximity(self, agent_pos):
        """
        Calcula la proximidad a obstáculos en las 4 direcciones cardinales.
        Retorna valores entre 0 (lejos) y 1 (muy cerca/colisión inminente)
        """
        x, y = agent_pos
        velocity = 0.25  # Velocidad del agente
        
        # Distancia de detección
        detect_dist = 1.0
        
        proximities = []
        
        # Arriba (action 0)
        dist_up = self._min_obstacle_distance(x, y, 0, velocity, detect_dist)
        proximities.append(1.0 - min(dist_up / detect_dist, 1.0))
        
        # Abajo (action 1)
        dist_down = self._min_obstacle_distance(x, y, 1, velocity, detect_dist)
        proximities.append(1.0 - min(dist_down / detect_dist, 1.0))
        
        # Izquierda (action 2)
        dist_left = self._min_obstacle_distance(x, y, 2, velocity, detect_dist)
        proximities.append(1.0 - min(dist_left / detect_dist, 1.0))
        
        # Derecha (action 3)
        dist_right = self._min_obstacle_distance(x, y, 3, velocity, detect_dist)
        proximities.append(1.0 - min(dist_right / detect_dist, 1.0))
        
        return proximities
    
    def _min_obstacle_distance(self, x, y, direction, velocity, max_dist):
        """Calcula distancia mínima a obstáculo en una dirección"""
        min_dist = max_dist
        
        # Simular movimiento en esa dirección
        for step in range(1, int(max_dist / velocity) + 1):
            check_dist = step * velocity
            if direction == 0:  # Arriba
                new_x, new_y = x, y + check_dist
            elif direction == 1:  # Abajo
                new_x, new_y = x, y - check_dist
            elif direction == 2:  # Izquierda
                new_x, new_y = x - check_dist, y
            else:  # Derecha
                new_x, new_y = x + check_dist, y
            
            if self._would_collide(new_x, new_y):
                min_dist = check_dist
                break
        
        return min_dist
    
    def _would_collide(self, x, y):
        """Verifica si una posición causaría colisión"""
        # Paredes
        if x <= self.agent_radius or x >= self.width - self.agent_radius:
            return True
        if y <= self.agent_radius or y >= self.height - self.agent_radius:
            return True
        
        # Estanterías
        for shelf in self.shelves:
            sx, sy, sw, sh = shelf
            if (sx - self.agent_radius <= x <= sx + sw + self.agent_radius and
                sy - self.agent_radius <= y <= sy + sh + self.agent_radius):
                return True
        
        return False
    
    @staticmethod
    def _distance(pos1, pos2):
        """Calcula distancia euclidiana"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    @staticmethod
    def _unit_vector(from_pos, to_pos):
        """Calcula vector unitario de from_pos hacia to_pos"""
        vec = np.array(to_pos) - np.array(from_pos)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.array([0.0, 0.0])
        return vec / norm
    
    def get_feature_size(self):
        """Retorna el tamaño del vector de features"""
        # 2 (pos) + 1 (has_obj) + 3 (dists) + 1 (min_dist) + 2 (dir_obj) + 
        # 1 (dist_delivery) + 2 (dir_delivery) + 6 (rel_positions) +
        # 4 (obstacle_proximity) + 1 (can_pick)
        return 23


class WarehouseFeedbackSimple:
    """
    Versión simplificada que usa directamente la observación raw normalizada
    Útil para algoritmos que pueden aprender directamente de las observaciones
    """
    def __init__(self):
        pass
    
    def process_observation(self, obs):
        """
        Normaliza la observación raw
        
        Args:
            obs: Observación del entorno (11 dimensiones)
            
        Returns:
            obs_normalized: Observación normalizada
        """
        obs_normalized = obs.copy()
        
        # Normalizar posiciones (0-10 -> 0-1)
        obs_normalized[0:8] = obs_normalized[0:8] / 10.0
        
        # Las flags ya están en [0, 1]
        # obs[8], obs[9], obs[10] no necesitan normalización
        
        return obs_normalized
    
    def get_feature_size(self):
        """Retorna el tamaño del vector de features"""
        return 11


if __name__ == "__main__":
    # Test de la representación
    print("=" * 60)
    print("PRUEBA DE REPRESENTACIÓN DE ALMACÉN")
    print("=" * 60)
    
    # Crear observación de prueba
    obs = np.array([
        5.0, 5.0,      # Posición agente
        2.0, 3.0,      # Objeto 1
        8.0, 4.0,      # Objeto 2
        5.0, 2.0,      # Objeto 3
        0.0,           # No tiene objeto
        0.0,           # No hay colisión
        0.0            # No hay entrega
    ], dtype=np.float32)
    
    print("\nObservación raw:")
    print(obs)
    
    # Probar representación compleja
    feedback_complex = WarehouseFeedback()
    features = feedback_complex.process_observation(obs)
    print(f"\nFeatures procesadas (complejas): {len(features)} dimensiones")
    print(features)
    
    # Probar representación simple
    feedback_simple = WarehouseFeedbackSimple()
    features_simple = feedback_simple.process_observation(obs)
    print(f"\nFeatures procesadas (simples): {len(features_simple)} dimensiones")
    print(features_simple)
    
    print("\n✅ Representación funcionando correctamente")
