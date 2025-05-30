import numpy as np

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2)) 