	import numpy as np
from sklearn.metrics import pairwise_distances

def phi(curvature,u_vect):
    if curvature > 0:
        r = (2/np.sqrt(curvature)) * np.arcsin(np.sqrt(u_vect) * np.sin(np.sqrt(curvature)/2))
    if curvature == 0:
        r = np.sqrt(u_vect)
    if curvature < 0:
        r = (2/np.sqrt(-curvature)) * np.arcsinh(np.sqrt(u_vect) * np.sinh(np.sqrt(-curvature)/2))
    return r    
        

def sample_uniformly(curvature, n_points):
    theta = 2 * np.pi * np.random.random_sample((n_points,))
    r = phi(curvature, np.random.random_sample((n_points,)))
    return np.stack((r,theta), axis = -1)

def geodesic_distance(curvature, x1 , x2):
    
    if curvature > 0:
        R = 1/np.sqrt(curvature)
        v1 = np.array([R * np.sin(x1[0]/R) * np.cos(x1[1]), 
                       R * np.sin(x1[0]/R) * np.sin(x1[1]),
                      R * np.cos(x1[0]/R)])
        
        v2 = np.array([R * np.sin(x2[0]/R) * np.cos(x2[1]), 
                       R * np.sin(x2[0]/R) * np.sin(x2[1]),
                      R * np.cos(x2[0]/R)])

        
        dist = R * np.arctan2(np.linalg.norm(np.cross(v1,v2)), (v1*v2).sum())
    
    if curvature == 0:
        v1 = np.array([x1[0]*np.cos(x1[1]), x1[0]*np.sin(x1[1])])
        v2 = np.array([x2[0]*np.cos(x2[1]), x2[0]*np.sin(x2[1])])
        dist = np.linalg.norm( (v1 - v2) )  
    
    if curvature < 0:
        R = 1/np.sqrt(-curvature)
        z = np.array([ np.tanh(x1[0]/(2 * R)) * np.cos(x1[1]),
                       np.tanh(x1[0]/(2 * R)) * np.sin(x1[1])])
        w = np.array([np.tanh(x2[0]/(2 * R)) * np.cos(x2[1]),
                       np.tanh(x2[0]/(2 * R)) * np.sin(x2[1])])
        temp = np.linalg.norm([(z*w).sum() - 1, np.linalg.det([z,w]) + 1])
        dist = 2 * R * np.arctanh(np.linalg.norm(z - w)/temp) 
        
    return dist

def distance_matrix(curvature,n_points):
    metric = lambda x1, x2 : geodesic_distance(curvature, x1 , x2)
    samples = sample_uniformly(curvature, n_points)
    return pairwise_distances(samples, metric = metric)

