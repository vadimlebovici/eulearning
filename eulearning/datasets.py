import os
import tadasets
import dpp
import numpy 			as np
from data_gen_curvature import distance_matrix

from scipy.io 			import loadmat
from .utils				import compute_OR_curvature, compute_FR_curvature, compute_hks_signature, compute_laplacian_eigenvector, compute_closeness_centrality, compute_edge_betweenness
from sklearn.metrics 	import pairwise_distances

# Toy example: dense lines in background noise
def gen_dense_lines_with_noise(n_pattern=1, n_noise=500, n_signal=20, size_signal=10, size_noise=50):
	X_noise = np.random.uniform(low=-size_noise, high=size_noise, size=(n_noise, 2))
	X_signals = []
	for _ in range(n_pattern):
		X_signal = np.zeros((n_signal,2))
		X_signal[:,0] = (2*np.random.uniform()-1)*size_noise*np.ones(n_signal)
		X_signal[:,1] = (np.random.random(size=n_signal)-0.5)*size_signal
		X_signals.append(X_signal)
	X = np.concatenate([X_noise] + X_signals, axis=0)
	return X

# Toy example: uniform and non-uniform samples on torus and sphere
def gen_torus_unif(n_pts, noise, a, c): 
	'''
	Draw n_pts uniformly at random on a torus of radii chosen at uniformly random in the intervals (a[0], a[1]) and (c[0], c[1]) respectively. A uniform noise of size noise is added to the points.
	'''
	a = a[0]+(a[1]-a[0])*np.random.rand() 
	c = c[0]+(c[1]-c[0])*np.random.rand()
	X = np.zeros((n_pts, 3))
	n_filled = 0
	while n_filled <n_pts:
		theta = np.random.uniform(0, 2*np.pi)
		eta = np.random.uniform(0, 1/np.pi)
		fx = (1+(a/c)*np.cos(theta))/(2*np.pi)
		if eta < fx:
			phi = np.random.uniform(0, 2*np.pi)
			X[n_filled] = [(c+a*np.cos(theta))*np.cos(phi),(c+a*np.cos(theta))*np.sin(phi), a*np.sin(theta)]
			n_filled+=1
	return X + noise*np.random.randn(*X.shape)

def gen_torus_non_unif(n_pts, noise, a, c): #torus has random sizes drawn in a and c intervals
	'''
	Draw n_pts on a torus of radii chosen at uniformly random in the intervals (a[0], a[1]) and (c[0], c[1]) respectively. The points are drawn in toroidal coordinates with angles chosen uniformly at random. A uniform noise of size noise is added to the points.
	'''
	a = a[0]+(a[1]-a[0])*np.random.rand() 
	c = c[0]+(c[1]-c[0])*np.random.rand()
	theta = np.random.uniform(0, 2*np.pi, size = n_pts)
	phi = np.random.uniform(0,2*np.pi, size = n_pts)
	X = np.zeros((n_pts, 3))
	X[:,0] = (c+a*np.cos(theta))*np.cos(phi)
	X[:,1] = (c+a*np.cos(theta))*np.sin(phi)
	X[:,2] = a*np.sin(theta)
	return X + noise*np.random.randn(*X.shape)

def gen_sphere_unif(n_pts, noise, r): 
	'''
	Draw n_pts uniformly on a sphere of radius chosen uniformly at random in the interval (r[0], r[1]). A uniform noise of size noise is added to the points.
	'''
	r = r[0]+ (r[1]-r[0])*np.random.rand() 
	return tadasets.dsphere(n_pts,2,  r, noise)

def gen_sphere_non_unif(n_pts, noise, r): #sphere has random radius drawn in r interval
	'''
	Draw n_pts on a sphere of radius chosen uniformly at random in the interval (r[0], r[1]). In spherical coordinates, one angle is chosen uniformly and the other is drawn from a normal distribution centered on the midpoint of its values. A uniform noise of size noise is added to the points.
	'''
	r = r[0] + (r[1]-r[0])*np.random.rand() 
	theta = np.random.uniform(0, np.pi, size = n_pts)
	phi = np.random.normal(loc = np.pi, scale = 2, size = n_pts)%(2*np.pi)
	X=np.zeros((n_pts, 3))
	X[:,0] = r*np.sin(phi)*np.cos(theta)
	X[:,1] = r*np.sin(phi)*np.sin(theta)
	X[:,2] = r*np.cos(phi)
	return X + noise*np.random.randn(*X.shape)

# Toy example: Poisson and Ginibre point clouds
def gen_Ginibre(n_pts, radius): 
	'''
	Generates a Ginibre point process on the centered disk of a given radius.
	'''
	X = dpp.sample(n_pts, kernel=dpp.kernels['ginibre'], quiet=True)
	X_gin = np.zeros((len(X), 2))
	X_gin[:,0] = np.real(X)
	X_gin[:,1] = np.imag(X)
	return radius*X_gin/np.sqrt(n_pts)

def gen_Poisson(n_pts, radius): 
	'''
	Generates a Poisson point process on the centered disk of a given radius.
	'''
	R = radius*np.sqrt(np.random.rand(n_pts))
	theta = 2*np.pi*np.random.rand(n_pts)
	X = np.zeros((n_pts, 2))
	X[:,0] = R*np.cos(theta)
	X[:,1] = R*np.sin(theta)
	return X

# Orbit5K dataset
def gen_orbit(n_pts, rho):
	'''
	Generates one orbit of ORBIT5K.
	'''
	X = np.zeros((n_pts, 2))
	X[0] = np.random.rand(2)
	for i in range(1, n_pts):
		x, y = X[i-1]
		x_new = (x+rho*y*(1-y))%1
		y_new = (y+rho*x_new*(1-x_new))%1
		X[i] = np.array([x_new, y_new])
	return X

def gen_orbit5K(rhos=[2.5, 3.5, 4.0, 4.1, 4.3], n_pts=1000, size_each=1000):
	'''
	Generates the full ORBIT5K dataset.
	'''
	n_classes = len(rhos)
	X = np.zeros((n_classes*size_each, n_pts, 2))
	y = []
	for i, rho in enumerate(rhos):
		for k in range(size_each):
			X[i*size_each+k] = gen_orbit(n_pts, rho)
			y.append(i)
	return X, y

# Graphs dataset
def build_vectorized_st_from_adjacency_matrix(A, filtrations):
	'''
	Build vectorized simplex trees from the adjacency matrix of a graph with num_vertices vertices and num_edges edges.

	Input:
		A 			: ndarray of size num_vertices x num_vertices, adjacency matrix of a graph
		filtrations : list of filtrations, each is a ndarray of size (num_vertices,)
	Output:
		vec_st 		: ndarray of size (num_vertices+num_edges, 1+len(filtrations)) containing for each simplex splx: [(-1)*dim(splx), filt_1(splx), ..., filt_n(splx)] where filt_j is the upper star filtration induced by the function filtrations[j].
	'''
	(xs, ys) = np.where(np.triu(A))
	num_vertices, num_edges = A.shape[0], xs.shape[0]
	num_filts = len(filtrations)
	vec_st = np.zeros((num_vertices+num_edges,num_filts+1))
	# Setting the values of vertices
	for i in range(num_vertices):
		vec_st[i,0] = 1
		for idf, f in enumerate(filtrations):
			vec_st[i,1+idf] = f[i]
	# Setting the values of edges
	for idx, x in enumerate(xs):
		id_edge = num_vertices+idx
		vec_st[id_edge,0] = -1
		for idf, f in enumerate(filtrations):
			vec_st[id_edge,idf+1] = np.max(np.take(f, [x, ys[idx]])) # Compute the upper star filtration on vertices of the edge
	return vec_st

def _extract_intrinsic_funcs_DHFR(): # This function does not need to be run again. It is placed here for completeness, in order to be adapted on other graph dataset. 
	'''
	Method extracting the functions defined on the nodes of DHFR and save them in a .txt file. 
	'''
	id_table = np.loadtxt('./data/DHFR/DHFR.graph_idx')
	funct = np.loadtxt('./data/DHFR/DHFR.node_attrs', delimiter=',')
	path_dataset = "./data/DHFR/"
	for k, graph_name in enumerate(os.listdir(path_dataset + "mat/")):
		gid = int(graph_name.split('_')[5])
		X=np.where(id_table==gid)
		f=funct[X]
		np.savetxt('./data/DHFR/func_0/f_{}'.format(k), f[:,0])
		np.savetxt('./data/DHFR/func_1/f_{}'.format(k), f[:,1])
		np.savetxt('./data/DHFR/func_2/f_{}'.format(k), f[:,2])

def load_graph_dataset(dataset, path_to_dataset, name_filtrations):
	'''
	Load a graph dataset from https://networkrepository.com/. Dataset has to be in the 'mat' format, available for the datasets 'MUTAG', 'COX2', 'DHFR', 'PROTEINS', 'NCI1', 'NCI109','IMDB-BINARY' and 'IMDB-MULTI' on the Perslay repository https://github.com/MathieuCarriere/perslay.
	
	Input:
		dataset				: str, name of the dataset
		path_to_dataset		: str, path to the mat/ folder containing th .mat format of the graphs
		name_filtrations	: list of str, name of filtrations. Available options are 'hks_time' for the heat kernel signature, 'ricci_alpha_iterations' for the Ollivier-Ricci curvature, 'forman' for the Forman-Ricci curvature, 'centrality' for the centrality function, 'betweenness' for the edge betweenness and 'func_ind' for the ind-th function pre-defined on the graphs of this specific dataset. For instance, 'hks_10.0', 'ricci_0.5_0', 'forman', 'centrality', 'betweenness', 'func_0'.
	Output:
		vec_sts				: list of ndarray of size (num_vertices+num_edges, 1+len(filtrations)) as returned by build_vectorized_st_from_adjacency_matrix. For each simplex splx, it contains a line [(-1)*dim(splx), filt_1(splx), ..., filt_n(splx)] where filt_j is the upper star filtration induced by the j-th filtrations.
		y					: ndarray of size (N,) where N is the number of graphs in the dataset. Contains the labels of the graphs.
	'''
	vec_sts = []
	y = []
	for k, graph_name in enumerate(os.listdir(path_to_dataset + "mat/")):
		A = np.array(loadmat(path_to_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
		name = graph_name.split("_")

		label = int(name[name.index("lb") + 1])
		y.append(label)

		filtrations = []
		for filt in name_filtrations:
			if 'hks' in filt: 										# of the form 'hks_time'
				time = float(filt.split('_')[-1])
				filtrations.append(compute_hks_signature(A, time))
			if 'ricci' in filt: 									# of the form 'ricci_alpha_iterations'
				splt_name = filt.split('_')
				alpha = float(splt_name[1])
				iterations = int(splt_name[-1])
				filtrations.append(compute_OR_curvature(A, alpha, iterations))
			if 'forman' in filt: 									# of the form 'forman'
				filtrations.append(compute_FR_curvature(A))
			if 'centrality' in filt:
				filtrations.append(compute_closeness_centrality(A))
			if 'betweenness' in filt:
				filtrations.append(compute_edge_betweenness(A))
			if 'func' in filt:										# of the form 'func_ind'
				ind = int(filt.split('_')[-1])
				func = np.loadtxt(path_to_dataset + '/func_' + str(ind) + '/f_' + str(k))
				filtrations.append(func)
		vec_sts.append(build_vectorized_st_from_adjacency_matrix(A, filtrations))
	return vec_sts, y

# Curvature datasets.

def modulus(curvature, u_vect): 
	'''
	Computes the radius of a point in polar coordinates for an arbitrary center, CDF inversion method.
	'''
	if curvature > 0:
		r = (2/np.sqrt(curvature)) * np.arcsin(np.sqrt(u_vect) * np.sin(np.sqrt(curvature)/2))
	if curvature == 0:
		r = np.sqrt(u_vect)
	if curvature < 0:
		r = (2/np.sqrt(-curvature)) * np.arcsinh(np.sqrt(u_vect) * np.sinh(np.sqrt(-curvature)/2))
	return r	
		

def sample_uniformly(curvature, n_pts):
	'''
	Generates n points in polar coordinates on the unit disk of the sphere/Euclidean space/hyperbolic space depending on whether K>0, K==0 or K<0.
	'''
	theta = 2 * np.pi * np.random.random_sample((n_pts,))
	r = modulus(curvature, np.random.random_sample((n_pts,)))
	return np.stack((r,theta), axis = -1)

def geodesic_distance(curvature, x1 , x2): 
	'''
	Computes the geodesic distance between two points on a space of a given curvature.
	'''
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

def distance_matrix(curvature, n_pts): 
	'''
	Samples n_pts uniformly on a space of a given curvature and returns their distance matrix for the geodesic distance.
	'''
	metric = lambda x1, x2 : geodesic_distance(curvature, x1 , x2)
	samples = sample_uniformly(curvature, n_pts)
	return pairwise_distances(samples, metric = metric)

def gen_curvature_pt_cld(n_pts, K):
	'''
	Generates distance matrix of n_pts uniformly drawn on a space of constant curvature K.
	'''
	return distance_matrix(K, n_pts)
