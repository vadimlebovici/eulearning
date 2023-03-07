import os
import numpy as np

from scipy.io 	import loadmat
from .utils		import compute_OR_curvature, compute_FR_curvature, compute_hks_signature, compute_laplacian_eigenvector, compute_closeness_centrality, compute_edge_betweenness

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

# Orbit5K dataset
def gen_orbit(n_pts, rho): # Generates one orbit of ORBIT5K
	X = np.zeros((n_pts, 2))
	X[0] = np.random.rand(2)
	for i in range(1, n_pts):
		x, y = X[i-1]
		x_new = (x+rho*y*(1-y))%1
		y_new = (y+rho*x_new*(1-x_new))%1
		X[i] = np.array([x_new, y_new])
	return X

def gen_orbit5K(rhos=[2.5, 3.5, 4.0, 4.1, 4.3], n_pts=1000, size_each=1000): # Generates the full ORBIT5K dataset
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
			name_filtrations	: list of str, name of filtrations. Available options are 'hks_time' for the heat kernel signature, 'ricci_alpha_iterations' for the Ollivier-Ricci curvature, 'forman' for the Forman-Ricci curvature, 'laplacian_ind' for the ind-th eigen vector of the graph laplacian, 'centrality' for the centrality function, 'betweenness' for the edge betweenness and 'func_ind' for the ind-th function pre-defined on the graphs of this specific dataset. For instance, 'hks_10.0', 'ricci_0.5_0', 'forman', 'laplacian_1', 'centrality', 'betweenness', 'func_0'.
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
			if 'laplacian' in filt:									# of the form 'laplacian_ind'
				ind = int(filt.split('_')[-1])
				filtrations.append(compute_laplacian_eigenvector(A, ind))
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