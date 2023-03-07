import os
import numpy 			as np
import numba 			as _nb
import sys 				as _sys
import tadasets

from scipy 				import special as _special
from scipy 				import optimize as _optimize
from scipy.io 			import loadmat
from .utils				import compute_OR_curvature, compute_FR_curvature, compute_hks_signature, compute_closeness_centrality, compute_edge_betweenness
from sklearn.metrics 	import pairwise_distances
from collections		import namedtuple as _namedtuple

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

# Graph datasets
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

def _extract_intrinsic_funcs_DHFR(): 
	'''
	Method extracting the functions defined on the nodes of DHFR and save them in a .txt file. 
	'''
	id_table = np.loadtxt('./data/DHFR/DHFR.graph_idx')
	funct = np.loadtxt('./data/DHFR/DHFR.node_attrs', delimiter=',')
	path_dataset = "./data/DHFR/"
	os.mkdir('./data/DHFR/func_0/')
	os.mkdir('./data/DHFR/func_1/')
	os.mkdir('./data/DHFR/func_2/')
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
			if 'func' in filt:										# of the form 'func_ind'. See demos/graphs.ipynb for an explanation on how to compute them.
				ind = int(filt.split('_')[-1])
				path_to_file = path_to_dataset + '/func_' + str(ind) + '/f_' + str(k)
				if not os.path.isfile(path_to_file):
					_extract_intrinsic_funcs_DHFR()
				func = np.loadtxt(path_to_file)
				filtrations.append(func)
		vec_sts.append(build_vectorized_st_from_adjacency_matrix(A, filtrations))
	return vec_sts, y

# Curvature datasets
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


# Toy example: Poisson and Ginibre point clouds
# To compute GPP and PPP, we need the following functions from the repository https://gitlab.inria.fr/gmoro/point_process/
#**********************************************************************#
#	Copyright (C) 2020 Guillaume Moroz <guillaume.moroz@inria.fr>	 #
#																	  #
# This program is free software: you can redistribute it and/or modify #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation, either version 2 of the License, or	#
# (at your option) any later version.								  #
#				  http://www.gnu.org/licenses/						#
#**********************************************************************#

@_nb.njit((_nb.complex128[::1], _nb.int64[::1], _nb.complex128[:,::1], _nb.float64[::1], _nb.int64, _nb.int64))
def _instantiate_polynomial(p, I, M, G, l, u):
	for i in range(l, u):
		for j in range(i, u):
			p[I[j]-I[i]] += M[i,j]*G[i]*G[j]

@_nb.njit((_nb.complex128[:,::1], _nb.complex128[::1], _nb.complex128[::1]))
def _fused_minus_outer(V, e, f):
	for i in range(V.shape[0]):
		for j in range(V.shape[1]):
			V[i,j] -= e[i]*f[j]

@_nb.guvectorize([(_nb.complex128[::1], _nb.complex128, _nb.complex128[::1], _nb.complex128[::1])],
				'(n),(),(n)->(n)')
def _fused_minus_outer_vec(v, c, f, res):
		for j in range(res.shape[0]):
			res[j] = v[j] - c*f[j]

def _V_minus_e_estar(V, e, epsilon):
		l, u = _argtruncate(e, epsilon)
		f = e.conjugate()
		if u-l > 0.9*len(e):
			_fused_minus_outer(V, e, f)
		else:
			_fused_minus_outer_vec(V[l:u, l:u], e[l:u], f[l:u], out=V[l:u, l:u])

@_nb.njit((_nb.complex128[::1], _nb.complex128))
def _horner(p, v):
	c = p[-1]
	for i in range(len(p)-2, -1, -1):
		c = c*v + p[i]
	return c


# Main Sampling functions
def sample_indices(kernel, R, epsilon=2**-53):
	I = []
	Lambda = 1
	i = 0
	R2 = R**2
	while Lambda > epsilon:
		Lambda = kernel.F(i, R2)
		if np.random.binomial(1, Lambda) == 1:
			I.append(i)
		i += 1
	return np.array(I, dtype='int64')

def sample_module(C, R, I, invLambdas, F, epsilon):
	i = np.random.multinomial(1, C).argmax()
	c = np.random.uniform()
	f = lambda r: F(I[i], r**2)*invLambdas[i] - c
	r = _optimize.brentq(f, 0, R, xtol=epsilon)
	return r

def sample_argument(V, r, I, invLambdas, g, epsilon):
	G = g(I,r)*np.sqrt(invLambdas)
	l, u = _argtruncate(G, epsilon)
	p = np.zeros(I[u-1] - I[l] + 1, dtype='complex128')
	_instantiate_polynomial(p, I, V, G, l, u)
	p[1:] /= 0.5*p[0]*1j*np.arange(1, p.size)
	p[0] = -np.sum(p[1:])
	n = np.arange(p.size)
	c = np.random.uniform()
	f = lambda alpha: alpha + np.real(_horner(p, np.exp(1j*alpha))) - c*2*np.pi
	alpha = _optimize.brentq(f, 0, 2*np.pi, xtol=epsilon)
	return alpha

def sample_points(kernel, R, I, epsilon=2**-53, print_point=lambda x,y,i:None):
	global points_list
	F, g = kernel
	n = len(I)
	W = np.zeros(n, dtype='complex128')
	U = np.ones(n, dtype='float64')
	V = np.identity(n, dtype='complex128')
	Lambdas = np.array([F(i, R**2) for i in I])
	invLambdas = 1/Lambdas
	for i in range(n, 0, -1):
		# Draw point Wi
		r = sample_module(U/U.sum(), R, I, invLambdas, F, epsilon)
		alpha = sample_argument(V, r, I, invLambdas, g, epsilon)
		p = r*np.exp(1j*alpha)
		W[n-i] = p
		px = p.real
		py = p.imag
		print_point(px, py, n-i)

		# Compute new vector ei
		phi = g(I,r)*np.exp(1j*alpha*I)*np.sqrt(invLambdas)
		l, u = _argtruncate(phi, epsilon)
		phi = V[:, l:u].dot(phi[l:u])
		e = phi/np.linalg.norm(phi)

		# Update arrays U and V
		U -= e.real**2 + e.imag**2
		U[U<0] = 0
		_V_minus_e_estar(V, e, epsilon)
	return V, W

# Kernels
# Use notations of the article : g(i,r) = sqrt(dF/dr(i,r**2))
# F(i,r) can be defined up to a function of i independant of r such that F(i, sup r) <= 1
# g(i,r) can be defined up to a function of r independant of i
# F and g should be callable with arrays as the first argument
Kernel = _namedtuple('Kernel', ['F','g'])

kernels = {
	# Ginibre point process
	'ginibre': Kernel(lambda i, r: _special.gammainc(i+1,r),
					  lambda i, r: np.where(i!=0, np.exp(i*np.log(r) - 0.5*(_special.gammaln(i+1) + r**2)),
												   np.exp(-0.5*r**2))),

	# Zeros of an analytic function with Gaussian coefficients
	'gaussian': Kernel(lambda i, r: np.power(r,i+1),
					   lambda i, r: np.power(r,i)*np.sqrt(i+1)),

	## Experimental kernels
	# Gaussian kernel times 1 - r**2
	'weighted': Kernel(lambda i, r: 3*(np.power(r,i+1) - 2*np.power(r, i+2)*(i+1)/(i+2) + np.power(r, i+3)*(i+1)/(i+3)),
					   lambda i, r: np.power(r,i)*np.sqrt(i+1)),

	# Uniform modules
	'pseudo-uniform': Kernel(lambda i, r: r,
							 lambda i, r: 1),
}

# Parser  for command line arguments
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('R', type=float, help="radius",
						default=1)
	parser.add_argument('-N', metavar='N', type=int,
						help="preset N points by truncating the kernel to the N first eigenfunctions",
						default=None)
	parser.add_argument('-k', '--kernel', metavar='kernel', type=str,
						help='kernel to sample : ginibre or gaussian',
						default='ginibre')
	parser.add_argument('-p', '--precision', metavar='prec  ', type=float,
						help="error tolerated for internal computations",
						default=2**-53)
	parser.add_argument('-s', '--size', metavar='size  ', type=float,
						help="points size in pixels",
						default=5)
	parser.add_argument('-t', '--time', metavar='time  ', type=int,
						help="refresh time in miliseconds",
						default=100)
	parser.add_argument('-o', '--output', metavar='output', type=str,
						help='name of file to output the data, implies --nogui',
						default=None)
	parser.add_argument('-e ', '--error', action='store_true',
						help="compute the error and the condition number for the result",
						default=False)
	parser.add_argument('-pg', '--profile', action='store_true',
						help="output time indicator some functions",
						default=False)
	parser.add_argument('-q ', '--quiet', action='store_true',
						help="disable information messages on standard output",
						default=False)
	parser.add_argument('--nogui', action='store_true',
						help="output points coordinate on the terminal",
						default=False)
	args = parser.parse_args()
	if not args.quiet:
		print("Importing libraries ...")

# Util functions compiled with numba

def _argtruncate(v, epsilon):
	vbig = abs(v) > epsilon
	l = np.argmax(vbig)
	u = len(vbig) - np.argmax(vbig[::-1])
	return l, u

# Qt interface functions
def _init_figure(R, size):
	global _scatter, _view, _spot, _app
	# Launch app
	pg.setConfigOptions(background = 'w', foreground = 'k')
	_app = pg.mkQApp()

	# Create the main view
	_view = pg.PlotWidget()
	_view.setRenderHint(pg.Qt.QtGui.QPainter.HighQualityAntialiasing)
	_view.resize(800, 600)
	_view.setRange(xRange=(-R,R), yRange=(-R,R))
	_view.setWindowTitle('Determinantal point process')
	_view.setTitle('Sampling the number of points ...')
	_view.setAspectLocked(True)
	_view.show()
		
	# Create the circle and add it to the view
	circle = pg.Qt.QtWidgets.QGraphicsEllipseItem()
	circle.setRect(-R, -R, 2*R, 2*R)
	circle.setPen(pg.mkPen(width=2, color='k'))
	_view.addItem(circle)
		
	# Create the scatter plot and add it to the view
	_scatter = pg.ScatterPlotItem(symbol='o')
	_scatter.setSize(size)
	_view.addItem(_scatter)

	# Spot
	_spot = np.empty(1, dtype=_scatter.data.dtype)
	_spot['pen'] = pg.mkPen(width=1, color='b')
	_spot['brush'] = pg.mkBrush(None)
	_spot['size'] = size
	_spot['visible'] = True
	if pg.Qt.QT_LIB not in ['PySide2', 'PySide6']:
		_spot['targetQRectValid'] = False
	_scatter.updateSpots(_spot)

	_app.processEvents()

def _update_figure():
	global _scatter, _view, npoints
	pad =  len(str(npoints))
	_view.setTitle('<pre>Sampling: {0: >{2}}/{1} points</pre>'.format(_scatter.data.size, npoints, pad))
	_scatter.prepareGeometryChange()
	_scatter.bounds = [None, None]

def _print_point_qt(px, py, i):
	_scatter.data.resize(i+1, refcheck=False)
	_scatter.data[i] = _spot
	if pg.Qt.QT_LIB not in ['PySide2', 'PySide6']:
		_scatter.data[i]['targetQRect'] = pg.Qt.QtCore.QRectF()
	_scatter.data[i]['x'] = px
	_scatter.data[i]['y'] = py
	_app.processEvents()

def qt_sample(R, N = None, kernel=kernels['ginibre'], precision=2**-53, size=5, refresh=100, error=False, quiet=False):
	global npoints, pg
	import pyqtgraph as pg
	if N is not None and kernel.F(N-1, R**2) == 0:
		raise ValueError("N is too big")
	_init_figure(R, size)
	if N is None:
		I = sample_indices(kernel, R, precision)
	else:
		I = np.arange(N)
	npoints = len(I)
	timer = pg.Qt.QtCore.QTimer()
	timer.timeout.connect(_update_figure)
	timer.start(refresh)
	V, W = sample_points(kernel, R, I, precision, _print_point_qt)
	timer.stop()
	_update_figure()
	_app.processEvents()
	if error:
		_view.setTitle('Computing the error and the condition number ...')
		_app.processEvents()
		Error = np.linalg.norm(V)
		tI = I.reshape(-1,1)
		M = kernel.g(tI, np.abs(W))*np.exp(1j*np.angle(W)*tI)/np.sqrt(kernel.F(tI, R**2))
		ConditionNumber = np.linalg.cond(M)
		_view.setTitle('<pre>Number of points: {0}		Error: {1:.3e}		Condition number: {2:.3e}</pre>'
					  .format(npoints, Error, ConditionNumber))
	else:
		_view.setTitle('<pre>Number of points: {0}</pre>'.format(npoints))
	Blue = pg.mkBrush('b')
	_scatter.setBrush([Blue]*len(_scatter.data))
	_app.exec_()

# Text interface functions
def _build_print_point(output, quiet, n):
	pad =  len(str(n))
	message = '\r{{0: >{0}}}/{1} '.format(pad, n)
	if output is None and quiet:
		print_point_txt = lambda x, y, i: None
	elif output is None and not quiet:
		print_point_txt = lambda x, y, i: _sys.stdout.write(message.format(i+1))
	elif output is not None and quiet:
		print_point_txt = lambda x, y, i: output.write("{0} {1}\n".format(x, y))
	else:
		print_point_txt = lambda x, y, i: _sys.stdout.write(message.format(i+1)) and output.write("{0} {1}\n".format(x, y))
	return print_point_txt

def sample(R, N = None, kernel=kernels['ginibre'], precision=2**-53, error=False, quiet=False, output=None):
	if N is None:
		if not quiet:
			print('Sampling the number of points ...')
		I = sample_indices(kernel, R, precision)
	else:
		if kernel.F(N-1, R**2) == 0:
			raise ValueError("N is too big")
		I = np.arange(N)
	print_point_txt = _build_print_point(output, quiet, len(I))
	if not quiet:
		print('Sampling the points ...')
	V, W = sample_points(kernel, R, I, precision, print_point_txt)
	if not quiet:
		print()
	if error:
		if not quiet:
			print('Computing the error and the condition number ...')
		Error = np.linalg.norm(V)
		tI = I.reshape(-1,1)
		M = kernel.g(tI, np.abs(W))*np.exp(1j*np.angle(W)*tI)/np.sqrt(kernel.F(tI, R**2))
		ConditionNumber = np.linalg.cond(M)
		if not quiet:
			print('Error: {0:.3e}'.format(Error))
			print('Condition number: {0:.3e}'.format(ConditionNumber))
		if output is not None:
			output.write('# Error: {0:.3e}\n'.format(Error))
			output.write('# Condition number: {0:.3e}\n'.format(ConditionNumber))
		return W, Error, ConditionNumber
	else:
		return W


## We may now compute the GPP and PPP
def gen_Ginibre(n_pts, radius): 
	'''
	Generates (approximately) n_pts of a Ginibre point process on the centered disk of a given radius.
	'''
	X = sample(np.sqrt(n_pts), kernel=kernels['ginibre'], quiet=True)
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