import numpy as np

from eulearning.utils 	import cartesian_product
from sklearn.base 		import BaseEstimator, TransformerMixin
from sklearn.utils 		import assert_all_finite


def compute_euler_curve(vec_st, bin_size): 
	# This version is faster in practice when computing Euler curves. It works similarly as compute_euler_profile, documented below. 
	assert_all_finite(vec_st)
	assert_all_finite(bin_size)

	t_min, t_max, resolution = bin_size
	scale = (resolution-1)/(t_max-t_min)
	ecc = np.zeros(resolution)
	n_splx = vec_st.shape[0]
	for i in range(n_splx):
		ind = max(np.ceil((vec_st[i,1]-t_min)*scale).astype(int), 0)
		if ind<resolution:
			ecc[ind]+= vec_st[i,0]
	return np.cumsum(ecc)

def compute_euler_profile(vec_st, bin_sizes):
	'''
	Input:
		vec_st 		: ndarray containing a vectorized simplex_tree as returned by get_vectorized_st
		bin_sizes	: list of the form [(low_1, high_1, num_1), ..., (low_k, high_k, num_k)] for a k-parameter Euler characteristic profile
	Output:
		ecp			: ndarray of size (num_1, ..., num_k) containing the k-parameter Euler characteristic profile associated to vec_st uniformly sampled on the grid [low_1,high_1] x ... x [low_k,high_k]
	'''
	assert_all_finite(vec_st)
	assert_all_finite(bin_sizes)

	num_filts = len(bin_sizes)
	if num_filts==1:
		ecp = compute_euler_curve(vec_st, bin_sizes[0])
	else:
		t_mins = np.array([size[0] for size in bin_sizes])
		t_maxs = np.array([size[1] for size in bin_sizes])
		resolutions = np.array([size[-1] for size in bin_sizes]).astype(int)
		scales = (resolutions-1)/(t_maxs-t_mins)
		ecp = np.zeros(resolutions)
		n_splx = vec_st.shape[0]
		for i in range(n_splx):
			sgn, filtrations = vec_st[i,0], vec_st[i,1:]
			inds = np.maximum(np.ceil((filtrations-t_mins)*scales),0).astype(int)
			if (inds<resolutions).all():
				ecp[tuple(inds)] += sgn
		for k in range(num_filts):
			ecp = np.cumsum(ecp, axis = k)
		ecp = ecp.transpose()
	return ecp

def compute_hybrid_transform(vec_st, bin_sizes, kernel):
	'''
	Input:
		vec_st 		: ndarray containing a vectorized simplex_tree as returned by get_vectorized_st
		bin_sizes	: list of the form [(low_1, high_1, num_1), ..., (low_k, high_k, num_k)] for a k-parameter hybrid transform
		kernel		: function, primitive kernel of the transform
	Output:
		ht			: ndarray of size (num_1, ..., num_k) containing the k-parameter hybrid transform of the Euler characteristic profile associated to vec_st. The transform is uniformly sampled on the grid [low_1,high_1] x ... x [low_k,high_k].
	'''
	assert_all_finite(vec_st)
	assert_all_finite(bin_sizes)

	ht_range = cartesian_product(*[np.linspace(*bin_size) for bin_size in bin_sizes])
	sgns = vec_st[:,0]
	values = vec_st[:, 1:]
	a = np.dot(ht_range, values.transpose()) 	
	ka = kernel(a)								
	ska = sgns * ka								
	ht = - np.sum(ska, axis=-1)
	ht = ht.reshape(*[bin_size[-1] for bin_size in bin_sizes]).transpose()
	return ht

def compute_radon_transform(vec_st, bin_sizes):
	'''
	Input:
		vec_st 		: ndarray containing a vectorized simplex_tree as returned by get_vectorized_st
		bin_sizes	: list of the form [(low_1, high_1, num_1), ..., (low_k, high_k, num_k)] for a k-parameter hybrid transform
	Output:
		rdn			: ndarray of size (num_1, ..., num_k) containing the k-parameter Radon transform of the Euler characteristic profile associated to vec_st. The transform is uniformly sampled on the grid [low_1,high_1] x ... x [low_k,high_k] as explained in the article (Section 3.1).
	'''
	assert_all_finite(vec_st)
	assert_all_finite(bin_sizes)
	rdn_range = cartesian_product(*[np.linspace(*bin_size) for bin_size in bin_sizes])
	sgns = vec_st[:,0]
	values = vec_st[:, 1:]
	a = np.dot(rdn_range, values.transpose()) <= 1 	
	sa = sgns * a									
	rdn = np.sum(sa, axis=-1)
	rdn = rdn.reshape(*[bin_size[-1] for bin_size in bin_sizes]).transpose()
	return rdn

#############################
# Scikit-learn transformers #
#############################
# Modified:
#	- Deleted params
#	- Flatten et normalize in __init__
#	- Added resolution

class EulerCharacteristicProfile(BaseEstimator, TransformerMixin):
	'''
	Transformer computing Euler characteristic profiles.

	The input should be a vectorized simplex tree (or a list of them) of type ndarray as returned by get_vectorized_st. The output is a ndarray representing a sampling of their Euler characteristic profiles on a discrete grid. 

	Parameters:
		resolution 	: tuple, shape of the sampling grid of the Euler characteristic profile
		quantiles	: list of tuples (lower_quantile, higher_quantile) for each filtrations of the vectorized simplex tree. The values of the quantiles will be considered as bounds for the sampling of the Euler profile. 
		val_ranges	: list of tuples of the form (lower_bound, higher_bound). The values are the bounds for the sampling of the Euler profile.
		flatten		: bool, if True, the Euler characteristic profiles are flatten.
		pt_cld		: bool, if True, the computation of the values of quantiles is optimized to skip the vertices on the first filtration (Cech or alpha) and focus on them for all other filtrations. Here, the vectorized simplex trees must be computed by upper star from the values on vertices.
		normalize	: bool determining (only when pt_cld=True) if an Euler profile coming from a point cloud should be normalized by the number of points (assuming that vertices are all at the beginning of the simplex tree).
	Attributes:
		n_params_		: int, number of parameters of the Euler characteristic profile
		val_quantiles_ 	: list of tuples of the form (value of lower_quantile, value of higher_quantile) containing the values of the quantiles when the quantiles are used.

	'''

	def __init__(self, resolution=(), quantiles=[], val_ranges=[], flatten=True, pt_cld=False, normalize=True):
		self.resolution = resolution
		self.quantiles = quantiles
		self.val_ranges = val_ranges
		self.flatten = flatten
		self.pt_cld = pt_cld
		self.normalize = normalize

	def fit(self, X, y=None):
		# Initialize n_params
		try:
			self.n_params_ = len(self.resolution)
		except TypeError:
			print('TypeError: Resolution should be a valid shape of type tuple.')

		# Dealing with several vectorized simplex trees at once
		if type(X) is list:
			vec_sts = X
		else:
			vec_sts = [X]

		# Computing quantiles if necessary and setting the values of val_ranges
		if not self.val_ranges:
			if self.pt_cld:
				val_quantiles = [(min([np.quantile(vec_st[:,1][vec_st[:,1]>0], self.quantiles[0][0]) for vec_st in vec_sts]), max([np.quantile(vec_st[:,1][vec_st[:,1]>0], self.quantiles[0][1]) for vec_st in vec_sts]))] + [(min([np.quantile(vec_st[:np.argmax(vec_st[:,0]<0),i+1], self.quantiles[i][0]) for vec_st in vec_sts]), max([np.quantile(vec_st[:np.argmax(vec_st[:,0]<0),i+1], self.quantiles[i][1]) for vec_st in vec_sts])) for i in range(1, self.n_params_)]
			else:
				val_quantiles = [(min([np.quantile(vec_st[:,i+1], self.quantiles[i][0]) for vec_st in vec_sts]), max([np.quantile(vec_st[:,i+1], self.quantiles[i][1]) for vec_st in vec_sts])) for i in range(self.n_params_)]
			self.val_quantiles_ = val_quantiles
			self.val_ranges = val_quantiles
		return self
	
	

	def transform(self, X):
		if not self.n_params_:
			print('Warning: Euler characteristic volumes are empty since n_params==0.')

		# Dealing with several vectorized simplex trees at once
		if type(X) is list:
			vec_sts = X
		else:
			vec_sts =[X]

		# Computing the Euler characteristic profiles
		bin_sizes = [(self.val_ranges[i][0], self.val_ranges[i][1], self.resolution[i]) for i in range(self.n_params_)]
		ecps = []
		for vec_st in vec_sts:
			if self.pt_cld and self.normalize:
				n_pts = np.argmax(vec_st[:,0]<0)
			else:
				n_pts = 1
			ecp = compute_euler_profile(vec_st, bin_sizes)/n_pts
			ecps.append(ecp.flatten() if self.flatten else ecp)
		
		# Dealing with several vectorized simplex trees at once
		if type(X) is list:
			return ecps
		else:
			return ecps[0]

class RadonTransform(BaseEstimator, TransformerMixin):
	'''
	Transformer computing Radon transforms of Euler characteristic profiles.

	The input should be a vectorized simplex tree (or a list of them) of type ndarray as returned by get_vectorized_st. The output is a ndarray representing a sampling of the Radon transform of their Euler characteristic profiles on a discrete grid. 

	Parameters:
		resolution 			: tuple, shape of the sampling grid of the Radon transform
		quantiles			: list of float representing the (lower) quantile for each filtration of the vectorized simplex tree. The values of the quantiles will used to compute bounds for the sampling of the Radon transform. The bound for each filtration is: [0, stretch_factor/max(min_val_quantile,(value of quantile))]. These bounds are meaningful only with wavelet or exp primitive kernels.
		val_ranges			: list of tuples of the form (lower_bound, higher_bound) for each parameter. The values are the bounds for the sampling of the Radon transform.
		stretch_factor		: float, used in the computation of bounds from quantiles (see quantiles parameter).
		min_val_quantile	: float, representing the minimum value admitted for the value of a quantile. It is used in the computation of bounds from quantiles (see quantiles parameter).
		flatten				: bool, if True, the Radon transforms are flatten.
		pt_cld				: bool, if True, the computation of the values of quantiles is optimized to skip the vertices on the first filtration (Cech or alpha) and focus on them for all other filtrations. Here, the vectorized simplex trees must be computed by upper star from the values on vertices.
		normalize			: bool determining (only when pt_cld=True) if a Radon transform coming from a point cloud should be normalized by the number of points (assuming that vertices are all at the beginning of the simplex tree).
	Attributes:
		n_params_			: int, number of parameters of the Radon transform
		val_quantiles_ 		: list of floats containing the values of the quantiles when the quantiles are used.

	'''

	def __init__(self, resolution=(), quantiles=[], val_ranges=[], flatten=True, pt_cld=False, normalize=True, stretch_factor=4., min_val_quantile=1e-7):
		self.resolution = resolution
		self.quantiles = quantiles
		self.val_ranges = val_ranges
		self.flatten = flatten
		self.pt_cld = pt_cld
		self.normalize = normalize
		self.stretch_factor = stretch_factor
		self.min_val_quantile = min_val_quantile
		
	def fit(self, X, y=None):
		# Initialize n_params
		try:
			self.n_params_ = len(self.resolution)
		except TypeError:
			print('TypeError: Resolution should be a valid shape of type tuple.')

		# Dealing with several vectorized simplex trees at once
		if type(X) is list:
			vec_sts = X
		else:
			vec_sts = [X]


		if not self.val_ranges:
			if self.pt_cld:
				val_quantiles = [min([np.quantile(vec_st[:,1][vec_st[:,1]>0], self.quantiles[0]) for vec_st in vec_sts])] + [min([np.quantile(vec_st[:np.argmax(vec_st[:,0]<0),i+1], self.quantiles[i]) for vec_st in vec_sts]) for i in range(1, self.n_params_)]
			else:
				val_quantiles = [min([np.quantile(vec_st[:,i+1], self.quantiles[i]) for vec_st in vec_sts]) for i in range(self.n_params_)]
			self.val_quantiles_ = val_quantiles
			self.val_ranges = [(0, self.stretch_factor/max(self.min_val_quantile,val_q)) for val_q in self.val_quantiles_]
		return self

	def transform(self, X):
		if not self.n_params_:
			print('Warning: Radon transforms are empty since n_params==0.')

		# Dealing with several vectorized simplex trees at once
		if type(X) is list:
			vec_sts = X
		else:
			vec_sts = [X]

		bin_sizes = [(self.val_ranges[i][0], self.val_ranges[i][1], self.resolution[i]) for i in range(self.n_params_)]
		radons = []
		for vec_st in vec_sts:
			if self.pt_cld and self.normalize:
				n_pts = np.argmax(vec_st[:,0]<0)
			else:
				n_pts = 1
			radon = compute_radon_transform(vec_st, bin_sizes)/n_pts
			radons.append(radon.flatten() if self.flatten else radon)

		# Dealing with several vectorized simplex trees at once
		if type(X) is list:
			return radons
		else:
			return radons[0]

class HybridTransform(BaseEstimator, TransformerMixin):
	'''
	Transformer computing hybrid transforms of Euler characteristic profiles.

	The input should be a vectorized simplex tree (or a list of them) of type ndarray as returned by get_vectorized_st. The output is a ndarray representing a sampling of the hybrid transform of their Euler characteristic profiles on a discrete grid. 

	Parameters:
		resolution 			: tuple, shape of the sampling grid of the hybrid transform
		quantiles			: list of float representing the (lower) quantile for each filtration of the vectorized simplex tree. The values of the quantiles will used to compute bounds for the sampling of the hybrid transform. The bound for each filtration is: [0, stretch_factor/max(min_val_quantile,(value of quantile))]. These bounds are meaningful only with wavelet or exp primitive kernels.
		val_ranges			: list of tuples of the form (lower_bound, higher_bound) for each parameter. The values are the bounds for the sampling of the hybrid transform.
		kernel_name			: string of the form 'wavelet_p', 'exp_p' or 'cos_p' to choose respectively between lambda x: x**p*np.exp(-np.abs(x)**p), lambda x: np.exp(-np.abs(x)**p) and lambda x: np.cos(x**p)
		kernel				: function, **primitive** kernel of the hybrid transform. If kernel_name is not None, this parameter is skipped.
		stretch_factor		: float, used in the computation of bounds from quantiles (see quantiles parameter).
		min_val_quantile	: float, representing the minimum value admitted for the value of a quantile. It is used in the computation of bounds from quantiles (see quantiles parameter).
		flatten				: bool, if True, the hybrid transforms are flatten.
		pt_cld				: bool, if True, the computation of the values of quantiles is optimized to skip the vertices on the first filtration (Cech or alpha) and focus on them for all other filtrations. Here, the vectorized simplex trees must be computed by upper star from the values on vertices.
		normalize			: bool determining (only when pt_cld=True) if a hybrid transform coming from a point cloud should be normalized by the number of points (assuming that vertices are all at the beginning of the simplex tree).
	Attributes:
		n_params_			: int, number of parameters of the hybrid transform
		val_quantiles_ 		: list of floats containing the values of the quantiles when the quantiles are used.

	'''

	def __init__(self, resolution=(), quantiles=[], val_ranges=[], kernel_name=None, kernel=None, stretch_factor=4, min_val_quantile=1e-7, flatten=True, pt_cld=False, normalize=True):
		self.resolution = resolution
		self.quantiles = quantiles
		self.val_ranges = val_ranges
		self.kernel_name = kernel_name
		self.kernel = kernel
		self.stretch_factor = stretch_factor
		self.min_val_quantile = min_val_quantile
		self.flatten = flatten
		self.pt_cld = pt_cld
		self.normalize = normalize

	def fit(self, X, y=None):
		# Initialize n_params
		try:
			self.n_params_ = len(self.resolution)
		except TypeError:
			print('TypeError: Resolution should be a valid shape of type tuple.')

		# Dealing with several vectorized simplex trees at once
		if type(X) is list:
			vec_sts = X
		else:
			vec_sts = [X]

		# Initialize kernel if needed
		if self.kernel_name is not None:
			if 'wavelet' in self.kernel_name: # of the form wavelet_power to get x**power*exp(-x**power)
				p = float(self.kernel_name.split('_')[-1])
				self.kernel = lambda x: x**p*np.exp(-np.abs(x)**p)
			elif 'cos' in self.kernel_name: # of the form cos_power to get cos(x**power)
				p = float(self.kernel_name.split('_')[-1])
				self.kernel = lambda x: np.cos(x**p)
			elif 'exp' in self.kernel_name: # of the form cos_power to get exp(-x**power)
				p = float(self.kernel_name.split('_')[-1])
				self.kernel = lambda x: np.exp(-np.abs(x)**p)
			else:
				print('KernelError: kernel '+self.kernel_name+' is not known.')

		# Computing quantiles if necessary and setting the values of val_ranges
		if not self.val_ranges:
			if self.pt_cld:
				val_quantiles = [min([np.quantile(vec_st[:,1][vec_st[:,1]>0], self.quantiles[0]) for vec_st in vec_sts])] + [min([np.quantile(vec_st[:np.argmax(vec_st[:,0]<0),i+1], self.quantiles[i]) for vec_st in vec_sts]) for i in range(1, self.n_params_)]
			else:
				val_quantiles = [min([np.quantile(vec_st[:,i+1], self.quantiles[i]) for vec_st in vec_sts]) for i in range(self.n_params_)]
			self.val_quantiles_ = val_quantiles
			self.val_ranges = [(0, self.stretch_factor/max(self.min_val_quantile,val_q)) for val_q in self.val_quantiles_]
		return self

	def transform(self, X):
		if not self.n_params_:
			print('Warning: Hybrid transforms are empty since n_params==0.')

		# Dealing with several vectorized simplex trees at once
		if type(X) is list:
			vec_sts = X
		else:
			vec_sts =[X]
		
		if self.kernel is None:
			print('TypeError: Kernel is of NoneType instead of lambda function.')
			return None
		else:
			bin_sizes = [(self.val_ranges[i][0], self.val_ranges[i][1], self.resolution[i]) for i in range(self.n_params_)]
			hts = []
			for vec_st in vec_sts:
				if self.pt_cld and self.normalize:
					n_pts = np.argmax(vec_st[:,0]<0)
				else:
					n_pts = 1
				ht_vals = compute_hybrid_transform(vec_st, bin_sizes, self.kernel)/n_pts
				hts.append(ht_vals.flatten() if self.flatten else ht_vals)
				
			# Dealing with several vectorized simplex trees at once
			if type(X) is list:
				return hts
			else:
				return hts[0]