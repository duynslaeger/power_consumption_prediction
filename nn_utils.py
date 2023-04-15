# Authors : Olivier Vertu Ndinga Oba, Bradon Mola, Sacha Duynslaeger
# EPB, ULB

from scipy import stats



# ----------- Preprocessing -----------------
# code inspired by the jupyter notebook of Vincent Imard, in the course "Introduction à la prédiction de séries temporelles" from the university of Grenoble

def normalize(data, MIN, MAX):
	normalized = (data - MIN) / (MAX - MIN)
	return normalized



def unnormalize(data, MIN, MAX):
	unnormalized = data*(MAX - MIN) + MIN
	return unnormalized


def remove_extremes(df):
    z_score = stats.zscore(df)
    filtered = (np.abs(z_score) < 2).all(axis = 1)
    new_df = df[filtered]
    return new_df


# ----------- Activation functions -----------------

def ReLU(X):
	"""
	Apply the Reactivation Linear Unit on a matrix. Basically, it applies max(0,x_i_j) to all element of the matrix X
		-X is a numpy matrix
	"""
	return X*(X>0)



# ----------- Loss -----------------


def mean_squared_error():
	# TO DO
	pass

