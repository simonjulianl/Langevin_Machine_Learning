import numpy as np
from tqdm import trange
from numpy import newaxis
from .base_simulation import Integration

class Finite_difference_first_derivative(Integration):

	def helper(self=None):
		'''print the common parameters helper'''
		for parent in Finite_difference_first_derivative.__bases__:
			print(help(parent))

	def __init__(self, *args, **kwargs) -> object:

		super(Finite_difference_first_derivative, self).__init__(*args, **kwargs)


	def integrate(self):

		hamiltonian = self._configuration['hamiltonian']
		q = self._configuration['phase_space'].get_q()
		delta = 0.000001

		q1x = self._configuration['phase_space'].get_q()
		q1y = self._configuration['phase_space'].get_q()
		q2x = self._configuration['phase_space'].get_q()
		q2y = self._configuration['phase_space'].get_q()
		q3x = self._configuration['phase_space'].get_q()
		q3y = self._configuration['phase_space'].get_q()
		q4x = self._configuration['phase_space'].get_q()
		q4y = self._configuration['phase_space'].get_q()

		print('input q')
		print(q)

		print('before q1x',q1x)
		q1x[:,0,0] = q[:,0,0] + delta # q1x + del_q1x
		print('after q1x',q1x)
		print('before q1y', q1y)
		q1y[:,0,1] = q[:,0,1] + delta # q1y + del_q1y
		print('after q1y',q1y)

		print('before q2x', q2x)
		q2x[:,1,0] = q[:,1,0] + delta # q2x + del_q2x
		print('after q2x',q2x)
		print('before q2y', q2y)
		q2y[:,1,1] = q[:,1,1] + delta # q2y + del_q2y
		print('after q2y', q2y)

		print('before q3x',q3x)
		q3x[:,2,0] = q[:,2,0] + delta # q1x + del_q1x
		print('after q3x',q3x)
		print('before q3y', q3y)
		q3y[:,2,1] = q[:,2,1] + delta # q1y + del_q1y
		print('after q3y',q3y)

		print('before q4x', q4x)
		q4x[:,3,0] = q[:,3,0] + delta # q2x + del_q2x
		print('after q4x',q4x)
		print('before q4y', q4y)
		q4y[:,3,1] = q[:,3,1] + delta # q2y + del_q2y
		print('after q4y', q4y)

		# partial U/ partical q1_x
		U_q1x = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		self._configuration['phase_space'].set_q(q1x)
		print('q1x_del',self._configuration['phase_space'].get_q())
		U_q1x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		derivative_U_q1x = ( U_q1x_del - U_q1x ) / delta

		# partial U/ partical q1_y
		U_q1y = U_q1x

		self._configuration['phase_space'].set_q(q1y)
		print('q1y_del', self._configuration['phase_space'].get_q())
		U_q1y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		derivative_U_q1y = (U_q1y_del - U_q1y) / delta

		# partial U/ partical q2_x
		U_q2x = U_q1x

		self._configuration['phase_space'].set_q(q2x)
		print('q2x_del', self._configuration['phase_space'].get_q())
		U_q2x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		derivative_U_q2x = (U_q2x_del - U_q2x) / delta

		# partial U/ partical q2_y
		U_q2y = U_q1x

		self._configuration['phase_space'].set_q(q2y)
		print('q2y_del', self._configuration['phase_space'].get_q())
		U_q2y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		derivative_U_q2y = (U_q2y_del - U_q2y) / delta

		# partial U/ partical q3_x
		U_q3x = U_q1x

		self._configuration['phase_space'].set_q(q3x)
		print('q3x_del', self._configuration['phase_space'].get_q())
		U_q3x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		derivative_U_q3x = (U_q3x_del - U_q3x) / delta

		# partial U/ partical q3_y
		U_q3y = U_q1x

		self._configuration['phase_space'].set_q(q3y)
		print('q3y_del', self._configuration['phase_space'].get_q())
		U_q3y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		derivative_U_q3y = (U_q3y_del - U_q3y) / delta

		# partial U/ partical q4_x
		U_q4x = U_q1x

		self._configuration['phase_space'].set_q(q4x)
		print('q4x_del', self._configuration['phase_space'].get_q())
		U_q4x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		derivative_U_q4x = (U_q4x_del - U_q4x) / delta

		# partial U/ partical q4_y
		U_q4y = U_q1x

		self._configuration['phase_space'].set_q(q4y)
		print('q4y_del', self._configuration['phase_space'].get_q())
		U_q4y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		derivative_U_q4y = (U_q4y_del - U_q4y) / delta

		first_derivative = np.array(
			[derivative_U_q1x, derivative_U_q1y, derivative_U_q2x, derivative_U_q2y ,
			 derivative_U_q3x, derivative_U_q3y, derivative_U_q4x, derivative_U_q4y])

		return first_derivative
