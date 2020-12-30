import numpy as np
from tqdm import trange
from numpy import newaxis
from .base_simulation import Integration

class Finite_difference_second_derivative(Integration):

	def helper(self=None):
		'''print the common parameters helper'''
		for parent in Finite_difference_second_derivative.__bases__:
			print(help(parent))

	def __init__(self, *args, **kwargs) -> object:

		super(Finite_difference_second_derivative, self).__init__(*args, **kwargs)


	def integrate(self):

		hamiltonian = self._configuration['hamiltonian']
		q = self._configuration['phase_space'].get_q()
		delta = 0.0000001

		q1x = self._configuration['phase_space'].get_q()
		print('type',q1x.dtype)
		q1y = self._configuration['phase_space'].get_q()
		q2x = self._configuration['phase_space'].get_q()
		q2y = self._configuration['phase_space'].get_q()
		q3x = self._configuration['phase_space'].get_q()
		q3y = self._configuration['phase_space'].get_q()
		q4x = self._configuration['phase_space'].get_q()
		q4y = self._configuration['phase_space'].get_q()

		q1x2 = self._configuration['phase_space'].get_q()
		q1x1y = self._configuration['phase_space'].get_q()
		q1x2x = self._configuration['phase_space'].get_q()
		q1x2y = self._configuration['phase_space'].get_q()
		q1x3x = self._configuration['phase_space'].get_q()
		q1x3y = self._configuration['phase_space'].get_q()
		q1x4x = self._configuration['phase_space'].get_q()
		q1x4y = self._configuration['phase_space'].get_q()

		q1y2 = self._configuration['phase_space'].get_q()
		q1y2x = self._configuration['phase_space'].get_q()
		q1y2y = self._configuration['phase_space'].get_q()
		q1y3x = self._configuration['phase_space'].get_q()
		q1y3y = self._configuration['phase_space'].get_q()
		q1y4x = self._configuration['phase_space'].get_q()
		q1y4y = self._configuration['phase_space'].get_q()

		q2x2 = self._configuration['phase_space'].get_q()
		q2x2y = self._configuration['phase_space'].get_q()
		q2x3x = self._configuration['phase_space'].get_q()
		q2x3y = self._configuration['phase_space'].get_q()
		q2x4x = self._configuration['phase_space'].get_q()
		q2x4y = self._configuration['phase_space'].get_q()

		q2y2 = self._configuration['phase_space'].get_q()
		q2y3x = self._configuration['phase_space'].get_q()
		q2y3y = self._configuration['phase_space'].get_q()
		q2y4x = self._configuration['phase_space'].get_q()
		q2y4y = self._configuration['phase_space'].get_q()

		q3x2 = self._configuration['phase_space'].get_q()
		q3x3y = self._configuration['phase_space'].get_q()
		q3x4x = self._configuration['phase_space'].get_q()
		q3x4y = self._configuration['phase_space'].get_q()

		q3y2 = self._configuration['phase_space'].get_q()
		q3y4x = self._configuration['phase_space'].get_q()
		q3y4y = self._configuration['phase_space'].get_q()

		q4x2 = self._configuration['phase_space'].get_q()
		q4x4y = self._configuration['phase_space'].get_q()

		q4y2 = self._configuration['phase_space'].get_q()

		print('input q')
		print(q)

		q1x[:,0,0] = q[:,0,0] + delta # q1x + del_q1x
		print('q1x',q1x)
		q1x2[:,0,0] = q[:,0,0] + 2*delta # q1x + 2*del_q1x
		print('q1x2',q1x2)

		q1x1y[:,0,0] = q[:,0,0] + delta  # 1x+del
		q1x1y[:,0,1] = q[:,0,1] + delta  # 1y+del
		print('q1x1y',q1x1y)

		q1x2x[:,0,0] = q[:,0,0] + delta
		q1x2x[:,1,0] = q[:,1,0] + delta
		print('q1x2x',q1x2x)

		q1x2y[:,0,0] = q[:,0,0] + delta  # 1x+del
		q1x2y[:,1,1] = q[:,1,1] + delta  # 2y+del
		print('q1x2y',q1x2y)

		q1x3x[:,0,0] = q[:,0,0] + delta
		q1x3x[:,2,0] = q[:,2,0] + delta
		print('q1x3x',q1x3x)

		q1x3y[:,0,0] = q[:,0,0] + delta
		q1x3y[:,2,1] = q[:,2,1] + delta
		print('q1x3y',q1x3y)

		q1x4x[:,0,0] = q[:,0,0] + delta
		q1x4x[:,3,0] = q[:,3,0] + delta
		print('q1x4x',q1x4x)

		q1x4y[:,0,0] = q[:,0,0] + delta
		q1x4y[:,3,1] = q[:,3,1] + delta
		print('q1x4y',q1x4y)

		q1y[:,0,1] = q[:,0,1] + delta # q1y + del_q1y
		print('q1y',q1y)
		q1y2[:,0,1] = q[:,0,1] + 2*delta # q1x + 2*del_q1x
		print('q1y2',q1y2)

		q1y2x[:,0,1] = q[:,0,1] + delta
		q1y2x[:,1,0] = q[:,1,0] + delta
		print('q1y2x',q1y2x)

		q1y2y[:,0,1] = q[:,0,1] + delta
		q1y2y[:,1,1] = q[:,1,1] + delta
		print('q1y2y',q1y2y)

		q1y3x[:,0,1] = q[:,0,1] + delta
		q1y3x[:,2,0] = q[:,2,0] + delta
		print('q1y3x',q1y3x)

		q1y3y[:,0,1] = q[:,0,1] + delta
		q1y3y[:,2,1] = q[:,2,1] + delta
		print('q1y3y',q1y3y)

		q1y4x[:,0,1] = q[:,0,1] + delta
		q1y4x[:,3,0] = q[:,3,0] + delta
		print('q1y4x',q1y4x)

		q1y4y[:,0,1] = q[:,0,1] + delta
		q1y4y[:,3,1] = q[:,3,1] + delta
		print('q1y4y',q1y4y)

		q2x[:,1,0] = q[:,1,0] + delta # q2x + del_q2x
		print('q2x',q2x)
		q2x2[:,1,0] = q[:,1,0] + 2*delta # q1x + 2*del_q1x
		print('q2x2',q2x2)

		q2x2y[:,1,0] = q[:,1,0] + delta # q1x + 2*del_q1x
		q2x2y[:,1,1] = q[:,1,1] + delta  # q1x + 2*del_q1x
		print('q2x2y',q2x2y)

		q2x3x[:,1,0] = q[:,1,0] + delta # q1x + 2*del_q1x
		q2x3x[:,2,0] = q[:,2,0] + delta  # q1x + 2*del_q1x
		print('q2x3x',q2x3x)

		q2x3y[:,1,0] = q[:,1,0] + delta # q1x + 2*del_q1x
		q2x3y[:,2,1] = q[:,2,1] + delta  # q1x + 2*del_q1x
		print('q2x3y',q2x3y)

		q2x4x[:,1,0] = q[:,1,0] + delta  # q1x + 2*del_q1x
		q2x4x[:,3,0] = q[:,3,0] + delta  # q1x + 2*del_q1x
		print('q2x4x', q2x4x)

		q2x4y[:,1,0] = q[:,1,0] + delta  # q1x + 2*del_q1x
		q2x4y[:,3,1] = q[:,3,1] + delta  # q1x + 2*del_q1x
		print('q2x4y', q2x4y)

		q2y[:,1,1] = q[:,1,1] + delta # q2y + del_q2y
		print('q2y', q2y)
		q2y2[:,1,1] = q[:,1,1] + 2*delta # q1x + 2*del_q1x
		print('q2y2',q2y2)

		q2y3x[:,1,1] = q[:,1,1] + delta  # q1x + 2*del_q1x
		q2y3x[:,2,0] = q[:,2,0] + delta  # q1x + 2*del_q1x
		print('q2y3x', q2y3x)

		q2y3y[:,1,1] = q[:,1,1] + delta  # q1x + 2*del_q1x
		q2y3y[:,2,1] = q[:,2,1] + delta  # q1x + 2*del_q1x
		print('q2y3y', q2y3y)

		q2y4x[:,1,1] = q[:,1,1] + delta  # q1x + 2*del_q1x
		q2y4x[:,3,0] = q[:,3,0] + delta  # q1x + 2*del_q1x
		print('q2y4x', q2y4x)

		q2y4y[:,1,1] = q[:,1,1] + delta  # q1x + 2*del_q1x
		q2y4y[:,3,1] = q[:,3,1] + delta  # q1x + 2*del_q1x
		print('q2y4y', q2y4y)

		q3x[:, 2, 0] = q[:, 2, 0] + delta  # q2x + del_q2x
		print('q3x', q3x)
		q3x2[:, 2, 0] = q[:, 2, 0] + 2 * delta  # q1x + 2*del_q1x
		print('q3x2', q3x2)

		q3x3y[:, 2, 0] = q[:, 2, 0] + delta  # q1x + 2*del_q1x
		q3x3y[:, 2, 1] = q[:, 2, 1] + delta  # q1x + 2*del_q1x
		print('q3x3y', q3x3y)

		q3x4x[:, 2, 0] = q[:, 2, 0] + delta  # q1x + 2*del_q1x
		q3x4x[:, 3, 0] = q[:, 3, 0] + delta  # q1x + 2*del_q1x
		print('q3x4x', q3x4x)

		q3x4y[:, 2, 0] = q[:, 2, 0] + delta  # q1x + 2*del_q1x
		q3x4y[:, 3, 1] = q[:, 3, 1] + delta  # q1x + 2*del_q1x
		print('q3x4y', q3x4y)

		q3y[:,2,1] = q[:,2,1] + delta # q2y + del_q2y
		print('q3y', q3y)
		q3y2[:,2,1] = q[:,2,1] + 2*delta # q1x + 2*del_q1x
		print('q3y2',q3y2)

		q3y4x[:,2,1] = q[:,2,1] + delta  # q1x + 2*del_q1x
		q3y4x[:,3,0] = q[:,3,0] + delta  # q1x + 2*del_q1x
		print('q3y4x', q3y4x)

		q3y4y[:,2,1] = q[:,2,1] + delta  # q1x + 2*del_q1x
		q3y4y[:,3,1] = q[:,3,1] + delta  # q1x + 2*del_q1x
		print('q3y4y', q3y4y)

		q4x[:, 3, 0] = q[:, 3, 0] + delta  # q2x + del_q2x
		print('q4x', q4x)
		q4x2[:, 3, 0] = q[:, 3, 0] + 2 * delta  # q1x + 2*del_q1x
		print('q4x2', q4x2)

		q4x4y[:, 3, 0] = q[:, 3, 0] + delta  # q1x + 2*del_q1x
		q4x4y[:, 3, 1] = q[:, 3, 1] + delta  # q1x + 2*del_q1x
		print('q4x4y', q4x4y)

		q4y[:, 3, 1] = q[:, 3, 1] + delta  # q2x + del_q2x
		print('q4y', q4y)
		q4y2[:, 3, 1] = q[:, 3, 1] + 2 * delta  # q1x + 2*del_q1x
		print('q4y2', q4y2)

		# partial2 U/ partical q1_x2
		U = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		self._configuration['phase_space'].set_q(q1x)
		print('q1x_del',self._configuration['phase_space'].get_q())
		U_q1x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		self._configuration['phase_space'].set_q(q1x2)
		print('q1x2_del',self._configuration['phase_space'].get_q())
		U_q1x_del2 = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1x2 = ( U_q1x_del2 - 2 * U_q1x_del + U ) / (delta * delta)

		# partial2 U/ partical q1_x_q1_y
		self._configuration['phase_space'].set_q(q1y)
		print('q1y_del', self._configuration['phase_space'].get_q())
		U_q1y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		self._configuration['phase_space'].set_q(q1x1y)
		print('q1x1y_del', self._configuration['phase_space'].get_q())
		U_q1x1y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1x1y = ( U_q1x1y_del -U_q1x_del - U_q1y_del + U ) / (delta * delta)

		# partial U2/ partical q1_x_q2_x
		self._configuration['phase_space'].set_q(q2x)
		print('q2x_del',self._configuration['phase_space'].get_q())
		U_q2x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		self._configuration['phase_space'].set_q(q1x2x)
		print('q1x2x_del',self._configuration['phase_space'].get_q())
		U_q1x2x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1x2x = ( U_q1x2x_del  -U_q1x_del - U_q2x_del + U ) / (delta * delta)

		# partial U2/ partical q1_x_q2_y
		self._configuration['phase_space'].set_q(q2y)
		print('q2y_del', self._configuration['phase_space'].get_q())
		U_q2y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		self._configuration['phase_space'].set_q(q1x2y)
		print('q1x2y_del', self._configuration['phase_space'].get_q())
		U_q1x2y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1x2y = (U_q1x2y_del - U_q1x_del - U_q2y_del + U) / (delta * delta)

		# partial U2/ partical q1_x_q3_x
		self._configuration['phase_space'].set_q(q3x)
		print('q3x_del',self._configuration['phase_space'].get_q())
		U_q3x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		self._configuration['phase_space'].set_q(q1x3x)
		print('q1x3x_del',self._configuration['phase_space'].get_q())
		U_q1x3x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1x3x = ( U_q1x3x_del  -U_q1x_del - U_q3x_del + U ) / (delta * delta)

		# partial U2/ partical q1_x_q3_y
		self._configuration['phase_space'].set_q(q3y)
		print('q3y_del', self._configuration['phase_space'].get_q())
		U_q3y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		self._configuration['phase_space'].set_q(q1x3y)
		print('q1x3y_del', self._configuration['phase_space'].get_q())
		U_q1x3y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1x3y = (U_q1x3y_del - U_q1x_del - U_q3y_del + U) / (delta * delta)

		# partial U2/ partical q1_x_q4_x
		self._configuration['phase_space'].set_q(q4x)
		print('q4x_del',self._configuration['phase_space'].get_q())
		U_q4x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		self._configuration['phase_space'].set_q(q1x4x)
		print('q1x4x_del',self._configuration['phase_space'].get_q())
		U_q1x4x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1x4x = ( U_q1x4x_del  -U_q1x_del - U_q4x_del + U ) / (delta * delta)

		# partial U2/ partical q1_x_q4_y
		self._configuration['phase_space'].set_q(q4y)
		print('q4y_del', self._configuration['phase_space'].get_q())
		U_q4y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		self._configuration['phase_space'].set_q(q1x4y)
		print('q1x4y_del', self._configuration['phase_space'].get_q())
		U_q1x4y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1x4y = (U_q1x4y_del - U_q1x_del - U_q4y_del + U) / (delta * delta)

		# partial U2/ partical q1_y_q1_x
		partial2_U_q1y1x = partial2_U_q1x1y

		# partial U2/ partical q1_y2
		self._configuration['phase_space'].set_q(q1y2)
		print('q1y2_del',self._configuration['phase_space'].get_q())
		U_q1y_del2 = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1y2 = ( U_q1y_del2 - 2 * U_q1y_del + U ) / (delta * delta)

		# partial U2/ partical q1_y_q2_x
		self._configuration['phase_space'].set_q(q1y2x)
		print('q1y2x_del', self._configuration['phase_space'].get_q())
		U_q1y2x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1y2x = (U_q1y2x_del - U_q1y_del - U_q2x_del + U) / (delta * delta)

		# partial U2/ partical q1_y_q2_y
		self._configuration['phase_space'].set_q(q1y2y)
		print('q1y2y_del', self._configuration['phase_space'].get_q())
		U_q1y2y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1y2y = (U_q1y2y_del - U_q1y_del - U_q2y_del + U) / (delta * delta)

		# partial U2/ partical q1_y_q3_x
		self._configuration['phase_space'].set_q(q1y3x)
		print('q1y3x_del', self._configuration['phase_space'].get_q())
		U_q1y3x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1y3x = (U_q1y3x_del - U_q1y_del - U_q3x_del + U) / (delta * delta)

		# partial U2/ partical q1_y_q3_y
		self._configuration['phase_space'].set_q(q1y3y)
		print('q1y3y_del', self._configuration['phase_space'].get_q())
		U_q1y3y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1y3y = (U_q1y3y_del - U_q1y_del - U_q3y_del + U) / (delta * delta)

		# partial U2/ partical q1_y_q4_x
		self._configuration['phase_space'].set_q(q1y4x)
		print('q1y4x_del', self._configuration['phase_space'].get_q())
		U_q1y4x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1y4x = (U_q1y4x_del - U_q1y_del - U_q4x_del + U) / (delta * delta)

		# partial U2/ partical q1_y_q4_y
		self._configuration['phase_space'].set_q(q1y4y)
		print('q1y4y_del', self._configuration['phase_space'].get_q())
		U_q1y4y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q1y4y = (U_q1y4y_del - U_q1y_del - U_q4y_del + U) / (delta * delta)

		# partial U2/ partical q2_x_q1_x
		partial2_U_q2x1x = partial2_U_q1x2x
		# partial U2/ partical q2_x_q1_y
		partial2_U_q2x1y = partial2_U_q1y2x

		# partial U2/ partical q2_x2
		self._configuration['phase_space'].set_q(q2x2)
		print('q2x2_del',self._configuration['phase_space'].get_q())
		U_q2x_del2 = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2x2 = ( U_q2x_del2 - 2 * U_q2x_del + U ) / (delta * delta)

		# partial U2/ partical q2_x_q2_y
		self._configuration['phase_space'].set_q(q2x2y)
		print('q2x2y_del', self._configuration['phase_space'].get_q())
		U_q2x2y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2x2y = (U_q2x2y_del - U_q2x_del - U_q2y_del + U) / (delta * delta)

		# partial U2/ partical q2_x_q3_x
		self._configuration['phase_space'].set_q(q2x3x)
		print('q2x3x_del', self._configuration['phase_space'].get_q())
		U_q2x3x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2x3x = (U_q2x3x_del - U_q2x_del - U_q3x_del + U) / (delta * delta)

		# partial U2/ partical q2_x_q3_y
		self._configuration['phase_space'].set_q(q2x3y)
		print('q2x3y_del', self._configuration['phase_space'].get_q())
		U_q2x3y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2x3y = (U_q2x3y_del - U_q2x_del - U_q3y_del + U) / (delta * delta)

		# partial U2/ partical q2_x_q4_x
		self._configuration['phase_space'].set_q(q2x4x)
		print('q2x4x_del', self._configuration['phase_space'].get_q())
		U_q2x4x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2x4x = (U_q2x4x_del - U_q2x_del - U_q4x_del + U) / (delta * delta)

		# partial U2/ partical q2_x_q4_y
		self._configuration['phase_space'].set_q(q2x4y)
		print('q2x4y_del', self._configuration['phase_space'].get_q())
		U_q2x4y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2x4y = (U_q2x4y_del - U_q2x_del - U_q4y_del + U) / (delta * delta)

		# partial U2/ partical q2_y_q1_x
		partial2_U_q2y1x = partial2_U_q1x2y
		# partial U2/ partical q2_y_q1_y
		partial2_U_q2y1y = partial2_U_q1y2y
		# partial U2/ partical q2_y_q2_x
		partial2_U_q2y2x = partial2_U_q2x2y

		# partial U2/ partical q2_y2
		self._configuration['phase_space'].set_q(q2y2)
		print('q2y2_del', self._configuration['phase_space'].get_q())
		U_q2y_del2 = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2y2 = (U_q2y_del2 - 2 * U_q2y_del + U) / (delta * delta)

		# partial U2/ partical q2_y_q3_x
		self._configuration['phase_space'].set_q(q2y3x)
		print('q2y3x_del', self._configuration['phase_space'].get_q())
		U_q2y3x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2y3x = (U_q2y3x_del - U_q2y_del - U_q3x_del + U) / (delta * delta)

		# partial U2/ partical q2_y_q3_y
		self._configuration['phase_space'].set_q(q2y3y)
		print('q2y3y_del', self._configuration['phase_space'].get_q())
		U_q2y3y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2y3y = (U_q2y3y_del - U_q2y_del - U_q3y_del + U) / (delta * delta)

		# partial U2/ partical q2_y_q4_x
		self._configuration['phase_space'].set_q(q2y4x)
		print('q2y4x_del', self._configuration['phase_space'].get_q())
		U_q2y4x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2y4x = (U_q2y4x_del - U_q2y_del - U_q4x_del + U) / (delta * delta)

		# partial U2/ partical q2_y_q4_y
		self._configuration['phase_space'].set_q(q2y4y)
		print('q2y4y_del', self._configuration['phase_space'].get_q())
		U_q2y4y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q2y4y = (U_q2y4y_del - U_q2y_del - U_q4y_del + U) / (delta * delta)

		# partial U2/ partical q3_x_q1_x
		partial2_U_q3x1x = partial2_U_q1x3x
		# partial U2/ partical q3_x_q1_y
		partial2_U_q3x1y = partial2_U_q1y3x
		# partial U2/ partical q3_x_q2_x
		partial2_U_q3x2x = partial2_U_q2x3x
		# partial U2/ partical q3_x_q2_y
		partial2_U_q3x2y = partial2_U_q2y3x

		# partial U2/ partical q3_x2
		self._configuration['phase_space'].set_q(q3x2)
		print('q3x2_del', self._configuration['phase_space'].get_q())
		U_q3x_del2 = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q3x2 = (U_q3x_del2 - 2 * U_q3x_del + U) / (delta * delta)

		# partial U2/ partical q3_x_q3_y
		self._configuration['phase_space'].set_q(q3x3y)
		print('q3x3y_del', self._configuration['phase_space'].get_q())
		U_q3x3y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q3x3y = (U_q3x3y_del - U_q3x_del - U_q3y_del + U) / (delta * delta)

		# partial U2/ partical q3_x_q4_x
		self._configuration['phase_space'].set_q(q3x4x)
		print('q3x4x_del', self._configuration['phase_space'].get_q())
		U_q3x4x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q3x4x = (U_q3x4x_del - U_q3x_del - U_q4x_del + U) / (delta * delta)

		# partial U2/ partical q3_x_q4_y
		self._configuration['phase_space'].set_q(q3x4y)
		print('q3x4y_del', self._configuration['phase_space'].get_q())
		U_q3x4y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q3x4y = (U_q3x4y_del - U_q3x_del - U_q4y_del + U) / (delta * delta)

		# partial U2/ partical q3_y_q1_x
		partial2_U_q3y1x = partial2_U_q1x3y
		# partial U2/ partical q3_y_q1_y
		partial2_U_q3y1y = partial2_U_q1y3y
		# partial U2/ partical q3_y_q2_x
		partial2_U_q3y2x = partial2_U_q2x3y
		# partial U2/ partical q3_y_q2_y
		partial2_U_q3y2y = partial2_U_q2y3y
		# partial U2/ partical q3_y_q3_x
		partial2_U_q3y3x = partial2_U_q3x3y

		# partial U2/ partical q3_y2
		self._configuration['phase_space'].set_q(q3y2)
		print('q3y2_del', self._configuration['phase_space'].get_q())
		U_q3y_del2 = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q3y2 = (U_q3y_del2 - 2 * U_q3y_del + U) / (delta * delta)

		# partial U2/ partical q3_y_q4_x
		self._configuration['phase_space'].set_q(q3y4x)
		print('q3y4x_del', self._configuration['phase_space'].get_q())
		U_q3y4x_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q3y4x = (U_q3y4x_del - U_q3y_del - U_q4x_del + U) / (delta * delta)

		# partial U2/ partical q3_y_q4_y
		self._configuration['phase_space'].set_q(q3y4y)
		print('q3y4y_del', self._configuration['phase_space'].get_q())
		U_q3y4y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q3y4y = (U_q3y4y_del - U_q3y_del - U_q4y_del + U) / (delta * delta)

		# partial U2/ partical q4_x_q1_x
		partial2_U_q4x1x = partial2_U_q1x4x
		# partial U2/ partical q4_x_q1_y
		partial2_U_q4x1y = partial2_U_q1y4x
		# partial U2/ partical q4_x_q2_x
		partial2_U_q4x2x = partial2_U_q2x4x
		# partial U2/ partical q4_x_q2_y
		partial2_U_q4x2y = partial2_U_q2y4x
		# partial U2/ partical q4_x_q3_x
		partial2_U_q4x3x = partial2_U_q3x4x
		# partial U2/ partical q4_x_q3_y
		partial2_U_q4x3y = partial2_U_q3y4x

		# partial U2/ partical q4_x2
		self._configuration['phase_space'].set_q(q4x2)
		print('q4x2_del', self._configuration['phase_space'].get_q())
		U_q4x_del2 = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q4x2 = (U_q4x_del2 - 2 * U_q4x_del + U) / (delta * delta)

		# partial U2/ partical q4_x_q4_y
		self._configuration['phase_space'].set_q(q4x4y)
		print('q4x4y_del', self._configuration['phase_space'].get_q())
		U_q4x4y_del = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q4x4y = (U_q4x4y_del - U_q4x_del - U_q4y_del + U) / (delta * delta)

		# partial U2/ partical q4_y_q1_x
		partial2_U_q4y1x = partial2_U_q1x4y
		# partial U2/ partical q4_y_q1_y
		partial2_U_q4y1y = partial2_U_q1y4y
		# partial U2/ partical q4_y_q2_x
		partial2_U_q4y2x = partial2_U_q2x4y
		# partial U2/ partical q4_y_q2_y
		partial2_U_q4y2y = partial2_U_q2y4y
		# partial U2/ partical q4_y_q3_x
		partial2_U_q4y3x = partial2_U_q3x4y
		# partial U2/ partical q4_y_q3_y
		partial2_U_q4y3y = partial2_U_q3y4y
		# partial U2/ partical q4_y_q4_x
		partial2_U_q4y4x = partial2_U_q4x4y

		# partial U2/ partical q4_y2
		self._configuration['phase_space'].set_q(q4y2)
		print('q4y2_del', self._configuration['phase_space'].get_q())
		U_q4y_del2 = hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

		partial2_U_q4y2 = (U_q4y_del2 - 2 * U_q4y_del + U) / (delta * delta)

		print('type', partial2_U_q4y2.dtype)
		second_derivative = np.array([partial2_U_q1x2, partial2_U_q1x1y, partial2_U_q1x2x, partial2_U_q1x2y, partial2_U_q1x3x, partial2_U_q1x3y, partial2_U_q1x4x, partial2_U_q1x4y,
									  partial2_U_q1y1x, partial2_U_q1y2, partial2_U_q1y2x ,partial2_U_q1y2y, partial2_U_q1y3x, partial2_U_q1y3y, partial2_U_q1y4x, partial2_U_q1y4y,
									  partial2_U_q2x1x, partial2_U_q2x1y,partial2_U_q2x2, partial2_U_q2x2y, partial2_U_q2x3x, partial2_U_q2x3y, partial2_U_q2x4x, partial2_U_q2x4y,
									  partial2_U_q2y1x, partial2_U_q2y1y,partial2_U_q2y2x,partial2_U_q2y2, partial2_U_q2y3x, partial2_U_q2y3y, partial2_U_q2y4x, partial2_U_q2y4y,
									  partial2_U_q3x1x, partial2_U_q3x1y, partial2_U_q3x2x, partial2_U_q3x2y, partial2_U_q3x2, partial2_U_q3x3y, partial2_U_q3x4x, partial2_U_q3x4y,
									  partial2_U_q3y1x, partial2_U_q3y1y, partial2_U_q3y2x, partial2_U_q3y2y, partial2_U_q3y3x, partial2_U_q3y2, partial2_U_q3y4x, partial2_U_q3y4y,
									  partial2_U_q4x1x, partial2_U_q4x1y, partial2_U_q4x2x, partial2_U_q4x2y, partial2_U_q4x3x, partial2_U_q4x3y, partial2_U_q4x2, partial2_U_q4x4y,
									  partial2_U_q4y1x, partial2_U_q4y1y, partial2_U_q4y2x, partial2_U_q4y2y, partial2_U_q4y3x, partial2_U_q4y3y, partial2_U_q4y4x, partial2_U_q4y2])

		return second_derivative
