import numpy as np
from tqdm import trange
from .base_simulation import Integration

class harmonic_integrator(Integration):

	def helper(self=None):
		'''print the common parameters helper'''
		for parent in harmonic_integrator.__bases__:
			print(help(parent))

	def __init__(self, *args, **kwargs) -> object:

		super(harmonic_integrator, self).__init__(*args, **kwargs)

		try:
			self._intSetting = {
				'iterations': kwargs['iterations'],
				'tau': kwargs['tau'],
				'integrator_method':kwargs['integrator_method']
			}

		except:
			raise TypeError('Integration setting error ( iterations / gamma / tau /integrator_method )')

	def integrate(self):

		tau = self._intSetting['tau']
		iterations = self._intSetting['iterations']
		N_particle = self._configuration['particle']
		N = self._configuration['N'] # nsamples
		DIM = self._configuration['DIM']
		integrator_method = self._intSetting['integrator_method']

		q_list = np.zeros((iterations, N, N_particle, DIM))
		p_list = np.zeros((iterations, N, N_particle, DIM))

		for i in trange(iterations):

			q = self._configuration['phase_space'].get_q()
			#p = self._configuration['phase_space'].get_p()

			u, y, ydot = integrator_method(tau/2, **self._configuration)

			# qdot = p terms of nsamples N
			p_temp = np.zeros(y.shape)
			for z in range(N):
				p_temp[z] = np.dot(u[z], ydot[z])  # qdot = u ydot

			p = p_temp.reshape(-1, N_particle, DIM)

			self._configuration['phase_space'].set_q(q)
			self._configuration['phase_space'].set_p(p)

			u, y, ydot = integrator_method(tau, **self._configuration)

			# q terms of nsamples N
			q_temp = np.zeros(y.shape)
			for z in range(N):
				q_temp[z] = np.dot(u[z], y[z]) # q = u y

			q = q_temp.reshape(-1, N_particle, DIM)

			for z in range(N):
				self._configuration['pb_q'].adjust_real(q[z], self._configuration['BoxSize'])

			self._configuration['phase_space'].set_q(q)
			self._configuration['phase_space'].set_p(p)

			u, y, ydot = integrator_method(tau/2,**self._configuration)

			# qdot = p terms of nsamples N
			p_temp = np.zeros(y.shape)
			for z in range(N):
				p_temp[z] = np.dot(u[z], ydot[z])

			p = p_temp.reshape(-1, N_particle, DIM)

			self._configuration['phase_space'].set_q(q)
			self._configuration['phase_space'].set_p(p)

			q_list[i] = self._configuration['phase_space'].get_q()
			p_list[i] = self._configuration['phase_space'].get_p()

		return(q_list, p_list)
