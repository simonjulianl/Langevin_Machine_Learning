import numpy as np
from tqdm import trange
from numpy import newaxis
from .base_simulation import Integration

# HK class analytic_method(Integration):
class harmonic_integrator(Integration):

	def helper(self=None):
		'''print the common parameters helper'''
		for parent in analytic_method.__bases__:
			print(help(parent))

	def __init__(self, *args, **kwargs) -> object:

		super(analytic_method, self).__init__(*args, **kwargs)

		try:
			self._intSetting = {
				'iterations': kwargs['iterations'],
				'DumpFreq': kwargs['DumpFreq'],
				'time_step': kwargs['time_step'],
				'integrator_method':kwargs['integrator_method']
			}

		except:
			raise TypeError('Integration setting error ( iterations / DumpFreq / gamma / time_step /integrator_method )')

	def integrate(self):

		time_step = self._intSetting['time_step']
		total_samples = self._intSetting['iterations'] // self._intSetting['DumpFreq']
		particle = self._configuration['particle']
		N = self._configuration['N'] # samples
		DIM = self._configuration['DIM']
		integrator_method = self._intSetting['integrator_method']

		q_list = np.zeros((total_samples, N, particle,  DIM))
		p_list = np.zeros((total_samples, N, particle,  DIM))

		for i in trange(total_samples):


			q = self._configuration['phase_space'].get_q()
			p = self._configuration['phase_space'].get_p()

			vecs, y, y_diff = integrator_method(time_step/2,**self._configuration)

			p_temp = np.zeros(y.shape)
			for z in range(N):
				p_temp[z] = np.dot(vecs[z], y_diff[z])

			p = p_temp.reshape(-1, particle, DIM)

			self._configuration['phase_space'].set_q(q)
			self._configuration['phase_space'].set_p(p)

			vecs, y, y_diff = integrator_method(time_step,**self._configuration)

			q_temp = np.zeros(y.shape)
			for z in range(N):
				q_temp[z] = np.dot(vecs[z], y[z])

			q = q_temp.reshape(-1,particle,DIM)

			for z in range(N):
				self._configuration['pb_q'].adjust_real(q[z], self._configuration['BoxSize'])

			self._configuration['phase_space'].set_q(q)
			self._configuration['phase_space'].set_p(p)

			vecs, y, y_diff = integrator_method(time_step/2,**self._configuration)

			p_temp = np.zeros(y.shape)

			for z in range(N):
				p_temp[z] = np.dot(vecs[z], y_diff[z])

			p = p_temp.reshape(-1, particle, DIM)

			self._configuration['phase_space'].set_q(q)
			self._configuration['phase_space'].set_p(p)

			q_list[i] = self._configuration['phase_space'].get_q()
			p_list[i] = self._configuration['phase_space'].get_p()

		return(q_list, p_list)
