import numpy as np
from tqdm import trange
from numpy import newaxis
from .base_simulation import Integration

class analytic_method(Integration):

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
				'time_step': kwargs['time_step']
			}

		except:
			raise TypeError('Integration setting error ( iterations / DumpFreq / gamma / time_step /integrator_method )')

	def integrate(self):

		time_step = self._intSetting['time_step']
		total_samples = self._intSetting['iterations'] // self._intSetting['DumpFreq']
		particle = self._configuration['particle']

		for i in trange(total_samples):

			q = self._configuration['phase_space'].get_q()
			p = self._configuration['phase_space'].get_p()
			Hamiltonian = self._configuration['hamiltonian']

			p_list_dummy = np.zeros(p.shape)  # to prevent KE from being integrated
			self._configuration['phase_space'].set_p(p_list_dummy)

			U = Hamiltonian.total_energy(self._configuration['phase_space'], self._configuration['pb_q'])

			self._configuration['phase_space'].set_q(q)
			V = Hamiltonian.dHdq(self._configuration['phase_space'], self._configuration['pb_q'])

			self._configuration['phase_space'].set_q(q)
			H = Hamiltonian.d2Hdq2(self._configuration['phase_space'], self._configuration['pb_q'])

			H = np.squeeze(H)
			print(H)
			# Real symmetric matrix
			#inv_H = np.linalg.inv(H)
			print('=========Hessian=========')
			print(H)
			print('=========Hessian.T=========')
			print(H.T)
			print('========= (Hessian - Hessian.T)**2 =========')
			print(( H - H.T)**2)

			vals, vecs = np.linalg.eig(H)

			vals_matrix = np.diag(vals)
			print('=========eigenvalue_matrix=========')
			print(vals_matrix)

			print('====inv_eigenvalue matrix=====')
			inv_vals_matrix = np.linalg.inv(vals_matrix)
			print(inv_vals_matrix)

			print('====eigenvalue=====')
			vals = np.expand_dims(vals,axis=1)
			print(vals)

			# divide index for positive eigenvalue and negative eigenvalue
			index = np.where(vals)[0]
			print('====index eigevalue=====')
			print(index)
			index_positive = np.where(vals>0)[0]
			print('====index positive eigevalue=====')
			print(index_positive)
			print(vals[index_positive])
			index_negative = np.where(vals<0)[0]
			print('====index negative eigevalue=====')
			print(index_negative)
			print(vals[index_negative])

			vals_positive = vals[index_positive]
			vals_negative = vals[index_negative]

			# all method returns True when all elements in the given iterable are true.
			#if vals_positive.all():
			# print(vals_positive)
			#if vals_negative.all():
			# print(vals_negative)

			#vals = vals.reshape(vals.shape[0])
			print('====eigenvector matrix u====')
			print(vecs)
			print('u1',vecs[:,0])
			print('u2',vecs[:,1])
			print('====eigenvector matrix inv====')
			inv_vecs = np.linalg.inv(vecs)
			print(inv_vecs)

			#symmetric matrix
			print('====symmetric matrix H====')
			print(np.dot(vecs,np.dot(vals_matrix,inv_vecs)))

			print('====grad_U(q(0))====')
			grad_potential = V.reshape(-1,1)
			print(grad_potential)
			print('====q(0)====')
			q0 = q.reshape(-1,1)
			print(q0)
			print("====q'(0)====")
			q0_diff = p.reshape(-1,1) # q'(0) = p
			print(q0_diff)
			#print('====a====')
			a = - grad_potential + np.dot(H,q0)
			#print(a)

			# Inital value
			# z(0) = -u^-1 grad_potential
			z0 = -np.dot(inv_vecs, grad_potential)
			print('====z(0)====')
			print(z0)

			# z'(0) = -inv_vals_matrix inv_vecs q'(0)
			z0_diff = - np.dot( vals_matrix ,np.dot( inv_vecs , q0_diff ))
			print("====z'(0)====")
			print(z0_diff)

			#t = 0.075
			#t = np.linspace(0,0.1,9)
			t =  time_step
			print("====give t====")
			print(t)

			# formal solution
			# coefficient c1 = z0 , c2 = z'(0) / sqrt(vals)
			# z = (z1,z2)
			z = np.zeros((z0[:newaxis]+t).shape)
			z_diff = np.zeros((z0[:newaxis]+t).shape)

			if vals_positive.all():

				z[index_positive] = z0[index_positive] * np.cos(np.sqrt(vals[index_positive])*t) + 1. / np.sqrt(vals[index_positive]) * z0_diff[index_positive] * np.sin(np.sqrt(vals[index_positive])*t)
				print('====z for positivie eigenvalue at t={}===='.format(t))
				print(z[index_positive])

				z_diff[index_positive] = -z0[index_positive] * np.sqrt(vals[index_positive]) * np.sin(np.sqrt(vals[index_positive])*t) +  z0_diff[index_positive] * np.cos(np.sqrt(vals[index_positive])*t)

			if vals_negative.all():

				beta = -vals_negative
				print('beta',beta)
				z[index_negative] = (z0[index_negative] * np.sqrt(beta) + z0_diff[index_negative])/ (2*np.sqrt(beta)) * np.exp(np.sqrt(beta)*t) + (z0[index_negative] * np.sqrt(beta) - z0_diff[index_negative])/ (2*np.sqrt(beta)) * np.exp(-np.sqrt(beta)*t)
				print('====z for negative eigenvalue at t={}===='.format(t))
				print(z[index_negative])

				z_diff[index_negative] = (z0[index_negative] * np.sqrt(beta) + z0_diff[index_negative])/ 2 * np.exp(np.sqrt(beta)*t) - (z0[index_negative] * np.sqrt(beta) - z0_diff[index_negative])/ 2 * np.exp(-np.sqrt(beta)*t)

			print('====z====')
			print(z)

			# q = -inv_H grad_potential + q0 - u inv_vals_matrix z
			# q = (q1,q2)
			q = - np.dot(np.dot(vecs,np.dot(inv_vals_matrix,inv_vecs)), grad_potential) + q0 - np.dot(vecs, np.dot(inv_vals_matrix , z))
			p = - np.dot(vecs, np.dot(inv_vals_matrix,z_diff))
			print('====q====')
			print(q)
			q_list = q.reshape(-1,particle,particle)
			print(q_list)
			print('====p====')
			print(p)
			p_list = p.reshape(-1,particle,particle)
			print(p_list)

			self._configuration['phase_space'].set_q(q_list)
			self._configuration['phase_space'].set_p(p_list)




