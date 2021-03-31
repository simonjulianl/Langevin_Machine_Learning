import glob

nparticle = 4
# io varaiables
root_path = '../../init_config/n{}/'.format(nparticle)
filename = glob.glob( root_path + 'run*/nparticle{}_new_nsim_rho0.1_T*_pos_test_sampled.pt'.format(nparticle))
print(filename)
