from integrator import methods
import math

class MD_parameters:

    seed = 9372211
    nsamples = 6 # total train/valid 20900 / when predict, set 20900 to load file / set 1000 when noML
    nsamples_batch = 10  # num. of batch for nsamples when prepare data before ML
    nsamples_ML = 1
    epsilon = 1.
    sigma = 1.
    mass = 1
    temp_list = [0.04]
    tau_short = 0.001  # short time step for label
    tau_long = 0.1
    nstack = 1  # 100 th step at short time step paired with first large time step
    tau_pair = int(tau_long / tau_short) # n iterations of short time step paired w large time step
    max_ts = 10. # use for predict more iterations
    iteration_batch = 1   #  use linear_integrator; setting 1 : train/valid,  int(max_ts ) : test or more iteration for gold standard

    crash_duplicate_ratio = 0.4 # use data_io
    integrator_method = methods.linear_velocity_verlet

    # optical flow
    pixels_batch = 32
    npixels = 32
    ML_iteration_batch = 1000 # the num. of saved files when test  1000 /
