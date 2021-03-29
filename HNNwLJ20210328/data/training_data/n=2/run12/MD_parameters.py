
class MD_parameters:

    data_path = 'data/training_data/n=2/run12'
    nsamples = 2 # total train/valid 20900 / when predict, set 20900 to load file / set 1000 when noML
    nsamples_batch = 2  # num. of batch for nsamples when prepare data before ML
    nsamples_ML = 1
    epsilon = 1.
    sigma = 1.
    mass = 2
    temp_list = [0.04, 0.32]

