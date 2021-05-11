import os
import sys
import json

class ML_config:

    data_dir             = '../data/gen_by_ML/'      # filename to save checkpoints

    train_pts      = 2   # selected number of nsamples
    valid_pts      = 2   # selected number of nsamples
    seed = 9372211

    # optimizer parameters
    lr = 0.0001
    lr_decay_step = 100
    lr_decay_rate = 0.9
    nepoch = 3
    batch_size = 2

    # MLP network parameters
    MLP_input = 5
    MLP_nhidden = 128
    activation = 'tanh'

    # CNN network parameters
    gridL         = 64             # HK
    cnn_channels  = 8		   # HK

if __name__=='__main__':


    argv = sys.argv

    if len(argv) != 5:
        print('usage <programe> <train_dir+filename> <valid_dir+filename> <basename> <json_dir>')
        quit()

    train_filename   = argv[1]
    valid_filename   = argv[2]
    basename         = argv[3]
    json_dir         = argv[4]

    ML_chk_pt_filename = 'None'    # if None then do not load model
    #chk_pt_file = chk_pt_path + filename  # load model

    
    ML_data_dir           = ML_config.data_dir + basename
    write_chk_pt_filename = ML_data_dir + '/' + basename + '_'
    write_loss_filename   = ML_data_dir + '/' + basename + '_loss.txt'

    data = { }

    data['ML_chk_pt_filename']      = ML_chk_pt_filename
    data['train_filename']          = train_filename
    data['valid_filename']          = valid_filename
    data['write_chk_pt_filename']   = write_chk_pt_filename
    data['write_loss_filename']     = write_loss_filename
    data['train_pts']               = ML_config.train_pts
    data['valid_pts']               = ML_config.valid_pts
    data['seed']                    = ML_config.seed
    data['lr']                      = ML_config.lr
    data['lr_decay_step']           = ML_config.lr_decay_step
    data['lr_decay_rate']           = ML_config.lr_decay_rate
    data['nepoch']                  = ML_config.nepoch
    data['batch_size']              = ML_config.batch_size
    data['MLP_input']               = ML_config.MLP_input
    data['MLP_nhidden']             = ML_config.MLP_nhidden
    data['activation']              = ML_config.activation
    data['gridL']                   = ML_config.gridL
    data['cnn_channels']            = ML_config.cnn_channels

    json_dir_name = json_dir +  basename

    if not os.path.exists(json_dir_name):
        os.makedirs(json_dir_name)

    json_file_name = json_dir_name + '/ML_config.dict'
    with open(json_file_name, 'w') as outfile:
        json.dump(data, outfile,indent=4)


