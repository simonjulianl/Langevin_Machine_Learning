import json

from ML_config import ML_config

if __name__=='__main__':

    data = { }

    # data['chk_pt_file'] = ML_config.chk_pt_file
    data['train_filename'] = ML_config.train_filename
    data['write_chk_pt_filename'] = ML_config.write_chk_pt_filename
    data['write_loss_filename'] = ML_config.write_loss_filename
    data['train_pts'] = ML_config.train_pts
    data['seed'] = ML_config.seed
    data['lr'] = ML_config.lr
    data['nepoch'] = ML_config.nepoch
    data['batch_size'] = ML_config.batch_size
    data['MLP_input'] = ML_config.MLP_input
    data['MLP_nhidden'] = ML_config.MLP_nhidden
    data['activation'] = ML_config.activation

    with open('ML_config.tmpl', 'w') as outfile:
        json.dump(data, outfile,indent=4)


