import torch
import os

def crash_log_and_quit(q_list, p_list):
    '''
        returns
        torch.tensor of qp_list
        shape is [2, nsamples,nparticles,DIM]

        shape of q_list is [nsamples,nparticles,DIM]
        shape of p_list is [nsamples,nparticles,DIM]
    '''

    crash_path = '../data/crash_dir/'
    crash_filename = crash_path + 'qp_state_before_crash.pt'

    if os.path.exists(crash_filename):
        # overwrite q, p list
        existed_qp_list = torch.load(crash_filename)
        qp_list   = torch.cat((existed_qp_list, torch.stack((q_list, p_list))), dim=1)

    else:
        qp_list = torch.stack((q_list, p_list))

    print(qp_list.shape)
    torch.save(qp_list, crash_filename)


    print('saving qp_state_before_crash and then quit')
    quit()
