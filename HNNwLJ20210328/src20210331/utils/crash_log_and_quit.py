import torch

def crash_log_and_quit(q_list, p_list):

    crash_path = '../data/crash_dir/'
    crash_filename = crash_path + 'qp_state_before_crash.pt'

    # write q, p list
    torch.save(torch.stack((q_list, p_list)), crash_filename)

    print('saving qp_state_before_crash and then quit')
    quit()
