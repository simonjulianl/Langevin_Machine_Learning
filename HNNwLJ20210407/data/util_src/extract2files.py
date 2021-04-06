import torch

def check_tau(tS1, tL1, tS2, tL2):

    # one of tau negative
    if tS1<0 and tL1<0: return True
    if tS2<0 and tL2<0: return True

    # tau's are equal
    if abs(tS1 - tS2) < 1e-5 and abs(tL1 - tL2) < 1e-5:
        return True

    return False


def extract2files(infile1, infile2):

    # given 2 files, extract the data and check that tau_long tau_short 
    # are consistent, return
    # qp1_list, qp2_list, tau_long, tau_short
    # qp1_list.shape = [nsamples, (q,p), trajectory, nparticles, DIM]
    # qp2_list.shape = [nsamples, (q,p), trajectory, nparticles, DIM]
 
    data1 = torch.load(infile1)
    data2 = torch.load(infile2)

    tau_short1 = data1['tau_short']
    tau_long1  = data1['tau_long']
    tau_short2 = data2['tau_short']
    tau_long2  = data2['tau_long']

    assert check_tau(tau_short1, tau_long1, tau_short2, tau_long2), 'error in tau'

    target_tau_short = max(tau_short1, tau_short2)
    target_tau_long  = max(tau_long1 ,tau_long2)

    qp1 = data1['qp_trajectory']
    qp2 = data2['qp_trajectory']

    return qp1, qp2, target_tau_short, target_tau_long


def save_to(outfile, qp_combine, tau_short, tau_long):
    data_combine = {'qp_trajectory' : qp_combine, 'tau_short' : tau_short, 'tau_long' : tau_long}
    torch.save(data_combine, outfile)
