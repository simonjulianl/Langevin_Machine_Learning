import sys
import torch

def check_tau(tS1, tL1, tS2, tL2):

    # one of tau negative
    if tS1<0 and tL1<0: return True
    if tS2<0 and tL2<0: return True

    # tau's are equal
    if abs(tS1 - tS2) < 1e-5 and abs(tL1 - tL2) < 1e-5:
        return True

    return False


if __name__=='__main__':

    argv = sys.argv

    infile1 = argv[1]
    infile2 = argv[2]
    outfile = argv[3]

    # we only take files with same tau_short tau_long
    # or negative values of tau_short tau_long - for MC output
    data1 = torch.load(infile1)
    data2 = torch.load(infile2)

    tau_short1 = data1['tau_short']
    tau_long1  = data1['tau_long']
    tau_short2 = data2['tau_short']
    tau_long2  = data2['tau_long']

    assert check_tau(tau_short1, tau_long1, tau_short2, tau_long2),'error in tau'

    target_tau_short = max(tau_short1, tau_short2)
    target_tau_long  = max(tau_long1 ,tau_long2)

    qp1 = data1['qp_trajectory']
    qp2 = data2['qp_trajectory']
    qp_combine = torch.cat((qp1,qp2))

    data_combine = {'qp_trajectory':qp_combine, 'tau_short': target_tau_short, 'tau_long':target_tau_long}

    torch.save(data_combine, outfile)
