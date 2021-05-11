import sys
import torch

from extract2files import extract2files4show
from extract2files import save_to_show


if __name__=='__main__':
    ''' given 2 files, concatenate data along different condition 
        ( e.g. temperature ) '''
    argv = sys.argv

    infile1 = argv[1]
    infile2 = argv[2]   
    outfile = argv[3]   # newfilename

    Accratio1, Accratio2, spec1, spec2 = extract2files4show(infile1, infile2)
    # Accratio.shape = [1]
    # spec.shape = [1]

    Accratio_combine = torch.cat((Accratio1, Accratio2),dim=0)
    spec_combine = torch.cat((spec1,spec2),dim=0)

    print('Accratio_combine', Accratio_combine.shape)
    print('spec_combine', spec_combine.shape)

    save_to_show(outfile, Accratio_combine, spec_combine)

