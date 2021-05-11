import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_ACCRatio(ACCRatio):

    t = np.arange(0.03, 0.69, 0.08)
    plt.title('ACCRatio with the temperature')
    plt.plot(t, ACCRatio, 'k-')
    plt.xlabel('T', fontsize=20)
    plt.ylabel(r'$ACCRatio$', fontsize=20)
    plt.legend()
    plt.show()

def plot_cv(spec, nparticle):

    t = np.arange(0.03, 0.69, 0.08)
    plt.title('specific heat with the temperature')
    plt.plot(t, spec, 'k-', label='n={}'.format(nparticle))
    plt.xlabel('T', fontsize=20)
    plt.ylabel(r'$c_{v}$', fontsize=20)
    plt.legend()
    plt.show()

if __name__ == '__main__':

        argv = sys.argv
        filename = argv[1]
        nparticle = argv[2]

        ACCRatio, spec = torch.load(filename)

        print('ACCRatio', ACCRatio)
        print('Spec', spec)

        plot_ACCRatio(ACCRatio)
        plot_cv(spec, nparticle)
