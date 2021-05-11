import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

class show_graph:

    @staticmethod
    def u_fluctuation(e, temp, nsample, mode):
        '''plot of potential energy at different temp or combined temp for nsamples ( train/ valid ) '''

        plt.title('mc steps appended to nsamples given at T={}'.format(temp),fontsize=15)
        plt.plot(e,'k-', label = 'potential energy using {} samples for {}'.format(nsample, mode))
        plt.xlabel('mcs',fontsize=20)
        plt.ylabel(r'$U_{ij}$',fontsize=20)
        plt.legend()
        plt.show()

    @staticmethod
    def compare2energy(e1, e2, temp, nsample1, nsample2):

        plt.title('mc steps appended to nsamples given at T={}'.format(temp),fontsize=15)
        plt.plot(e1,'k-', label = 'potential energy using {} samples for training'.format(nsample1))
        plt.plot(e2,'r-', label = 'potential energy using {} samples for training'.format(nsample2))
        plt.xlabel('mcs',fontsize=20)
        plt.ylabel(r'$U_{ij}$',fontsize=20)
        plt.legend()
        plt.show()

    @staticmethod
    def u_distribution4nsamples(u1, u2, temp, nparticle, boxsize, nsample1, nsample2):
        '''plot of energy distribution at different temp or combined temp for nsamples ( train/ valid )

        parameter
        -----------
        u1          : potential energy for train data
        u2          : potential energy for valid data
        nsamples1   : the number of train data
        nsamples2   : the number of valid data
        '''

        # plt.xlim(xmin=-5.1, xmax = -4.3)
        fig, ax = plt.subplots()
        plt.hist(u1.numpy(), bins=100, color='k', alpha = 0.5, label = 'train: histogram of {} samples'.format(nsample1))
        plt.hist(u2.numpy(), bins=100, color='r', alpha = 0.5, label = 'valid: histogram of {} samples'.format(nsample2))
        plt.xlabel(r'$U_{ij}$',fontsize=20)
        plt.ylabel('hist', fontsize=20)
        anchored_text = AnchoredText('nparticle={} boxsize={:.3f} temp={} data split = train {} / valid {}'.format(nparticle, boxsize, temp, nsample1, nsample2), loc= 'upper left', prop=dict(fontweight="normal", size=12))
        ax.add_artist(anchored_text)

        plt.legend()
        plt.show()

