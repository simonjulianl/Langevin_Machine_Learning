import torch.optim.lr_scheduler as scheduler

class optimizer:

    def __init__(self, op, lr):

        self.lr = lr
        self.opttype = op
        self.optname = self.opttype.__name__
        print('optimizer initialized : op ',op,' lr ',lr)

    def create(self, para, every_n_epoch, decay_rate):
        print('created ',self.optname,' with lr ',self.lr)
        opt = self.opttype(para, self.lr)
        lr_sch = scheduler.StepLR(opt,every_n_epoch,decay_rate)
        return opt,lr_sch

    def name(self):
        return self.optname

