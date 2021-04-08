
class optimizer:

    def __init__(self, op, lr):

        self.lr = lr
        self.opttype = op
        self.optname = self.opttype.__name__

    def create(self, para):
        return self.opttype(para, self.lr)

    def name(self):
        return self.optname

