
class base:

    def dHdq1(self):

        print('call base dHdq1')

class derived(base):

    def dHdq1(self):
        print('call derived dHdq1')

if __name__ == '__main__':

    d = derived()

    d.dHdq1()
