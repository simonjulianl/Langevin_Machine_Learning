class A:
    def __init__(self):
        self.terms = []
    def f(self):
        print('call super')


class B(A):
    def __init__(self):
        super().__init__()

    def f(self):
        print('here child class')
        # calling the parent's class
        # super().f() # func in parent class
        return


if __name__ == '__main__':

    obj_b = B()
    print('__main__', obj_b.f())

    u = obj_b.terms
    print('u ',u)
    quit()
    print(A)
    #one method
    super(type(obj_b), obj_b).f()

    # second method
    super(B, obj_b).f()

    # third method
    A.f(obj_b) # func in parent class

    # forth method
    obj_b.__class__.__bases__[0].f(obj_b)
    # obj_b.super().f()
