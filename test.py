class A(object):
    def __init__(self):
        pass

    def __iter__(self, i):
        self.i = i

    def next(self):
        print self.i
