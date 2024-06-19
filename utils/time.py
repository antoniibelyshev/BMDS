from time import time


def timeit(fun):
    def inner(*args, **kwargs):
        t1 = time()
        res = fun(*args, **kwargs)
        t2 = time()
        print(fun.__name__, 'runtime:', t2 - t1, 's')
        return res
    return inner