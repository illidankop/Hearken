import time

def elapsed_time(f):
    def wrapper(self, *args, **kwargs):
        st_time = time.time()
        value = f(self, *args, **kwargs)
        print(f"{self.__class__.__name__}: {f.__name__}: method elapsed time", time.time() - st_time)
        return value

    return wrapper
