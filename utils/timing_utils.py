import time

def hesapla_sure(func, *args, **kwargs):
    start = time.time()
    results = func(*args, **kwargs)
    end = time.time()
    sure = round(end - start, 2)
    return results, sure
