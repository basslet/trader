def frange(start, stop, step):
    while start < stop:
        yield start
        start += step
