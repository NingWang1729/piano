import time
from contextlib import contextmanager
from datetime import datetime


@contextmanager
def time_code(label='Code Block'):
    print(f'\n[{label}] start @: {datetime.now().strftime("%m_%d_%Y_%H:%M:%S")}', flush=True)
    start = time.time()
    yield
    end = time.time()
    min_, sec_ = int((end - start) / 60), int((end - start) % 60)
    print(f'[{label}] runtime: {min_} min {sec_} sec', flush=True)
