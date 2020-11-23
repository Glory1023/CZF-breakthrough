'''Helper functions'''
import time


def timer(wait_time):
    '''Timeout generator'''
    if wait_time:
        deadline = time.time() + wait_time
        while True:
            timeout = deadline - time.time()
            if timeout <= 0:
                break
            yield timeout
