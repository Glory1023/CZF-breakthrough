'''LRU Cache'''
from collections import OrderedDict


class LRUCache(OrderedDict):
    '''LRU Cache'''
    def __init__(self, *args, capacity: int, **kwargs):
        self.capacity = capacity
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        super().move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)
        while len(self) > self.capacity:
            super().__delitem__(next(iter(self)))
