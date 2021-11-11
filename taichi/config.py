import copy
from typing import Iterable, Tuple


class Config:
    """
    A simple wrapper around a dictionary type
    that supports two different ways to retrieve the value
    from the key for convenience & easier code search accessing the attribute.
    A. config.key
    B. config[key]
    Support most of the builtin functions of a dictionary, such as
    len(config)
    for x in config:
    print(config)
    x in config
    for k,v in config.items():
    should be supported

    adapted from https://github.com/MetaMind/eloq_dial/blob/main/config.py
    """

    def __init__(self, dictionary: dict = None):
        if dictionary is None:
            dictionary = dict()
        self._keys = set(dictionary)
        for key, value in dictionary.items():
            self.__setattr__(key, value)

    def __repr__(self):
        d = self.__dict__.copy()
        del d["_keys"]  # remove keys attribute from printing
        return d.__repr__()

    def copy(self):
        """Creates a deep copy of this instance"""
        return copy.deepcopy(self)

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __contains__(self, item) -> bool:
        return item in self._keys

    def __iter__(self) -> Iterable[str]:
        return self._keys.__iter__()

    def items(self) -> Iterable[Tuple]:
        return ((key, self[key]) for key in self)

    def __len__(self) -> int:
        return len(self._keys)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for key, value in other.items():
            if key not in self._keys or value != self[key]:
                return False
        return True
