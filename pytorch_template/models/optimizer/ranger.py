from typing import Iterator

from .lookahead import Lookahead
from .radam import RAdam


def Ranger(params: Iterator, alpha: float = 0.5, k: int = 6, *args, **kwargs):
    radam = RAdam(params, *args, **kwargs)
    ranger = Lookahead(radam, alpha, k)
    return ranger
