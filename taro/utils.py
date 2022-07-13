from typing import Dict, Generic, List, TypeVar

import attr
from typing_extensions import Self

K = TypeVar("K")
V = TypeVar("V")


@attr.s
class ScopedDict(Generic[K, V]):
    _dicts: List[Dict[K, V]] = attr.ib(init=False, factory=list)

    def __attrs_post_init__(self) -> None:
        # Add a global dict
        self._dicts.append({})

    def __enter__(self) -> Self:
        self._dicts.append({})
        return self

    def __getitem__(self, k: K) -> V:
        for d in reversed(self._dicts):
            if k in d:
                return d[k]
        raise KeyError(f"Missing key {k}")

    def __setitem__(self, k: K, v: V) -> None:
        self._dicts[-1][k] = v

    def __contains__(self, k: K) -> bool:
        try:
            self[k]
            return True
        except KeyError:
            return False

    def to_dict(self) -> Dict[K, V]:
        res = {}
        for d in self._dicts:
            res = {**res, **d}
        return res

    def __exit__(self, type, value, traceback):
        self._dicts.pop()
