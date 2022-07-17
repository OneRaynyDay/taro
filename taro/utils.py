from typing import Callable, Dict, Generic, List, TypeVar

import attr
from typing_extensions import Self

K = TypeVar("K")
V = TypeVar("V")


@attr.s
class ScopedDict(Generic[K, V]):
    _dicts: List[Dict[K, V]] = attr.ib(init=False, factory=list)
    _dict_factory: Callable[[], Dict[K, V]] = attr.ib(default=dict)
    append: bool = attr.ib(default=True)

    def __attrs_post_init__(self) -> None:
        # Add a global dict
        self._dicts.append(self._dict_factory())

    def __enter__(self) -> Self:
        self._dicts.append(self._dict_factory())
        return self

    def __getitem__(self, k: K) -> V:
        if not self.append:
            return self._dicts[-1][k]

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
        if not self.append:
            return self._dicts[-1]
        res = {}
        for d in self._dicts:
            res = {**res, **d}
        return res

    def __exit__(self, type, value, traceback):
        self._dicts.pop()
