import numpy as np
from typing import *


# def my_func(a: (float | int | None) = 100):
#     a = 100 if a is None else a
#     return a

# def my_func(a: Union[float, int, None] = 100):
#     a = 100 if a is None else a
#     return a


def my_func(a: Optional[Union[float, int]] = 100) -> float:
    a = 100 if a is None else a
    return a


def my_func2(a: Optional[int] = 100) -> int:
    a = 100 if a is None else a
    return a


class MyClass:
    pass

def my_func3(lst: Sequence[MyClass]):
    return len(lst)


if __name__ == '__main__':
    # print(my_func(1, 2))

    print(my_func3(set([1, 2., 3, 4])))
