"""Phase 2a - Parcel prioritisation by sorting.

Sort key: (deadline ASC, priority DESC, weight ASC, parcel_id ASC)
  - earliest deadline first (EDF) is the natural courier rule,
  - ties broken by higher priority, then lighter parcel, then id (deterministic).

Four comparison sorts are implemented from scratch so they can be compared:

  Algorithm       Best        Average     Worst       Stable  Extra space
  Merge sort      n log n     n log n     n log n     yes     O(n)
  Quick sort      n log n     n log n     n^2         no      O(log n)
  Heap sort       n log n     n log n     n log n     no      O(1)
  Insertion sort  n           n^2         n^2         yes     O(1)

Merge sort is the CHOSEN algorithm: guaranteed O(n log n) and stable (parcels with
equal keys keep their manifest order, which matters to dispatch staff).
"""
from __future__ import annotations

import random
from typing import Callable, List, Sequence, Tuple, TypeVar

from .models import Parcel

T = TypeVar("T")


def parcel_key(p: Parcel) -> Tuple[int, int, int, str]:
    return (p.deadline, -p.priority, p.weight, p.parcel_id)


# ---------------------------------------------------------------- merge sort
def merge_sort(items: Sequence[T], key: Callable[[T], object] = lambda x: x) -> List[T]:
    """Top-down stable merge sort. Returns a new list."""
    arr = list(items)
    if len(arr) <= 1:
        return arr
    keys = [key(a) for a in arr]           # compute each key once
    idx = list(range(len(arr)))
    buf = [0] * len(arr)

    def _sort(lo: int, hi: int) -> None:   # sorts idx[lo:hi]
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        _sort(lo, mid)
        _sort(mid, hi)
        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if keys[idx[j]] < keys[idx[i]]:   # strict < keeps it stable
                buf[k] = idx[j]; j += 1
            else:
                buf[k] = idx[i]; i += 1
            k += 1
        while i < mid:
            buf[k] = idx[i]; i += 1; k += 1
        while j < hi:
            buf[k] = idx[j]; j += 1; k += 1
        idx[lo:hi] = buf[lo:hi]

    _sort(0, len(arr))
    return [arr[i] for i in idx]


# ---------------------------------------------------------------- quick sort
def quick_sort(items: Sequence[T], key: Callable[[T], object] = lambda x: x) -> List[T]:
    """Iterative quicksort, random pivot, 3-way partition (handles duplicates)."""
    arr = list(items)
    keys = [key(a) for a in arr]
    rng = random.Random(42)
    stack = [(0, len(arr) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue
        p = rng.randint(lo, hi)
        pivot = keys[p]
        lt, i, gt = lo, lo, hi
        while i <= gt:
            if keys[i] < pivot:
                arr[lt], arr[i] = arr[i], arr[lt]; keys[lt], keys[i] = keys[i], keys[lt]
                lt += 1; i += 1
            elif pivot < keys[i]:
                arr[gt], arr[i] = arr[i], arr[gt]; keys[gt], keys[i] = keys[i], keys[gt]
                gt -= 1
            else:
                i += 1
        stack.append((lo, lt - 1))
        stack.append((gt + 1, hi))
    return arr


# ---------------------------------------------------------------- heap sort
def heap_sort(items: Sequence[T], key: Callable[[T], object] = lambda x: x) -> List[T]:
    arr = list(items)
    keys = [key(a) for a in arr]
    n = len(arr)

    def sift_down(start: int, end: int) -> None:
        root = start
        while 2 * root + 1 <= end:
            child = 2 * root + 1
            if child + 1 <= end and keys[child] < keys[child + 1]:
                child += 1
            if keys[root] < keys[child]:
                arr[root], arr[child] = arr[child], arr[root]
                keys[root], keys[child] = keys[child], keys[root]
                root = child
            else:
                return

    for start in range(n // 2 - 1, -1, -1):
        sift_down(start, n - 1)
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        keys[0], keys[end] = keys[end], keys[0]
        sift_down(0, end - 1)
    return arr


# ---------------------------------------------------------------- insertion sort
def insertion_sort(items: Sequence[T], key: Callable[[T], object] = lambda x: x) -> List[T]:
    arr = list(items)
    keys = [key(a) for a in arr]
    for i in range(1, len(arr)):
        a, k = arr[i], keys[i]
        j = i - 1
        while j >= 0 and k < keys[j]:
            arr[j + 1] = arr[j]; keys[j + 1] = keys[j]
            j -= 1
        arr[j + 1] = a; keys[j + 1] = k
    return arr


SORTERS = {
    "merge": merge_sort,
    "quick": quick_sort,
    "heap": heap_sort,
    "insertion": insertion_sort,
}


def prioritise(parcels: Sequence[Parcel], algorithm: str = "merge") -> List[Parcel]:
    """Return parcels in dispatch order (most urgent first)."""
    return SORTERS[algorithm](parcels, key=parcel_key)
