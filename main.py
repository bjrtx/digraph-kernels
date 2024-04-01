"""
Functions for computing the kernels of fuzzy circular interval graphs.
See https://arxiv.org/abs/2202.04476 for background.
"""
import itertools
from functools import cache
from collections.abc import Iterable, Collection, Hashable, Sequence
from typing import TypeVar

import matplotlib.pyplot
import networkx
from more_itertools import all_equal

V = TypeVar('V', bound=Hashable)


def is_adjacent(g: networkx.DiGraph, u: V, v: V):
    return g.has_edge(u, v) or g.has_edge(v, u)


def is_clique(g: networkx.DiGraph, k: Iterable[V]):
    return all(is_adjacent(g, *p) for p in itertools.combinations(k, 2))


def is_homogenous(g: networkx.DiGraph, k: Iterable[V], x: V):
    return all_equal(is_adjacent(g, x, y) for y in k)


def is_homogenous_pair(g: networkx.DiGraph, k1: Iterable[V], k2: Iterable[V]):
    k1, k2 = set(k1), set(k2)
    return all((
        is_clique(g, k1),
        is_clique(g, k2),
        k1.isdisjoint(k2),
        *(
            is_homogenous(g, k1, u)
            and is_homogenous(g, k2, u)
            for u in g
            if u not in k1 and u not in k2
        )
    ))


def reduce(g: networkx.DiGraph, k1, k2):
    assert is_homogenous_pair(g, k1, k2)
    v = list(g.nodes()) + ['x1', 'y1', 'x2', 'y2']
    v = [n for n in v if n not in k1 and n not in k2]
    e = [(a, b) for (a, b) in g.edges if a not in k1 and b not in k1 and a not in k2 and b not in k2]


def validate_nice_flig_model(g: networkx.DiGraph, ordering: Collection[Collection[V]], intervals: Collection[tuple[int, int]]):
    assert set(g.nodes) == set().union(*ordering)
    assert len(g.nodes) == sum(len(s) for s in ordering)
    assert all(is_clique(g, s) for s in ordering)
    assert len(intervals) <= len(g.nodes)
    endpoints = set(itertools.chain.from_iterable(intervals))
    assert all(isinstance(x, int) for x in endpoints)
    assert len(endpoints) == 2 * len(intervals)

    mapping = {p: i for i, s in enumerate(ordering) for p in s}
    for x, y in itertools.combinations(g.nodes, 2):
        if is_adjacent(g, x, y):
            assert any(a <= mapping[x] <= b and a <= mapping[y] <= b for a, b in intervals)
        else:
            assert all(
                {a, b} == {mapping[x], mapping[y]}
                for a, b in intervals
                if a <= mapping[x] <= b and a <= mapping[y] <= b
            )


def flig_kernels(g: networkx.DiGraph, ordering: Sequence[Collection[V]], intervals: Collection[tuple[int, int]]):
    """
    :param g: Input FLIG.
    :param ordering: ordered list of equivalence classes
    :param intervals: List of intervals in the nice FLIG model as pairs of start-end indices
    :return:
    """
    validate_nice_flig_model(g, ordering, intervals)
    mapping = {p: i for i, s in enumerate(ordering) for p in s}
    left_endpoints = [None for _ in ordering]
    for ia, ib in intervals:
        left_endpoints[ib] = ia
    left_vertices = [ordering[ia] if ia is not None else [] for ia in left_endpoints]
    assert not any(left_vertices[mapping[i]] for s in left_vertices for i in s)

    def interval(x: V, y: V, left_open=False, right_open=False):
        ix, iy = mapping[x], mapping[y]
        assert ix <= iy
        for i in range(ix + left_open, iy + 1 - right_open):
            yield from ordering[i]

    @cache
    def selectable(x: V, y: V):
        assert mapping[x] < mapping[y]
        return [
            m
            for m in interval(x, y, left_open=True, right_open=True)
            if not is_adjacent(g, x, m) and not is_adjacent(g, y, m)
               and m not in left_vertices[mapping[y]]
               and absorbs([m, y], interval(m, y, left_open=True))
        ]

    @cache
    def selectable_constrained(x: V, y: V, m: V | None):
        assert mapping[x] < mapping[y]
        assert m in left_vertices[mapping[y]]
        assert not is_adjacent(g, m, x) and not is_adjacent(g, m, y)
        return [
            w
            for w in interval(x, m, left_open=True, right_open=True)
            if not is_adjacent(g, x, w) and not is_adjacent(g, m, w)
               and absorbs((w, m, y), interval(m, y, left_open=True))
        ]

    def absorbs(absorber: Iterable[V], absorbee: Iterable[V]):
        return all(
            any(v == x or g.has_successor(v, x) for x in absorber)
            for v in absorbee
        )

    @cache
    def constrained_flig_kernel_between(x: V, y: V, m: V | None):
        if m is None:
            return [
                k + [y]
                for w in selectable(x, y)
                for k in flig_kernel_between(x, w)
            ]

        else:
            assert m in left_vertices[mapping[y]]
            assert not (is_adjacent(g, x, m) or is_adjacent(g, y, m))

            if absorbs([x, m, y], interval(x, y)):
                return [[x, m, y]]
            else:
                return [
                    k + [m, y]
                    for w in selectable_constrained(x, y, m)
                    for k in flig_kernel_between(x, w)
                ]

    @cache
    def flig_kernel_between(x: V, y: V):
        assert mapping[x] < mapping[y]
        assert not is_adjacent(g, x, y)
        if absorbs([x, y], interval(x, y)):
            return [[x, y]]
        else:
            return list(
                itertools.chain.from_iterable(
                    constrained_flig_kernel_between(x, y, m)
                    for m in left_vertices[mapping[y]]
                    if not is_adjacent(g, m, x) and not is_adjacent(g, m, y)
                )
            ) + list(constrained_flig_kernel_between(x, y, None))

    for c1, c2 in itertools.combinations(ordering, 2):
        for p in itertools.product(c1, c2):
            if not is_adjacent(g, *p):
                for k in flig_kernel_between(*p):
                    if absorbs(k, g.nodes):
                        yield k


if __name__ == '__main__':
    example_nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    example_ordering = [['a', 'b'], [], ['c', 'd'], [], ['e'], [], ['f'], [], [], ['g', 'h', 'i']]
    example_intervals = [(0, 2), (1, 5), (3, 7), (8, 9)]
    example_g = networkx.DiGraph([('a', 'b'), ('a', 'd'), ('c', 'b'), ('c', 'd'), ('c', 'e'), ('d', 'e'), ('e', 'f'),
                                  ('g', 'h'), ('h', 'i'), ('g', 'i')])
    validate_nice_flig_model(example_g, ordering=example_ordering, intervals=example_intervals)
    for k in flig_kernels(example_g, example_ordering, example_intervals):
        print("found", k)
    networkx.draw(example_g, labels={n: n for n in example_nodes})

    matplotlib.pyplot.show()
