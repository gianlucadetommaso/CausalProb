#!/usr/bin/env python


class Graph:
    def __init__(self, edges):
        self.variables = set()
        for (p, c) in edges:
            self.variables.update((p, c))

        self.children = {v: [] for v in self.variables}
        self.parents = {v: [] for v in self.variables}
        for (p, c) in edges:
            self.parents[c].append(p)
            self.children[p].append(c)


def set_visit_time(v, graph, visited, times, t) -> int:
    visited[v] = True

    for c in graph.children[v]:
        if not visited[c]:
            t = set_visit_time(c, graph, visited, times, t)

    times[v] = t
    t += 1

    return t


def is_dag(graph):
    visited = {v: False for v in graph.variables}
    times = {v: None for v in graph.variables}
    t = 0

    for v in graph.variables:
        if not visited[v]:
            t = set_visit_time(v, graph, visited, times, t)
    print(times)
    for v in graph.variables:
        for c in graph.children[v]:
            if times[c] >= times[v]:
                return False

    return True


if __name__ == '__main__':

    edges = [('V', 'X'), ('V', 'Y'), ('X', 'Y')]
    graph = Graph(edges)
    if not is_dag(graph):
        raise Exception('The graph is not a DAG.')




