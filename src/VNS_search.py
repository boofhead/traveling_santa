import logging
import pickle
from copy import copy
from numpy import arange, zeros, max, min, abs, squeeze, argwhere, isin, inf
from numpy.random import choice

from src.traveling_santa import CityMap

logging.basicConfig(level=logging.INFO)


class Node:
    def __init__(self, id):
        self.id = id
        self.label = self.id
        self.neighbours = []
        self.f_max = None
        self.f_min = None
        self.bandwidth = None
        self.critical_nodes = []

    def update(self):
        labels = zeros(len(self.neighbours), dtype=int)
        for i, node in enumerate(self.neighbours):
            labels[i] = node.label
        self.f_max = max(labels)
        self.f_min = min(labels)
        bandwidth = abs(labels - self.label)
        self.bandwidth = max(bandwidth)
        self.critical_nodes = [self.neighbours[i] for i in argwhere(bandwidth == self.bandwidth)[0]]

    def set_label(self, label):
        self.label = label
        for node in self.neighbours:
            node.update()
        self.update()


class Graph:
    def __init__(self, adj_dict):
        self.nodes = [Node(i) for i in range(len(adj_dict))]
        for i, neighbours in adj_dict.items():
            for node_id in list(neighbours):
                self.nodes[i].neighbours.append(self.nodes[node_id])
            self.nodes[i].update()
        self.labels = arange(len(self.nodes), dtype=int)
        self.bandwidth = None
        self.critical_nodes = []
        self.update_bandwidth()

    def update_bandwidth(self):
        self.bandwidth = 0
        self.critical_nodes = []
        for node in self.nodes:
            if node.bandwidth > self.bandwidth:
                self.bandwidth = node.bandwidth
                self.critical_nodes = [node]
            elif node.bandwidth == self.bandwidth:
                self.critical_nodes.append(node)

    def update_from_labels(self, labels=None):
        if labels:
            self.labels = labels
        for i, node in enumerate(self.nodes):
            node.set_label(self.labels[i])
        self.update_bandwidth()

    def swap_nodes_by_id(self, a, b):
        if a == b:
            return
        label_a = self.labels[a]
        label_b = self.labels[b]
        self.nodes[a].set_label(label_b)
        self.nodes[b].set_label(label_a)
        self.labels[a] = label_b
        self.labels[b] = label_a
        self.update_bandwidth()

    def ids_from_labels(self, labels):
        return squeeze(argwhere(isin(labels, self.labels)))


class VariableNeighbourhoodSearch:

    def __init__(self, adj_dict: dict = CityMap().adjacency_dict):
        self.graph = Graph(adj_dict)

    def initial_solution(self):
        level = {self.graph.nodes[0]}
        prev_level = set()
        levels = [level]
        j = 1
        while True:
            next_level = set()
            for node in list(level):
                next_level |= set(node.neighbours)
            next_level -= prev_level
            next_level -= level
            if len(next_level):
                prev_level = level
                level = next_level
                levels.append(level)
                logging.info(f"level {j} filled with {len(level)} nodes")
                j += 1
            else:
                break
        assigned = set()
        i = 0
        for level in levels:
            for node in list(level):
                to_assign = set(node.neighbours) - assigned
                for j in list(to_assign):
                    self.graph.labels[j.id] = i
                    assigned.add(j)
                    i += 1
        self.graph.update_from_labels()

    def shaking(self, k):
        bw_cut = self.graph.bandwidth
        big_k = self.graph.critical_nodes
        while len(big_k) < k:
            bw_cut -= 1
            big_k = [node for node in self.graph.nodes if node.bandwidth >= bw_cut]
        for _ in range(k):
            u = choice(big_k)
            v = choice(u.critical_nodes)
            distance = inf
            swap_id = None
            for w_id in self.graph.ids_from_labels(arange(u.f_min, u.f_max + 1)):
                if w_id == v.id:
                    continue
                w = self.graph.nodes[w_id]
                test = max(w.f_max - v.label, v.label - w.f_min)
                if test < distance:
                    distance = test
                    swap_id = w_id
            self.graph.swap_nodes_by_id(v.id, swap_id)

    def local_search(self):
        can_improve = True
        n_crit = len(self.graph.critical_nodes)
        old_bandwidth = self.graph.bandwidth
        i = 0
        while can_improve:
            logging.info(f"running local opt cycle {i}")
            logging.info(f"bandwidth: {old_bandwidth}")
            i += 1
            can_improve = False
            for node in self.graph.critical_nodes:
                u = node.id
                mid = (node.f_min + node.f_max) / 2
                logging.info(f"targeting move from {node.label} to {int(mid)}")
                lim = abs(mid - node.label)
                a = int(mid)
                b = a + 1
                dist = 0
                while dist < lim:
                    if abs(a - mid) < abs(b - mid):
                        test = a
                        a -= 1
                        dist = abs(b - mid)
                    else:
                        test = b
                        b += 1
                        dist = abs(a - mid)
                    v = squeeze(argwhere(self.graph.labels == test))
                    self.graph.swap_nodes_by_id(u, v)
                    if old_bandwidth > self.graph.bandwidth or (
                            old_bandwidth == self.graph.bandwidth and len(self.graph.critical_nodes) < n_crit):
                        logging.info(f"swapping nodes {self.graph.labels[u]} and {self.graph.labels[v]}")
                        n_crit = len(self.graph.critical_nodes)
                        old_bandwidth = self.graph.bandwidth
                        can_improve = True
                        break
                    else:
                        self.graph.swap_nodes_by_id(u, v)

    def move(self, old_labels, old_bandwidth, old_n_crit, alpha):
        if self.graph.bandwidth < old_bandwidth:
            return True
        if self.graph.bandwidth == old_bandwidth and (
                old_n_crit > len(self.graph.critical_nodes) or self.calc_ro(old_labels) > alpha):
            return True
        self.graph.update_from_labels(old_labels)
        return False

    def calc_ro(self, labels):
        return sum(self.graph.labels == labels) - 1

    def optimise(self, k_min=16, k_max=256, k_step=16, alpha=64):
        self.initial_solution()
        self.local_search()
        k = k_min
        while k < k_max:
            logging.info(f"shaking {k} nodes")
            old_bandwidth = self.graph.bandwidth
            old_labels = copy(self.graph.labels)
            old_ncrit = len(self.graph.critical_nodes)
            self.shaking(k)
            self.local_search()
            if self.move(old_bandwidth, old_labels, old_ncrit, alpha):
                k = k_min
            else:
                k += k_step


def main():
    vns = VariableNeighbourhoodSearch()
    vns.optimise()
    with open('../store/labels.pickle', 'wb') as f:
        pickle.dump(vns.graph.labels, f)
    return


if __name__ == "__main__":
    main()
