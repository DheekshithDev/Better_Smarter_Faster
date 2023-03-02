import math
import numpy as np
import random
import heapq as heap
import pickle as pkl


def BFS(node_from, node_to_go_to):
    cur_node_pos = node_from
    dists = {}
    for h in main_graph[cur_node_pos]:
        unchecked_nodes = [[h]]
        checked_nodes = []
        heap.heapify(checked_nodes)
        checked_nodes.append(cur_node_pos)
        while len(unchecked_nodes) != 0:
            m = unchecked_nodes.pop(0)
            last_node = m[-1]
            if last_node == node_to_go_to:
                path = list(m)
                dists[h] = path
                unchecked_nodes = []
                break
            for n in main_graph[last_node]:
                if n in checked_nodes:
                    continue
                path = list(m)
                path.append(n)
                unchecked_nodes.append(path)
                if n == node_to_go_to:
                    dists[h] = path
                    unchecked_nodes = []
                    break
            checked_nodes.append(last_node)
    return dists


load_env = open('C:\Users\Arthur King\PycharmProjects\520_INTROAI_P3\Env', 'rb')
main_graph = pkl.load(load_env)

load_u = open('C:\Users\Arthur King\PycharmProjects\520_INTROAI_P3\u_vals', 'rb')
u_str = pkl.load(load_u)

u_sets = [u_str[(i1, i2, i3)]
          for i1 in range(50)
          for i2 in range(50)
          for i3 in range(50)]

all_sets_list = [[i1, i2, i3]
                 for i1 in range(50)
                 for i2 in range(50)
                 for i3 in range(50)]
u_sets_t = u_sets

u_sets_t = np.expand_dims(u_sets_t, axis=1)

x1 = []
x2 = []
for i, j in enumerate(u_sets_t):
    if j < 30:
        x1.append(j)
        x2.append(u_sets_t[i])

x2 = np.array(x2)
x1 = np.array(x1)


class Network:

    def __init__(self, x2, x1):
        self.sets_t = None
        self.ND_1 = np.random.randn(36, 5) * 0.01
        self.ND_2 = np.random.randn(1, 36) * 0.01
        self.n_1 = np.zeros((36, 1))
        self.n_2 = np.zeros((1, 1))
        self.x2 = np.transpose(x2)
        self.x1 = np.transpose(x1)

    def run_func(self):
        l1 = np.mean(self.x1 - self.sets_t)
        loss = ((l1 ** 2) * 0.5)
        # print('Loss = ', loss)
        print('Am I running?') # Checker

    def tol_tot_main(self, phi_1):
        phi_1[phi_1 >= 0] = 1
        phi_1[phi_1 < 0] = 0.01
        return phi_1

    def tol(self, all_sets_list):
        return np.maximum(all_sets_list, 0)

    def tol_tot(self, all_sets_list):
        all_sets_list[all_sets_list < 0] = 1 * 0.01 * all_sets_list[all_sets_list < 0]
        return all_sets_list

    def tol_main(self, phi_1):
        return (phi_1 > 0) * 1 * 1.0

    def forth(self):
        self.phi_1 = np.dot(self.W1, self.x2) + self.b1
        self.all_sets_list = self.tol(self.phi_1)
        self.phi_2 = np.dot(self.W2, self.all_sets_list) + self.b2
        self.phi_2 = self.phi_2

    def reverse(self):
        mn = self.x1.shape[1]
        entei_2 = (self.sets_t - self.x1)
        dum_2 = (1 / mn) * (np.dot(entei_2, self.all_sets_list.T))
        sm_2 = (1 / mn) * np.sum(entei_2)
        entei_1 = np.multiply(np.dot(self.ND_2.T, entei_2), self.tol_main(self.phi_1))
        dum_1 = (1 / mn) * (np.dot(entei_1, self.x2.T))
        # print('Is it working ?') FIXED
        sm_1 = (1 / mn) * (np.sum(entei_1))
        self.W1 = self.ND_1 - 0.001 * dum_1
        self.b1 = self.n_1 - 0.001 * sm_1
        self.W2 = self.ND_2 - 0.001 * dum_2
        self.b2 = self.n_2 - 0.001 * sm_2
        # print('crct vals = ', dum_1)
        print('vals = ', sm_1, sm_2)
        #print('rr = ', dum2)
        return dum_1, sm_1, dum_2, sm_2


# RUN
run = Network(x2, x1)
run.run_func()
