import math
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import heapq as heap
import pickle as pkl
import environment as env


class Network:

    def __init__(self, x2, x1):
        self.sets_t = None
        self.ND_1 = np.random.randn(36, 5) * 0.01
        self.ND_2 = np.random.randn(1, 36) * 0.01
        self.n_1 = np.zeros((36, 1))
        self.n_2 = np.zeros((1, 1))
        self.x2 = np.transpose(x2)
        self.x1 = np.transpose(x1)
    #
    # def run_func(self):
    #     l1 = np.mean(self.x1 - self.sets_t)
    #     loss = ((l1 ** 2) * 0.5)

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
        sm_1 = (1 / mn) * (np.sum(entei_1))
        self.W1 = self.ND_1 - 0.001 * dum_1
        self.b1 = self.n_1 - 0.001 * sm_1
        self.W2 = self.ND_2 - 0.001 * dum_2
        self.b2 = self.n_2 - 0.001 * sm_2
        return dum_1, sm_1, dum_2, sm_2

def agent_V_par_game(agent_pos, prey_pos, predator_pos, temp_dict=None,
                     belief_states_prey=None, belief_states=None, head_nodes=None, t=None):

    global choice
    current_agent_pos = agent_pos
    current_prey_pos = prey_pos
    current_predator_pos = predator_pos
    agent_survey_node_loc = 0
    predator_initial_found_loc = 0
    uncalculated_nodes = []

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

    # START
    # SAME EXCEPTION
    load_u = open('C:\Users\Arthur King\PycharmProjects\520_INTROAI_P3\u_vals', 'rb')
    u_str = pkl.load(load_u)

    temp_dict = defaultdict(list)
    temp_dict_2 = {}

    # AGENT MOVEMENT
    print('belief_states 1 = ', belief_states)
    print('belief_stats_SUM = ', sum(belief_states.values()))
    highest_prob_val_node_list = []
    prob_move_to_agent = 0.6
    prob_move_random = 0.4
    if t > 1:
        if 1 in belief_states.values():
            print('Did I come here? 2')
            head_nodes.append(predator_initial_found_loc)

        head_nodes = list(dict.fromkeys(head_nodes))

        for i in head_nodes:
            sorted_pred_to_agent_vals = []
            shortest_to_agent_frm_predator = BFS(i, current_agent_pos)
            print('shortest_to_agent_frm_predator = ', shortest_to_agent_frm_predator)
            for s1 in sorted(shortest_to_agent_frm_predator, key=lambda k: len(shortest_to_agent_frm_predator[k]),
                             reverse=False):
                sorted_pred_to_agent_vals.append(s1)

            best_path_to_agent = sorted_pred_to_agent_vals[0]

            print('sorted_pred_to_agent_vals = ', sorted_pred_to_agent_vals)

            for j in main_graph[i]:
                uncalculated_nodes.append(j)
            for k in uncalculated_nodes:
                if len(main_graph[i]) == 3:
                    if current_agent_pos in uncalculated_nodes:
                        temp_dict[k].append((belief_states[i] * (1 / 2)))
                    elif current_agent_pos not in uncalculated_nodes:
                        if k == best_path_to_agent:
                            temp_dict[k].append((belief_states[i] * prob_move_to_agent))
                            continue
                        temp_dict[k].append((belief_states[i] * prob_move_random))
                else:
                    if current_agent_pos in uncalculated_nodes:
                        temp_dict[k].append((belief_states[i] * 1))
                    elif current_agent_pos not in uncalculated_nodes:
                        if k == best_path_to_agent:
                            temp_dict[k].append((belief_states[i] * prob_move_to_agent))
                            continue
                        temp_dict[k].append((belief_states[i] * (prob_move_random * 2)))

            uncalculated_nodes.clear()
            belief_states[i] = 0

        for lk in temp_dict.keys():
            if len(temp_dict[lk]) == 1:
                belief_states[lk] = temp_dict[lk][-1]
            else:
                belief_states[lk] = sum(temp_dict[lk])

        belief_states[current_agent_pos] = 0
        uncalculated_nodes.clear()

        for i in head_nodes:
            for j in main_graph[i]:
                uncalculated_nodes.append(j)

        if current_agent_pos in uncalculated_nodes:
            [uncalculated_nodes.remove(r) for r in uncalculated_nodes if r == current_agent_pos]
        temp_dict.clear()
        highest_prob_val_node_list.clear()
        for p2 in sorted(belief_states, key=lambda k: belief_states[k], reverse=True):
            highest_prob_val_node_list.append(p2)
        max_prob = belief_states[highest_prob_val_node_list[0]]
        print('max_prob = ', max_prob)
        highest_prob_val_node = highest_prob_val_node_list[0]
        agent_survey_node_loc = highest_prob_val_node

    elif 1 not in belief_states.values():
        print('1 I came here')
        er_check = list(belief_states.keys())
        er_check.remove(current_agent_pos)
        agent_survey_node_loc = random.choice(er_check)

    print('agent_survey_node_loc = ', agent_survey_node_loc)

    if agent_survey_node_loc != current_predator_pos:
        if t > 1:
            print('Uncalculated Nodes 2 = ', uncalculated_nodes)
            print('agent_survey_node_loc 2 = ', agent_survey_node_loc)

            [uncalculated_nodes.remove(r) for r in uncalculated_nodes if r == agent_survey_node_loc]

            for j in uncalculated_nodes:
                temp_dict_2[j] = belief_states[j] / (1 - belief_states[agent_survey_node_loc])

            head_nodes.clear()
            for add in uncalculated_nodes:
                head_nodes.append(add)
            uncalculated_nodes.clear()

            for lk in temp_dict_2.keys():
                belief_states[lk] = temp_dict_2[lk]

            belief_states[agent_survey_node_loc] = 0

            temp_dict_2.clear()

            print('belief_states 44 =', belief_states)
            print('belief_stats_SUM_44 = ', sum(belief_states.values()))

    elif agent_survey_node_loc == current_predator_pos:
        print('I found predator here once!')
        head_nodes.clear()
        uncalculated_nodes.clear()
        for i in belief_states.keys():
            if i == agent_survey_node_loc:
                belief_states[i] = 1
                continue
            belief_states[i] = 0

    sorted_prey_vals = []
    # Find prey dist to agent here
    total_cost_to_prey_path = BFS(current_agent_pos, current_prey_pos)
    for k2 in sorted(total_cost_to_prey_path, key=lambda k: len(total_cost_to_prey_path[k]), reverse=False):
        sorted_prey_vals.append(k2)
    current_agent_pos = current_agent_pos

    u_str_par = dict()
    pr_guess = 50 * [(1 / 49)]
    pr_guess[current_agent_pos] = 0
    pr_guess = np.array(pr_guess)

    def run_for_probs(nod_n, tot_sum):
        if tot_sum == 0:
            return 0
        else:
            ng = nod_n / tot_sum
            return ng

    current_predator_pos = current_predator_pos
    current_prey_pos = current_prey_pos

    temp_ar = np.where(np.isclose(pr_guess, np.amax(pr_guess)))[0]
    fh = np.random.choice(temp_ar)

    if fh != current_prey_pos:
        funh = np.vectorize(run_for_probs())
        pr_guess[fh] = 0
        pr_guess = funh(pr_guess, np.sum(pr_guess))

        array = np.where(np.isclose(pr_guess, np.amax(pr_guess)))[0]
        choice = np.random.choice(array)
    else:
        pr_guess.fill(0)
        pr_guess[fh] = 1

    belief_s = pr_guess

    neigh_ag = [k for k in main_graph[current_agent_pos]]

    neigh_pd = [k for k in main_graph[current_predator_pos]]

    for i in neigh_ag:
        for j in neigh_pd:
            util = 0
            for p in range(1, len(pr_guess)):
                util = util + pr_guess[current_prey_pos] * u_str[
                    (current_agent_pos, current_prey_pos, current_predator_pos)]
            util_got = util
            u_str_par[(i, j)] = (belief_s, util_got)

    tot_avg_list = []
    for i in neigh_ag:
        tot_avg = 0
        for j in neigh_pd:
            tot_avg = tot_avg + u_str_par[(i, j)][1]
        tot_avg_list.append(tot_avg)

    min_u = math.inf
    r_ind = 0
    for j, i in enumerate(tot_avg_list):
        if i < min_u:
            r_ind = j

    current_agent_pos = neigh_ag[r_ind]

    funh = np.vectorize(run_for_probs())
    pr_guess[current_agent_pos] = 0
    pr_guess = funh(pr_guess, np.sum(pr_guess))
    pr_guess = np.dot(pr_guess, [main_graph[current_prey_pos]])
    pr_guess[current_agent_pos] = 0

    if current_agent_pos == current_prey_pos == current_predator_pos:
        print('GAME OVER! EVERYBODY DIED!')
        return None

    if current_agent_pos == current_predator_pos:
        print("PREDATOR WON! AGENT DIED!")
        return 'P1'

    if current_agent_pos == current_prey_pos:
        print('AGENT WON!')
        return 'A1'

    # PREDATOR MOVEMENT
    sorted_agent_vals = []
    total_cost_to_agent_path = BFS(current_predator_pos, current_agent_pos)
    for h1 in sorted(total_cost_to_agent_path, key=lambda k: len(total_cost_to_agent_path[k]), reverse=False):
        sorted_agent_vals.append(h1)
    possible_jump_nodes = main_graph[current_predator_pos]
    possible_jump_nodes.remove(sorted_agent_vals[0])
    possible_jump_nodes.append(sorted_agent_vals[0])
    # print('possible_jump_nodes = ', possible_jump_nodes)
    if len(possible_jump_nodes) == 3:
        rd = random.choices(possible_jump_nodes, [0.2, 0.2, 0.6], k=1)[-1]
    else:
        rd = random.choices(possible_jump_nodes, [0.4, 0.6], k=1)[-1]

    current_predator_pos = rd

    print('current_predator_pos Taken = ', current_predator_pos)

    if current_predator_pos == current_agent_pos:
        print("PREDATOR WON! AGENT DIED!")
        return 'P1'


    # PREY MOVEMENT
    if len((main_graph[current_prey_pos] + [current_prey_pos])) == 4:
        current_prey_pos_list = random.choices((main_graph[current_prey_pos] + [current_prey_pos]),
                                               [25, 25, 25, 25], k=1)
    else:
        current_prey_pos_list = random.choices((main_graph[current_prey_pos] + [current_prey_pos]),
                                               [33.33, 33.33, 33.33], k=1)

    current_prey_got_pos = current_prey_pos_list[-1]

    if current_prey_got_pos == current_agent_pos:
        print('I am not taking this becoz agent there!')
    else:
        current_prey_pos = current_prey_got_pos

    if current_prey_pos == current_agent_pos:
        print("AGENT WON!")
        return 'A1'


if __name__ == '__main__':

    load_env = open('C:\Users\Arthur King\PycharmProjects\520_INTROAI_P3\Env', 'rb')
    main_graph = pkl.load(load_env)

    agent_random_loc = random.choice(list(main_graph.keys()))

    while True:
        prey_random_loc = random.choice(list(main_graph.keys()))
        predator_random_loc = random.choice(list(main_graph.keys()))
        if prey_random_loc == agent_random_loc or predator_random_loc == agent_random_loc:
            continue
        break

    agent_pos = agent_random_loc
    prey_pos = prey_random_loc
    predator_pos = predator_random_loc

    print('current_agent_pos = ', agent_pos)
    print('current_prey_pos = ', prey_pos)
    print('current_predator_pos = ', predator_pos)

    final_answer = agent_V_par_game(agent_pos, prey_pos, predator_pos)
    if final_answer == 'A1':
        print('AGENT WON! GAME OVER')
    if final_answer == 'P1':
        print('PREDATOR WON! GAME OVER')
