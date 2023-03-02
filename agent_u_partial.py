import math

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import heapq as heap
import pickle as pkl
import environment as env

def agent_U_par_game(agent_pos, prey_pos, predator_pos):

    iterations = 0
    global choice
    current_agent_pos = agent_pos
    current_prey_pos = prey_pos
    current_predator_pos = predator_pos

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
    try:
        load_u = open('C:\Users\Arthur King\PycharmProjects\520_INTROAI_P3\u_vals', 'rb')
        u_str = pkl.load(load_u)
    except FileNotFoundError as e:
        print(e)

    while iterations <= 500:
        iterations += 1
        # AGENT MOVEMENT
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
                    util = util + pr_guess[current_prey_pos] * u_str[(current_agent_pos, current_prey_pos, current_predator_pos)]
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
        funh = np.vectorize(run_for_probs())
        pr_guess = np.dot(pr_guess, [main_graph[current_prey_pos]])
        pr_guess[current_agent_pos] = 0
        pr_guess = funh(pr_guess, np.sum(pr_guess))
        # print(pr_guess)

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

    final_answer = agent_U_par_game(agent_pos, prey_pos, predator_pos)
    if final_answer == 'A1':
        print('AGENT WON! GAME OVER')
    if final_answer == 'P1':
        print('PREDATOR WON! GAME OVER')
