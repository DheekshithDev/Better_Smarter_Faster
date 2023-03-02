import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import heapq as heap
import pickle as pkl
import environment as env

# # Imports
# G = env.G
# agent_pos = env.agent_pos
# prey_pos = env.prey_pos
# predator_pos = env.predator_pos
# color_map = env.color_map
# pos = env.pos

# print('agent_pos = ', agent_pos)
# print('prey_pos = ', prey_pos)
# print('pred_pos = '), predator_pos
#
# nx.draw(G, with_labels=True, pos=pos, node_color=color_map)
# plt.show()

def agent_U_game(agent_pos, prey_pos, predator_pos):

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
    load_u = open('C:\Users\Arthur King\PycharmProjects\520_INTROAI_P3\u_vals', 'rb')
    u_str = pkl.load(load_u)
    # AGENT MOVEMENT
    current_agent_pos = current_agent_pos
    least_u_val = math.inf
    u_str_vals = []
    for u1, u2 in enumerate(main_graph[current_agent_pos]):
        print('Am I coming here? check_1')
        u_str_vals.append((u1, u_str[(u2, current_prey_pos, current_predator_pos)]))

        if u_str[(u2, current_prey_pos, current_predator_pos)] < least_u_val:
            least_u_val = u_str[(u2, current_prey_pos, current_predator_pos)]

            print('am i coming here? check 2')
            node = u2

    current_agent_pos = node

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

    final_answer = agent_U_game(agent_pos, prey_pos, predator_pos)
    if final_answer == 'A1':
        print('AGENT WON! GAME OVER')
    if final_answer == 'P1':
        print('PREDATOR WON! GAME OVER')
