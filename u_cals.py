import random
import heapq as heap
import math
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

def cal_u_vals():
    util_actv = {}
    u_str = {}
    x = True

    rewd = 1 # 0.1 Always Total
    dis_penl = 0.0002
    error_m = -1.00
    ct = 0
    for current_agent_pos in range(50):
        for current_prey_pos in range(50):
            for current_predator_pos in range(50):
                u_str[(current_agent_pos, current_prey_pos, current_predator_pos)] = random.randrange(5, 35)
                ct = ct + 1

    # Using Value dis_itr
    for current_agent_pos in range(50):
        for current_prey_pos in range(50):
            for current_predator_pos in range(50):
                if current_agent_pos != current_predator_pos and current_agent_pos == current_prey_pos:
                    u_str[(current_agent_pos, current_prey_pos, current_predator_pos)] = float(0.)

                elif abs(current_prey_pos - current_agent_pos) == 1:
                    u_str[(current_agent_pos, current_prey_pos, current_predator_pos)] = 1.0

                elif current_agent_pos == current_predator_pos:
                    u_str[(current_agent_pos, current_prey_pos, current_predator_pos)] = math.inf

                else:
                    u_str[(current_agent_pos, current_prey_pos, current_predator_pos)] = rewd
    # Here I am using val_itr
    u_str = u_str
    # Meh... just for reference
    while x:
        act = []
        err_max = 0
        util_actv.clear()
        for current_agent_pos in range(50):
            for current_prey_pos in range(50):
                for current_predator_pos in range(50):
                    if current_agent_pos != current_predator_pos \
                            and current_agent_pos == current_prey_pos:
                        util_actv[(current_agent_pos, current_prey_pos, current_predator_pos)] = 0.0

                    elif abs(current_prey_pos - current_agent_pos) == 1:
                        util_actv[(current_agent_pos, current_prey_pos, current_predator_pos)] = 1

                    elif abs(current_agent_pos - current_predator_pos) \
                            or current_agent_pos == current_predator_pos == 1:
                        util_actv[(current_agent_pos, current_prey_pos, current_predator_pos)] = (10 * (10 ** 8))

                    else:
                        for agd in main_graph[current_agent_pos]:
                            nei_prd_pos = []
                            nei_py_pos = [current_prey_pos]
                            nei_py_pos.extend(main_graph[current_prey_pos])
                            nei_prd_pos.extend(main_graph[current_predator_pos])
                            # Evaluation
                            fut_util = 0
                            probs_tot = 0
                            for py in nei_py_pos:
                                for prd in nei_prd_pos:
                                    dtt_bfs = BFS(current_agent_pos, current_predator_pos)
                                    sorted_dtt_vals = []
                                    for dt1 in sorted(dtt_bfs,
                                                     key=lambda k: len(dtt_bfs[k]), reverse=False):
                                        sorted_dtt_vals.append(dt1)
                                    dt2 = len(sorted_dtt_vals)
                                    xg = random.randrange(1, 4)
                                    dtt = ((xg * 0.6)/dt2) + (0.4/len(main_graph[current_predator_pos]))
                                    probs = ((1.0 / len(nei_py_pos)) * dtt)
                                    probs_tot = probs_tot + probs
                                    fut_util = (fut_util + probs * ((u_str[(agd, py, prd)]) + rewd))
                            actv_acts = fut_util
                            act.append(actv_acts)
                        util_actv[(current_agent_pos, current_prey_pos, current_predator_pos)] = min(act)

        for current_agent_pos in range(50):
            for current_prey_pos in range(50):
                for current_predator_pos in range(50):
                    err_x = err_max, abs(util_actv[(current_agent_pos, current_prey_pos, current_predator_pos)] -
                                      u_str[(current_agent_pos, current_prey_pos, current_predator_pos)])
                    err_max = max(err_x)

        if err_max < dis_penl:
            break

        u_str = util_actv

    # Saving the environment and U* data of the same environment, because environment is randomly generated everytime.
    file_to_save_env = open('C:\Users\Arthur King\PycharmProjects\520_INTROAI_P3\Env', 'wb')
    pkl.dump(main_graph, file_to_save_env)
    file_to_save_env.close()
    # We can calculate U* everytime but very computational intensive.
    file_to_save_u_vals = open('C:\Users\Arthur King\PycharmProjects\520_INTROAI_P3\u_vals', 'wb')
    pkl.dump(u_str, file_to_save_u_vals)
    file_to_save_u_vals.close()

if __name__ == '__main__':
    begin_agent_pos = 0
    begin_prey_pos = 0
    begin_predator_pos = 0
    # Change 50
    main_graph = {1: [2, 50]}

    all_random_nodes = []
    # Change 50
    for node in range(2, 51):
        # Change 50
        if node == 50:
            main_graph[node] = [1, node - 1]
            continue
        main_graph[node] = [node + 1, node - 1]
    # Change 50
    while len(all_random_nodes) != 50:
        random_node_r_neigh = []
        random_node_l_neigh = []
        random_node_neighs = []

        random_node = random.choice(list(main_graph.keys()))
        if random_node in all_random_nodes:
            continue
        all_random_nodes.append(random_node)

        if len(main_graph[random_node]) == 2:
            n = 2
            # Change 50
            last_element = 50
            first_element = 1
            while n <= 5:
                if (random_node - n) > 0:
                    random_node_l_neigh.append(random_node - n)
                else:
                    random_node_l_neigh.append(random_node - n + last_element)
                    # There is a slight problem here because for node 10 & 1 it is taking immediate adjacent neighbours #SOLVED
                    # last_element = last_element - 1

                    # Change 50
                if (random_node + n) <= 50:
                    random_node_r_neigh.append(random_node + n)
                else:
                    random_node_r_neigh.append(random_node + n - last_element)
                    # first_element = first_element + 1
                n = n + 1

            random_node_neighs = random_node_l_neigh + random_node_r_neigh

            for i in range(len(random_node_neighs)):
                attach_node_rand = random.choice(random_node_neighs)
                if len(main_graph[attach_node_rand]) != 2:
                    continue
                main_graph[random_node].append(attach_node_rand)
                main_graph[attach_node_rand].append(random_node)
                break
        else:
            continue
    print(main_graph)

    U = cal_u_vals()
