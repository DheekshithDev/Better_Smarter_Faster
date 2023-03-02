import networkx as nx
import matplotlib.pyplot as plt
import random


def create_graph():
    all_random_nodes = []
    G = nx.Graph()
    G.add_nodes_from(range(1, 51))
    G.add_edges_from([(1, 2), (1, 50)])
    for node in range(2, 51):
        if node == 50:
            G.add_edges_from([(node, node - 1), (node, 1)])
            continue
        G.add_edges_from([(node, node + 1), (node, node - 1)])

    while len(all_random_nodes) != 50:
        random_node_r_neigh = []
        random_node_l_neigh = []
        random_node_neighs = []

        random_node = random.choice(list(G.nodes))
        if random_node in all_random_nodes:
            continue
        all_random_nodes.append(random_node)

        if G.degree[random_node] == 2:
            n = 2
            last_element = 50
            first_element = 1
            while n <= 5:
                if (random_node - n) > 0:
                    random_node_l_neigh.append(random_node - n)
                else:
                    random_node_l_neigh.append(abs((random_node - n) + last_element))

                if (random_node + n) <= 50:
                    random_node_r_neigh.append(random_node + n)
                else:
                    random_node_r_neigh.append(abs((random_node + n) - last_element))
                n = n + 1

            random_node_neighs = random_node_l_neigh + random_node_r_neigh

            for i in range(len(random_node_neighs)):
                attach_node_rand = random.choice(random_node_neighs)
                if G.degree[attach_node_rand] != 2:
                    continue
                G.add_edge(random_node, attach_node_rand)
                G.add_edge(attach_node_rand, random_node)
                break
        else:
            continue

    return G


G = create_graph()
begin_agent_pos = 0
begin_prey_pos = 0
begin_predator_pos = 0

agent_random_loc = random.choice(list(G.nodes))

while True:
    prey_random_loc = random.choice(list(G.nodes))
    predator_random_loc = random.choice(list(G.nodes))
    if prey_random_loc == agent_random_loc or predator_random_loc == agent_random_loc:
        continue
    break

agent_pos = agent_random_loc
prey_pos = prey_random_loc
predator_pos = predator_random_loc

print('agent_pos = ', agent_pos)
print('prey_pos = ', prey_pos)
print('predator_pos = ', predator_pos)

agent_pos = 38
predator_pos = 41
prey_pos = 21

# DRAW GRAPH
color_map = []
for n in list(G.nodes):
    if n == agent_pos:
        color_map.append('tab:blue')
        continue
    if n == prey_pos:
        color_map.append('tab:green')
        continue
    if n == predator_pos:
        color_map.append('tab:red')
        continue
    color_map.append("tab:gray")

pos = nx.circular_layout(G)
# node_size=10
nx.draw(G, with_labels=False, pos=pos, node_color=color_map, node_size=40)
plt.show()


