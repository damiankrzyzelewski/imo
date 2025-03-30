import numpy as np
import random
import matplotlib.pyplot as plt
import tsplib95
import math

def load_tsplib_instance(filename):
    problem = tsplib95.load(filename)
    nodes = list(problem.get_nodes())
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    coords = np.array([problem.node_coords[node] for node in nodes])
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
                distance_matrix[i, j] = round(dist)
    return distance_matrix, coords

def plot_solution(coords, cycle1, cycle2, cost, title):
    plt.figure(figsize=(8, 6))
    for cycle, color in zip([cycle1, cycle2], ['b', 'g']):
        cycle_coords = np.array([coords[i] for i in cycle] + [coords[cycle[0]]])
        plt.plot(cycle_coords[:, 0], cycle_coords[:, 1], f'{color}-o', markersize=5)
    plt.scatter(coords[:, 0], coords[:, 1], color='red', marker='x')
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=8, ha='right')
    plt.title(f"{title}\nCost: {cost}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def calc_cycle_cost(cycle, distance_matrix):
    return sum(distance_matrix[cycle[i - 1], cycle[i]] for i in range(len(cycle)))

def calc_total_cost(cycle1, cycle2, distance_matrix):
    return calc_cycle_cost(cycle1, distance_matrix) + calc_cycle_cost(cycle2, distance_matrix)

# zwraca losowy podział na cykle
def random_cycles(distance_matrix):
    num_nodes = len(distance_matrix)
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    split_index = num_nodes // 2
    return nodes[:split_index], nodes[split_index:]

# znajduje najlepszą pozycje do wstawienia elementu (dla weighted 2-regret)
def insert_best_position(cycle, node, distance_matrix):
    best_position = None
    best_increase = float('inf')

    for i in range(len(cycle)):
        prev_node = cycle[i - 1]
        next_node = cycle[i]
        increase = distance_matrix[prev_node, node] + distance_matrix[node, next_node] - distance_matrix[
            prev_node, next_node]

        if increase < best_increase:
            best_increase = increase
            best_position = i

    cycle.insert(best_position, node)

# podział początkowy dla 2-regret
def split_nodes(distance_matrix):
    num_nodes = len(distance_matrix)
    available_nodes = set(range(num_nodes))

    start1 = random.choice(list(available_nodes))
    available_nodes.remove(start1)
    start2 = max(available_nodes, key=lambda x: distance_matrix[start1, x])
    available_nodes.remove(start2)

    size1 = (num_nodes + 1) // 2
    size2 = num_nodes // 2
    return start1, start2, size1, size2, available_nodes

# heurystyka ważonego 2-żalu
def weighted_regret_heuristic(distance_matrix, alpha=1.0, beta=-1.0):
    start1, start2, size1, size2, available_nodes = split_nodes(distance_matrix)
    cycle1 = [start1]
    cycle2 = [start2]
    while available_nodes:
        for cycle, size in zip([cycle1, cycle2], [size1, size2]):
            if len(cycle) < size:
                best_node = None
                best_score = -float('inf')
                for node in available_nodes:
                    costs = sorted([distance_matrix[node, cycle[i - 1]] + distance_matrix[node, cycle[i]] -
                                    distance_matrix[cycle[i - 1], cycle[i]] for i in range(len(cycle))])
                    regret = costs[1] - costs[0] if len(costs) > 1 else costs[0]
                    greedy_cost = costs[0]
                    score = alpha * regret + beta * greedy_cost
                    if score > best_score:
                        best_score = score
                        best_node = node
                insert_best_position(cycle, best_node, distance_matrix)
                available_nodes.remove(best_node)
    cost = calc_total_cost(cycle1, cycle2, distance_matrix)
    return cycle1, cycle2, cost

def local_search(distance_matrix, cycle1, cycle2, strategy, intra_move_type):
    # Obliczamy początkowy koszt całego rozwiązania
    current_cost = calc_total_cost(cycle1, cycle2, distance_matrix)
    improved = True  # Czy znaleziono lepsze rozwiązanie w ostatniej iteracji

    while improved:
        improved = False
        best_move = None
        best_delta = 0

        # Zbieramy wszystkie możliwe ruchy: między cyklami (swap) oraz wewnątrz jednego cyklu (vertex/edge swap)
        moves = []

        # Międzycyklowe swap'y
        for i in range(len(cycle1)):
            for j in range(len(cycle2)):
                moves.append((i, j, "swap"))

        # Wewnątrzcyklowe ruchy: zależne od intra_move_type ("vertex_swap" lub "edge_swap")
        for cycle in [cycle1, cycle2]:
            for i in range(len(cycle) - 1):
                for j in range(i + 1, len(cycle)):
                    moves.append((i, j, intra_move_type, cycle))

        # Jeżeli greedy ls to losowa kolejność przeszukiwania ruchów
        if strategy == 2:
            random.shuffle(moves)

        # Przeglądamy wszystkie możliwe ruchy
        for move in moves:
            if move[2] == "swap":
                # swap między cyklami
                i, j, _ = move
                a, b = cycle1[i], cycle2[j]
                delta = 0

                # Oblicz zmianę kosztu w obu cyklach po swap a <-> b
                for ni, cycle, node in [(i, cycle1, a), (j, cycle2, b)]:
                    prev = cycle[ni - 1]
                    next = cycle[(ni + 1) % len(cycle)]
                    old_cost = distance_matrix[prev, node] + distance_matrix[node, next]
                    new_cost = distance_matrix[prev, b if cycle is cycle1 else a] + distance_matrix[b if cycle is cycle1 else a, next]
                    delta += new_cost - old_cost

                # Czy ruch poprawia koszt
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("swap", i, j)
                    # Jeżeli Greedy to przerywamy poszukiwania
                    if strategy == 2:
                        break

            elif move[2] == "vertex_swap":
                # Zamiana dwóch wierzchołków w jednym cyklu
                i, j, _, cycle = move
                if i == j:
                    continue
                a, b = cycle[i], cycle[j]
                n = len(cycle)
                prev_i, next_i = cycle[(i - 1) % n], cycle[(i + 1) % n]
                prev_j, next_j = cycle[(j - 1) % n], cycle[(j + 1) % n]

                delta = 0
                # Wierzchołki nie będące sąsiadami
                if (i + 1) % n != j and (j + 1) % n != i:
                    delta -= distance_matrix[prev_i, a] + distance_matrix[a, next_i]
                    delta -= distance_matrix[prev_j, b] + distance_matrix[b, next_j]
                    delta += distance_matrix[prev_i, b] + distance_matrix[b, next_i]
                    delta += distance_matrix[prev_j, a] + distance_matrix[a, next_j]
                else:
                    # Sąsiadujące wierzchołki – osobno, żeby nie policzyć 2x tej samej krawędzi
                    if (i + 1) % n == j:
                        delta -= distance_matrix[prev_i, a] + distance_matrix[a, b] + distance_matrix[b, next_j]
                        delta += distance_matrix[prev_i, b] + distance_matrix[b, a] + distance_matrix[a, next_j]
                    else:
                        delta -= distance_matrix[prev_j, b] + distance_matrix[b, a] + distance_matrix[a, next_i]
                        delta += distance_matrix[prev_j, a] + distance_matrix[a, b] + distance_matrix[b, next_i]

                # Czy ruch poprawia koszt
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("vertex_swap", i, j, cycle)
                    # Jeżeli Greedy to przerywamy poszukiwania
                    if strategy == 2:
                        break

            else:
                # Edge swap w cyklu: odwrócenie fragmentu między i+1 a j
                i, j, _, cycle = move
                if i >= j:
                    continue
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                c, d = cycle[j], cycle[(j + 1) % len(cycle)]

                # Oblicz zmianę długości po odwróceniu fragmentu
                delta = -distance_matrix[a, b] - distance_matrix[c, d]
                delta += distance_matrix[a, c] + distance_matrix[b, d]

                if delta < best_delta:
                    best_delta = delta
                    best_move = ("edge_swap", i, j, cycle)
                    if strategy == 2:
                        break

        # Jeśli znaleźliśmy najlepszy ruch to wykonujemy go
        if best_move:
            improved = True
            move_type = best_move[0]
            if move_type == "swap":
                i, j = best_move[1], best_move[2]
                cycle1[i], cycle2[j] = cycle2[j], cycle1[i]
            elif move_type == "vertex_swap":
                i, j, cycle = best_move[1], best_move[2], best_move[3]
                cycle[i], cycle[j] = cycle[j], cycle[i]
            else:  # edge_swap
                i, j, cycle = best_move[1], best_move[2], best_move[3]
                cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])

            # Zaktualizuj koszt po ruchu
            current_cost += best_delta

    return cycle1, cycle2, current_cost



if __name__ == "__main__":
    filename = "kroB200.tsp"
    distance_matrix, coords = load_tsplib_instance(filename)
    print("Wybierz rozwiązanie początkowe:")
    print("1 - Losowe cykle")
    print("2 - Heurystyka 2-żal")
    start_choice = int(input("Podaj numer: "))
    if start_choice == 1:
        cycle1, cycle2 = random_cycles(distance_matrix)
    else:
        cycle1, cycle2, initial_cost = weighted_regret_heuristic(distance_matrix)
    initial_cost = calc_total_cost(cycle1, cycle2, distance_matrix)
    print(f"Initial Cost: {initial_cost}")
    print("Wybierz metodę lokalnego przeszukiwania:")
    print("1 - Steepest Local Search")
    print("2 - Greedy Local Search")
    search_choice = int(input("Podaj numer: "))
    print("Wybierz typ ruchu wewnątrztrasowego:")
    print("1 - Zamiana krawędzi")
    print("2 - Zamiana wierzchołków")
    intra_move_choice = int(input("Podaj numer: "))
    intra_move_type = "edge_swap" if intra_move_choice == 1 else "vertex_swap"
    cycle1, cycle2, cost = local_search(distance_matrix, cycle1, cycle2, search_choice, intra_move_type)
    print(f"Final Cost: {cost}")
    plot_solution(coords, cycle1, cycle2, cost, "Final Solution")
