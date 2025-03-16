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


def nearest_neighbor(distance_matrix):
    start1, start2, size1, size2, available_nodes = split_nodes(distance_matrix)

    cycle1 = [start1]
    cycle2 = [start2]

    while available_nodes:
        if len(cycle1) < size1:
            last = cycle1[-1]
            next_node = min(available_nodes, key=lambda x: distance_matrix[last, x])
            cycle1.append(next_node)
        else:
            last = cycle2[-1]
            next_node = min(available_nodes, key=lambda x: distance_matrix[last, x])
            cycle2.append(next_node)
        available_nodes.remove(next_node)

    cost = sum(distance_matrix[cycle1[i], cycle1[i + 1]] for i in range(-1, len(cycle1) - 1))
    cost += sum(distance_matrix[cycle2[i], cycle2[i + 1]] for i in range(-1, len(cycle2) - 1))
    return cycle1, cycle2, cost

def greedy_cycle(distance_matrix):
    start1, start2, size1, size2, available_nodes = split_nodes(distance_matrix)

    cycle1 = [start1, min(available_nodes, key=lambda x: distance_matrix[start1, x])]
    available_nodes.remove(cycle1[1])
    cycle2 = [start2, min(available_nodes, key=lambda x: distance_matrix[start2, x])]
    available_nodes.remove(cycle2[1])

    while available_nodes:
        for cycle, size in zip([cycle1, cycle2], [size1, size2]):
            if len(cycle) < size:
                best_insert = None
                best_increase = float('inf')
                for node in available_nodes:
                    for i in range(len(cycle)):
                        increase = (distance_matrix[cycle[i - 1], node] + distance_matrix[node, cycle[i]] -
                                    distance_matrix[cycle[i - 1], cycle[i]])
                        if increase < best_increase:
                            best_increase = increase
                            best_insert = (node, i)
                cycle.insert(best_insert[1], best_insert[0])
                available_nodes.remove(best_insert[0])

    cost = sum(distance_matrix[cycle1[i], cycle1[i + 1]] for i in range(-1, len(cycle1) - 1))
    cost += sum(distance_matrix[cycle2[i], cycle2[i + 1]] for i in range(-1, len(cycle2) - 1))
    return cycle1, cycle2, cost

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


def regret_heuristic(distance_matrix):
    start1, start2, size1, size2, available_nodes = split_nodes(distance_matrix)

    cycle1 = [start1]
    cycle2 = [start2]

    while available_nodes:
        for cycle, size in zip([cycle1, cycle2], [size1, size2]):
            if len(cycle) < size:
                best_node = None
                best_regret = -float('inf')

                for node in available_nodes:
                    costs = sorted([distance_matrix[node, cycle[i - 1]] + distance_matrix[node, cycle[i]] -
                                    distance_matrix[cycle[i - 1], cycle[i]] for i in range(len(cycle))])
                    regret = costs[1] - costs[0] if len(costs) > 1 else costs[0]

                    if regret > best_regret:
                        best_regret = regret
                        best_node = node

                insert_best_position(cycle, best_node, distance_matrix)
                available_nodes.remove(best_node)

    cost = sum(distance_matrix[cycle1[i], cycle1[i + 1]] for i in range(-1, len(cycle1) - 1))
    cost += sum(distance_matrix[cycle2[i], cycle2[i + 1]] for i in range(-1, len(cycle2) - 1))

    return cycle1, cycle2, cost


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

    cost = sum(distance_matrix[cycle1[i], cycle1[i + 1]] for i in range(-1, len(cycle1) - 1))
    cost += sum(distance_matrix[cycle2[i], cycle2[i + 1]] for i in range(-1, len(cycle2) - 1))

    return cycle1, cycle2, cost


def run_experiments(distance_matrix, algorithm, runs=100):
    results = []
    best_solution, best_cost = None, float('inf')

    for _ in range(runs):
        cycle1, cycle2, cost = algorithm(distance_matrix)
        if cost < best_cost:
            best_cost = cost
            best_solution = (cycle1, cycle2)
        results.append(cost)

    return np.mean(results), min(results), max(results), best_solution, best_cost


if __name__ == "__main__":
    distance_matrix, coords = load_tsplib_instance("kroB200.tsp")

    print("Wybierz heurystykę:")
    print("1 - Nearest Neighbor")
    print("2 - Greedy Cycle")
    print("3 - Regret Heuristic")
    print("4 - Weighted Regret Heuristic")
    choice = int(input("Podaj numer: "))

    algorithms = {
        1: nearest_neighbor,
        2: greedy_cycle,
        3: regret_heuristic,
        4: weighted_regret_heuristic
    }

    if choice in algorithms:
        avg, min_cost, max_cost, best_solution, best_cost = run_experiments(distance_matrix, algorithms[choice])
        print(f"Selected Method: Avg={avg}, Min={min_cost}, Max={max_cost}")
        plot_solution(coords, best_solution[0], best_solution[1], best_cost, "Selected Solution")
    else:
        print("Niepoprawny wybór.")