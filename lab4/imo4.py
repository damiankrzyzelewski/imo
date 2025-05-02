import numpy as np
import random
import matplotlib.pyplot as plt
import tsplib95
import math
import time
from collections import deque


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

def random_cycles(distance_matrix):
    num_nodes = len(distance_matrix)
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    split_index = num_nodes // 2
    return nodes[:split_index], nodes[split_index:]

def local_search_LM(distance_matrix, cycle1, cycle2):

    # Funkcja pomocnicza do obliczania łącznego kosztu dwóch cykli
    def calc_total_cost(c1, c2, dist):
        cost = 0
        for cycle in [c1, c2]:
            for i in range(len(cycle)):
                cost += dist[cycle[i], cycle[(i + 1) % len(cycle)]]
        return cost

    # Sprawdza czy dana krawędź występuje w zbiorze (w obie strony)
    def edge_in_edges(edge, edge_list):
        return edge in edge_list or (edge[1], edge[0]) in edge_list

    # Sprawdza, czy ruch jest nadal możliwy do wykonania
    def is_move_applicable(removed_edges, cycle1, cycle2):
        current_edges = {(cycle1[i], cycle1[(i + 1) % len(cycle1)]) for i in range(len(cycle1))}
        current_edges.update((cycle2[i], cycle2[(i + 1) % len(cycle2)]) for i in range(len(cycle2)))
        all_present = True
        same_direction = True
        for e in removed_edges:
            if e not in current_edges and (e[1], e[0]) not in current_edges:
                return "remove"  # edges no longer exist
            elif e not in current_edges:
                same_direction = False
        return "apply" if same_direction else "skip"

    # Inicjalizacja LM lub generowanie nowych ruchów na podstawie ostatniego ruchu
    def add_new_moves(LM, cycle1, cycle2, distance_matrix, recent_move):
        tmoves = []
        # Inicjalizacja LM
        if recent_move is None:
            for i in range(len(cycle1)):
                for j in range(len(cycle2)):
                    tmoves.append((i, j, "swap"))
            for cycle, cycle_id in [(cycle1, 1), (cycle2, 2)]:
                for i in range(len(cycle)):
                    for j in range(i + 1, len(cycle)):
                        tmoves.append((i, j, "edge_swap", cycle_id))
        # Generowanie nowych ruchów na podstawie ostatniego ruchu
        else:
            if recent_move[0] == "swap":
                i, j = recent_move[1], recent_move[2]
                # indeksy elementów, na które miał wpływ ostatni ruch
                indices1 = {i, (i - 1) % len(cycle1), (i + 1) % len(cycle1)}
                indices2 = {j, (j - 1) % len(cycle2), (j + 1) % len(cycle2)}
                # generujemy możliwe swapy
                for ni in indices1:
                    for nj in range(len(cycle2)):
                        tmoves.append((ni, nj, "swap"))
                for nj in indices2:
                    for ni in range(len(cycle1)):
                        tmoves.append((ni, nj, "swap"))
                # generujemy możliwe edge_swapy
                for idx in indices1:
                    for jdx in range(len(cycle1)):
                        if idx != jdx:
                            tmoves.append((min(idx, jdx), max(idx, jdx), "edge_swap", 1))
                for idx in indices2:
                    for jdx in range(len(cycle2)):
                        if idx != jdx:
                            tmoves.append((min(idx, jdx), max(idx, jdx), "edge_swap", 2))
            elif recent_move[0] == "edge_swap":
                i, j, cycle_id = recent_move[1], recent_move[2], recent_move[3]
                current_cycle = cycle1 if cycle_id == 1 else cycle2
                # indeksy elementów, na które miał wpływ ostatni ruch
                affected_indices = [k % len(current_cycle) for k in range(i - 1, j + 2)]
                # generujemy możliwe edge_swapy
                for a in affected_indices:
                    for b in range(a + 1, a + len(current_cycle)):
                        idx_b = b % len(current_cycle)
                        tmoves.append((min(a, idx_b), max(a, idx_b), "edge_swap", cycle_id))
                affected_nodes = {current_cycle[k] for k in affected_indices}
                # generujemy możliwe swapy
                for i, v1 in enumerate(cycle1):
                    if v1 in affected_nodes:
                        for j in range(len(cycle2)):
                            tmoves.append((i, j, "swap"))
                for j, v2 in enumerate(cycle2):
                    if v2 in affected_nodes:
                        for i in range(len(cycle1)):
                            tmoves.append((i, j, "swap"))
        # dla każdego wygenerowanego ruchu liczymy delte i wstawiamy go do LM
        for move in tmoves:
            if move[2] == "swap":
                i, j = move[0], move[1]
                try:
                    a, b = cycle1[i], cycle2[j]
                    prev_i, next_i = cycle1[(i - 1) % len(cycle1)], cycle1[(i + 1) % len(cycle1)]
                    prev_j, next_j = cycle2[(j - 1) % len(cycle2)], cycle2[(j + 1) % len(cycle2)]
                except IndexError:
                    continue
                delta = -distance_matrix[prev_i, a] - distance_matrix[a, next_i]
                delta -= distance_matrix[prev_j, b] + distance_matrix[b, next_j]
                delta += distance_matrix[prev_i, b] + distance_matrix[b, next_i]
                delta += distance_matrix[prev_j, a] + distance_matrix[a, next_j]
                if delta < 0:
                    LM.append([("swap", i, j), delta, [(prev_i, a), (a, next_i), (prev_j, b), (b, next_j)]])
            else:
                i, j, _, cycle_id = move
                if i == j:
                    continue
                current_cycle = cycle1 if cycle_id == 1 else cycle2
                a, b = current_cycle[i], current_cycle[(i + 1) % len(current_cycle)]
                c, d = current_cycle[j], current_cycle[(j + 1) % len(current_cycle)]
                delta = -distance_matrix[a, b] - distance_matrix[c, d]
                delta += distance_matrix[a, c] + distance_matrix[b, d]
                if delta < 0:
                    LM.append([("edge_swap", i, j, cycle_id), delta, [(a, b), (c, d)]])

    # aktualizacja LM po ostatnim ruchu
    def update_LM_after_move(LM, last_move, cycle1, cycle2, distance_matrix):
        def is_affected(move_data, changed_nodes, removed_edges):
            if move_data[0] == "swap":
                _, i, j = move_data
                try:
                    v1, v2 = cycle1[i], cycle2[j]
                    return v1 in changed_nodes or v2 in changed_nodes
                except IndexError:
                    return True
            elif move_data[0] == "edge_swap":
                _, i, j, cycle_id = move_data
                current_cycle = cycle1 if cycle_id == 1 else cycle2
                e1 = (current_cycle[i], current_cycle[(i + 1) % len(current_cycle)])
                e2 = (current_cycle[j], current_cycle[(j + 1) % len(current_cycle)])
                return edge_in_edges(e1, removed_edges) or edge_in_edges(e2, removed_edges)
            return False

        if last_move[0] == "swap":
            i, j = last_move[1], last_move[2]
            changed_nodes = {
                cycle1[i], cycle2[j],
                cycle1[(i - 1) % len(cycle1)], cycle1[(i + 1) % len(cycle1)],
                cycle2[(j - 1) % len(cycle2)], cycle2[(j + 1) % len(cycle2)]
            }
            removed_edges = [
                (cycle1[(i - 1) % len(cycle1)], cycle1[i]),
                (cycle1[i], cycle1[(i + 1) % len(cycle1)]),
                (cycle2[(j - 1) % len(cycle2)], cycle2[j]),
                (cycle2[j], cycle2[(j + 1) % len(cycle2)])
            ]
        else:
            i, j, cycle_id = last_move[1], last_move[2], last_move[3]
            current_cycle = cycle1 if cycle_id == 1 else cycle2
            start, end = (i - 1) % len(current_cycle), (j + 1) % len(current_cycle)
            if start <= end:
                changed_nodes = set(current_cycle[k] for k in range(start, end + 1))
                removed_edges = [(current_cycle[k], current_cycle[(k + 1) % len(current_cycle)]) for k in range(start, end)]
            else:
                indices = list(range(start, len(current_cycle))) + list(range(0, end + 1))
                changed_nodes = set(current_cycle[k] for k in indices)
                removed_edges = [(current_cycle[k], current_cycle[(k + 1) % len(current_cycle)]) for k in indices[:-1]]

        LM = [move for move in LM if not is_affected(move[0], changed_nodes, removed_edges)]
        add_new_moves(LM, cycle1, cycle2, distance_matrix, last_move)
        return sorted(LM, key=lambda x: (x[1], x[0][1], x[0][2], -ord(x[0][0][0])))


    current_cost = calc_total_cost(cycle1, cycle2, distance_matrix)
    LM = deque()
    add_new_moves(LM, cycle1, cycle2, distance_matrix, None)
    LM = deque(sorted(LM, key=lambda x: (x[1], x[0][1], x[0][2], -ord(x[0][0][0]))))

    while True:
        best_move = None
        length = len(LM)
        for _ in range(length):
            move = LM.popleft()
            move_data, delta, removed_edges = move
            applicability = is_move_applicable(removed_edges, cycle1, cycle2)

            if applicability == "remove":
                continue  # nie wrzucamy z powrotem do LM - nieaplikowalny
            elif applicability == "skip":
                LM.append(move)  # odłóż z powrotem do LM - krawędzie są w przeciwnym kierunku
            elif applicability == "apply":
                best_move = move # aplikowalny - wykonaj ruch
                break

        if not best_move:
            break  # brak ruchów aplikowalnych

        # wykonujemy ruch
        move_data, delta, removed_edges = best_move
        if move_data[0] == "swap":
            i, j = move_data[1], move_data[2]
            cycle1[i], cycle2[j] = cycle2[j], cycle1[i]
        else:
            i, j, cycle_id = move_data[1], move_data[2], move_data[3]
            current_cycle = cycle1 if cycle_id == 1 else cycle2
            segment = [current_cycle[(k) % len(current_cycle)] for k in range(i + 1, j + 1)]
            for k, idx in enumerate(range(i + 1, j + 1)):
                current_cycle[idx % len(current_cycle)] = segment[-(k + 1)]
        # aktualizuj koszt
        current_cost += delta
        # zaktualizuj LM
        LM = deque(update_LM_after_move(list(LM), move_data, cycle1, cycle2, distance_matrix))
    current_cost = calc_total_cost(cycle1, cycle2, distance_matrix)
    return cycle1, cycle2, current_cost

def calc_cycle_cost(cycle, distance_matrix):
    return sum(distance_matrix[cycle[i - 1], cycle[i]] for i in range(len(cycle)))

def calc_total_cost(cycle1, cycle2, distance_matrix):
    return calc_cycle_cost(cycle1, distance_matrix) + calc_cycle_cost(cycle2, distance_matrix)

def MSLS(distance_matrix, num_iterations=200):
    best_cost = float('inf')
    best_cycle1, best_cycle2 = None, None

    start_time = time.time()

    for _ in range(num_iterations):
        cycle1, cycle2 = random_cycles(distance_matrix)
        cycle1, cycle2, cost = local_search_LM(distance_matrix, cycle1, cycle2)
        if cost < best_cost:
            best_cost = cost
            best_cycle1 = cycle1.copy()
            best_cycle2 = cycle2.copy()

    end_time = time.time()

    return best_cycle1, best_cycle2, best_cost, end_time - start_time


def perturbation_ILS(cycle1, cycle2, num_vertex_swaps=4, num_edge_swaps=2):
    # Zamiana wierzchołków między cyklami
    for _ in range(num_vertex_swaps):
        idx1 = random.randint(0, len(cycle1) - 1)
        idx2 = random.randint(0, len(cycle2) - 1)
        cycle1[idx1], cycle2[idx2] = cycle2[idx2], cycle1[idx1]

    # Operacje 2-opt w cyklu 1 (zamiana krawędzi)
    for _ in range(num_edge_swaps):
        if len(cycle1) >= 4:
            i, j = sorted(random.sample(range(len(cycle1)), 2))
            if j - i > 1:  # sensowna długość do odwrócenia
                cycle1[i:j] = reversed(cycle1[i:j])

    # Operacje 2-opt w cyklu 2
    for _ in range(num_edge_swaps):
        if len(cycle2) >= 4:
            i, j = sorted(random.sample(range(len(cycle2)), 2))
            if j - i > 1:
                cycle2[i:j] = reversed(cycle2[i:j])

    return cycle1, cycle2

def ILS(distance_matrix, time_limit):
    cycle1, cycle2 = random_cycles(distance_matrix)
    cycle1, cycle2, cost = local_search_LM(distance_matrix, cycle1, cycle2)

    best_cycle1 = cycle1.copy()
    best_cycle2 = cycle2.copy()
    best_cost = cost

    num_perturbations = 0
    start_time = time.time()

    while time.time() - start_time < time_limit:
        y_cycle1, y_cycle2 = best_cycle1.copy(), best_cycle2.copy()
        y_cycle1, y_cycle2 = perturbation_ILS(y_cycle1, y_cycle2)
        y_cycle1, y_cycle2, y_cost = local_search_LM(distance_matrix, y_cycle1, y_cycle2)

        if y_cost < best_cost:
            best_cycle1 = y_cycle1
            best_cycle2 = y_cycle2
            best_cost = y_cost

        num_perturbations += 1

    return best_cycle1, best_cycle2, best_cost, num_perturbations


def destroy_repair(cycle1, cycle2, distance_matrix, destroy_frac=0.3):
    def get_edges_with_lengths(cycle):
        edges = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            length = distance_matrix[u, v]
            edges.append((length, i, u, v))  # długość, indeks, węzły
        edges.sort(reverse=True)
        return edges

    def is_near_other_cycle(u, v, other_cycle, threshold=20):
        return any(
            distance_matrix[u, node] < threshold or distance_matrix[v, node] < threshold
            for node in other_cycle
        )

    def collect_region(cycle, index, region_size):
        n = len(cycle)
        return {cycle[(index + offset) % n] for offset in range(-region_size, region_size + 1)}

    total_nodes = len(cycle1) + len(cycle2)
    nodes_per_region = int((destroy_frac * total_nodes) // 3)
    region_size = max(1, (nodes_per_region - 1) // 2)

    region1 = set()
    region2 = set()

    # Region 1: długa krawędź z cycle1 blisko cycle2
    for length, idx, u, v in get_edges_with_lengths(cycle1):
        if is_near_other_cycle(u, v, cycle2):
            region1 = collect_region(cycle1, idx, region_size)
            break
    if not region1:
        # najdłuższa krawędź
        _, idx, _, _ = get_edges_with_lengths(cycle1)[0]
        region1 = collect_region(cycle1, idx, region_size)

    # Region 2: długa krawędź z cycle2 blisko cycle1
    for length, idx, u, v in get_edges_with_lengths(cycle2):
        if is_near_other_cycle(u, v, cycle1):
            region2 = collect_region(cycle2, idx, region_size)
            break
    if not region2:
        # najdłuższa krawędź
        _, idx, _, _ = get_edges_with_lengths(cycle2)[0]
        region2 = collect_region(cycle2, idx, region_size)

    # Region 3: losowy
    rand_cycle = random.choice([cycle1, cycle2])
    rand_index = random.randint(0, len(rand_cycle) - 1)
    region3 = collect_region(rand_cycle, rand_index, region_size)

    # Połącz
    to_remove = region1.union(region2).union(region3)
    max_remove = int(destroy_frac * total_nodes)
    if len(to_remove) > max_remove:
        to_remove = set(random.sample(to_remove, max_remove))

    # Rozdziel pozostałe węzły
    all_nodes = cycle1 + cycle2
    remaining = [node for node in all_nodes if node not in to_remove]
    half = len(remaining) // 2
    new_cycle1 = remaining[:half]
    new_cycle2 = remaining[half:]

    # Naprawa: wstaw usunięte węzły tam, gdzie koszt rośnie najmniej
    for node in to_remove:
        best_increase = float('inf')
        best_cycle = None
        best_pos = None

        for cycle in [new_cycle1, new_cycle2]:
            for i in range(len(cycle)):
                prev = cycle[i - 1]
                next = cycle[i % len(cycle)]
                increase = (
                    distance_matrix[prev, node]
                    + distance_matrix[node, next]
                    - distance_matrix[prev, next]
                )
                if increase < best_increase:
                    best_increase = increase
                    best_cycle = cycle
                    best_pos = i

        best_cycle.insert(best_pos, node)

    return new_cycle1, new_cycle2



def LNS(distance_matrix, time_limit, if_LS=True):
    cycle1, cycle2 = random_cycles(distance_matrix)
    cycle1, cycle2, cost = local_search_LM(distance_matrix, cycle1, cycle2)

    best_cycle1 = cycle1.copy()
    best_cycle2 = cycle2.copy()
    best_cost = cost

    num_perturbations = 0
    start_time = time.time()

    while time.time() - start_time < time_limit:
        y_cycle1, y_cycle2 = best_cycle1.copy(), best_cycle2.copy()
        y_cycle1, y_cycle2 = destroy_repair(y_cycle1, y_cycle2, distance_matrix)
        if if_LS:
            y_cycle1, y_cycle2, y_cost = local_search_LM(distance_matrix, y_cycle1, y_cycle2)
        else:
            y_cost = calc_total_cost(y_cycle1, y_cycle2, distance_matrix)

        if y_cost < best_cost:
            best_cycle1 = y_cycle1
            best_cycle2 = y_cycle2
            best_cost = y_cost

        num_perturbations += 1

    return best_cycle1, best_cycle2, best_cost, num_perturbations

def experiment(filename):
    print(f"==== INSTANCJA {filename} ====")
    distance_matrix, coords = load_tsplib_instance(filename)
    # MSLS
    msls_costs = []
    msls_times = []
    best_msls_cycle1 = None
    best_msls_cycle2 = None
    best_msls_cost = float('inf')

    for i in range(10):
        print(f"MSLS iteracja {i+1}/10")
        c1, c2, cost, t = MSLS(distance_matrix)
        if cost < best_msls_cost:
            best_msls_cycle1, best_msls_cycle2 = c1, c2
            best_msls_cost = cost
        msls_costs.append(cost)
        msls_times.append(t)
    avg_msls_time = np.mean(msls_times)

    print(f"MSLS: Avg Cost: {np.mean(msls_costs):.1f}  Min Cost: {np.min(msls_costs)}  Max: {np.max(msls_costs)} \n Avg time: {avg_msls_time} ({np.min(msls_times)}-{np.max(msls_times)})")
    plot_solution(coords, best_msls_cycle1, best_msls_cycle2, best_msls_cost, "MSLS - Best Solution")

    # ILS
    ils_costs = []
    ils_perturbations = []
    best_ils_cycle1 = None
    best_ils_cycle2 = None
    best_ils_cost = float('inf')
    ils_times = []
    for i in range(10):
        start_time = time.time()
        print(f"ILS iteracja {i+1}/10")
        c1, c2, cost, num = ILS(distance_matrix, avg_msls_time)
        end_time = time.time()
        if cost < best_ils_cost:
            best_ils_cycle1, best_ils_cycle2 = c1, c2
            best_ils_cost = cost
        ils_costs.append(cost)
        ils_perturbations.append(num)
        ils_times.append(end_time-start_time)


    print(f"ILS: Avg Cost: {np.mean(ils_costs):.1f}  Min: {np.min(ils_costs)}  Max: {np.max(ils_costs)}  Avg Perturbations: {np.mean(ils_perturbations):.1f}")
    print(f"Avg time: {np.mean(ils_times)} ({np.min(ils_times)}-{np.max(ils_times)})")
    plot_solution(coords, best_ils_cycle1, best_ils_cycle2, best_ils_cost, "ILS - Best Solution")
    # LNS
    lns_costs = []
    lns_perturbations = []
    best_lns_cycle1 = None
    best_lns_cycle2 = None
    best_lns_cost = float('inf')
    LNS_times = []
    for i in range(10):
        print(f"LNS iteracja {i+1}/10")
        start_time = time.time()
        c1, c2, cost, num = LNS(distance_matrix, avg_msls_time)
        end_time = time.time()
        if cost < best_lns_cost:
            best_lns_cycle1, best_lns_cycle2 = c1, c2
            best_lns_cost = cost
        lns_costs.append(cost)
        lns_perturbations.append(num)
        LNS_times.append(end_time-start_time)

    print(f"LNS: Avg Cost: {np.mean(lns_costs):.1f}  Min: {np.min(lns_costs)}  Max: {np.max(lns_costs)}  Avg Perturbations: {np.mean(lns_perturbations):.1f}")
    print(f"Avg time: {np.mean(LNS_times)} ({np.min(LNS_times)}-{np.max(LNS_times)})")
    plot_solution(coords, best_lns_cycle1, best_lns_cycle2, best_lns_cost, "LNS - Best Solution")

    # LNS bez LS
    lns_costs = []
    lns_perturbations = []
    best_lns_cycle1 = None
    best_lns_cycle2 = None
    best_lns_cost = float('inf')
    LNS_times = []
    for i in range(10):
        print(f"LNS iteracja {i + 1}/10")
        start_time = time.time()
        c1, c2, cost, num = LNS(distance_matrix, avg_msls_time, if_LS=False)
        end_time = time.time()
        if cost < best_lns_cost:
            best_lns_cycle1, best_lns_cycle2 = c1, c2
            best_lns_cost = cost
        lns_costs.append(cost)
        lns_perturbations.append(num)
        LNS_times.append(end_time - start_time)

    print(
        f"LNS bez LS: Avg Cost: {np.mean(lns_costs):.1f}  Min: {np.min(lns_costs)}  Max: {np.max(lns_costs)}  Avg Perturbations: {np.mean(lns_perturbations):.1f}")
    print(f"Avg time: {np.mean(LNS_times)} ({np.min(LNS_times)}-{np.max(LNS_times)})")
    plot_solution(coords, best_lns_cycle1, best_lns_cycle2, best_lns_cost, "LNS - Best Solution")


if __name__ == "__main__":
    experiment("kroa200.tsp")
    #experiment("krob200.tsp")
