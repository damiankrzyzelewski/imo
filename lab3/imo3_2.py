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

def calc_cycle_cost(cycle, distance_matrix):
    return sum(distance_matrix[cycle[i - 1], cycle[i]] for i in range(len(cycle)))

def calc_total_cost(cycle1, cycle2, distance_matrix):
    return calc_cycle_cost(cycle1, distance_matrix) + calc_cycle_cost(cycle2, distance_matrix)

def random_cycles(distance_matrix):
    num_nodes = len(distance_matrix)
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    split_index = num_nodes // 2
    return nodes[:split_index], nodes[split_index:]

def local_search(distance_matrix, cycle1, cycle2):
    current_cost = calc_total_cost(cycle1, cycle2, distance_matrix)
    improved = True
    while improved:
        improved = False
        best_move = None
        best_delta = 0
        moves = []

        for i in range(len(cycle1)):
            for j in range(len(cycle2)):
                moves.append((i, j, "swap"))

        for cycle in [cycle1, cycle2]:
            for i in range(len(cycle) - 1):
                for j in range(i + 1, len(cycle)):
                    moves.append((i, j, "edge_swap", cycle))

        for move in moves:
            if move[2] == "swap":
                i, j, _ = move
                a, b = cycle1[i], cycle2[j]
                delta = 0
                for ni, cycle, node in [(i, cycle1, a), (j, cycle2, b)]:
                    prev = cycle[ni - 1]
                    next = cycle[(ni + 1) % len(cycle)]
                    new_node = b if cycle is cycle1 else a
                    delta += distance_matrix[prev, new_node] + distance_matrix[new_node, next]
                    delta -= distance_matrix[prev, node] + distance_matrix[node, next]
                if delta < best_delta or (best_move is not None and (delta == best_delta and i < best_move[1])) or (best_move is not None and (delta == best_delta and i == best_move[1] and j < best_move[2]) or (best_move is not None and (delta == best_delta and i == best_move[1] and j == best_move[2] and move[2] == "swap"))):
                    best_delta = delta
                    best_move = ("swap", i, j)
            else:
                i, j, _, cycle = move
                if i >= j:
                    continue
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                c, d = cycle[j], cycle[(j + 1) % len(cycle)]
                delta = -distance_matrix[a, b] - distance_matrix[c, d]
                delta += distance_matrix[a, c] + distance_matrix[b, d]
                if delta < best_delta or (best_move is not None and (delta == best_delta and i < best_move[1])) or (best_move is not None and (delta == best_delta and i == best_move[1] and j < best_move[2]) or (best_move is not None and (delta == best_delta and i == best_move[1] and j == best_move[2] and move[2] == "swap"))):
                    best_delta = delta
                    best_move = ("edge_swap", i, j, cycle)

        if best_move:
            improved = True
            move_type = best_move[0]
            if move_type == "swap":
                i, j = best_move[1], best_move[2]
                cycle1[i], cycle2[j] = cycle2[j], cycle1[i]
            else:
                i, j, cycle = best_move[1], best_move[2], best_move[3]
                cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
            current_cost += best_delta

    return cycle1, cycle2, current_cost

def compute_candidate_edges(distance_matrix, k=10):
    num_nodes = len(distance_matrix)
    candidate_edges = {}
    for i in range(num_nodes):
        neighbors = np.argsort(distance_matrix[i])[1:k+1]
        candidate_edges[i] = set(neighbors)
    return candidate_edges


def steepest_candidate_edge_swap(distance_matrix, cycle1, cycle2, candidate_edges):
    improved = True
    current_cost = calc_total_cost(cycle1, cycle2, distance_matrix)  # koszt początkowy rozwiązania

    while improved:
        improved = False
        best_delta = 0
        best_move = None

        # --- Ruchy między cyklami ---
        for i in range(len(cycle1)):
            for j in range(len(cycle2)):
                a, b = cycle1[i], cycle2[j]

                # rozważamy tylko wierzchołki, które są kandydatami
                if b not in candidate_edges[a] and a not in candidate_edges[b]:
                    continue

                # sąsiedzi a w cycle1
                a_prev = cycle1[i - 1]
                a_next = cycle1[(i + 1) % len(cycle1)]
                # sąsiedzi b w cycle2
                b_prev = cycle2[j - 1]
                b_next = cycle2[(j + 1) % len(cycle2)]

                # oblicz zmianę kosztu po zamianie a <-> b
                delta = 0
                delta -= distance_matrix[a_prev][a] + distance_matrix[a][a_next]
                delta -= distance_matrix[b_prev][b] + distance_matrix[b][b_next]
                delta += distance_matrix[a_prev][b] + distance_matrix[b][a_next]
                delta += distance_matrix[b_prev][a] + distance_matrix[a][b_next]

                # aktualizuj najlepszy ruch, jeśli poprawia rozwiązanie
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("swap", i, j)

        # --- Ruchy wewnątrz jednego cyklu ---
        for cycle in [cycle1, cycle2]:
            n = len(cycle)
            for i in range(n):
                n1 = cycle[i]
                next_i = (i + 1) % n
                prev_i = (i - 1) % n
                for j in range(n):
                    if i == j:
                        continue
                    n2 = cycle[j]
                    if n2 not in candidate_edges[n1]:
                        continue

                    next_j = (j + 1) % n
                    prev_j = (j - 1) % n

                    # przypadek: krawędzie (n1, next_n1) oraz (n2, next_n2)
                    a, b = n1, cycle[next_i]
                    c, d = n2, cycle[next_j]
                    if len({a, b, c, d}) == 4:
                        delta = -distance_matrix[a][b] - distance_matrix[c][d] + distance_matrix[a][c] + distance_matrix[b][d]
                        if (c in candidate_edges[a] or a in candidate_edges[c]):
                            if delta < best_delta:
                                best_delta = delta
                                best_move = ("intra_next", i, j, cycle)

                    # przypadek: krawędzie (prev_n1, n1) oraz (prev_n2, n2)
                    a, b = cycle[prev_i], n1
                    c, d = cycle[prev_j], n2
                    if len({a, b, c, d}) == 4:
                        delta = -distance_matrix[a][b] - distance_matrix[c][d] + distance_matrix[a][c] + distance_matrix[b][d]
                        if b in candidate_edges[d] or d in candidate_edges[b]:
                            if delta < best_delta:
                                best_delta = delta
                                best_move = ("intra_prev", i, j, cycle)

        # wykonujemy najlepszy znaleziony ruch
        if best_move:
            improved = True
            move_type = best_move[0]

            if move_type == "swap":
                i, j = best_move[1], best_move[2]
                cycle1[i], cycle2[j] = cycle2[j], cycle1[i]

            elif move_type == "intra_next":
                i, j, cycle = best_move[1:]
                i_next = (i + 1) % len(cycle)
                j_next = (j + 1) % len(cycle)
                # odwrócenie fragmentu cyklu między sąsiednimi krawędziami
                if i_next < j:
                    cycle[i_next:j + 1] = reversed(cycle[i_next:j + 1])
                else:
                    cycle[j_next:i + 1] = reversed(cycle[j_next:i + 1])

            elif move_type == "intra_prev":
                i, j, cycle = best_move[1:]
                i_prev = (i - 1) % len(cycle)
                j_prev = (j - 1) % len(cycle)
                # odwrócenie fragmentu cyklu między sąsiednimi krawędziami
                if j < i_prev:
                    cycle[j:i_prev + 1] = reversed(cycle[j:i_prev + 1])
                else:
                    cycle[i:j_prev + 1] = reversed(cycle[i:j_prev + 1])

            # zaktualizuj całkowity koszt rozwiązania
            current_cost += best_delta

    # zwróć nowe cykle i ich koszt
    return cycle1, cycle2, current_cost


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
    return cycle1, cycle2, current_cost



def main():
    filename = "kroB200.tsp"
    distance_matrix, coords = load_tsplib_instance(filename)

    print("== Losowe rozwiązanie ==")
    cycle1, cycle2 = random_cycles(distance_matrix)
    init_cost = calc_total_cost(cycle1, cycle2, distance_matrix)
    print(f"Initial Cost: {init_cost}")

    print("\n== Steepest Local Search (standard edge swap) ==")
    start = time.time()
    c1_std, c2_std, cost_std = local_search(distance_matrix, cycle1.copy(), cycle2.copy())
    end = time.time()
    print(f"Czas: {end - start}")
    print(f"Final Cost: {cost_std}")

    print("\n== Kandydacki Steepest Edge Swap ==")
    start = time.time()
    candidate_edges = compute_candidate_edges(distance_matrix)
    c1_cand, c2_cand, cost_cand = steepest_candidate_edge_swap(distance_matrix, cycle1.copy(), cycle2.copy(), candidate_edges)
    end = time.time()
    print(f"Czas: {end - start}")
    print(f"Final Cost: {cost_cand}")

    print("\n== LM Steepest Edge Swap ==")
    start = time.time()
    c1_LM, c2_LM, cost_LM = local_search_LM(distance_matrix, cycle1.copy(), cycle2.copy())
    end = time.time()
    print(f"Czas: {end - start}")
    print(f"Final Cost: {cost_LM}")

if __name__ == "__main__":
    main()