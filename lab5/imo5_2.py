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

def calc_total_cost(c1, c2, dist):
    cost = 0
    for cycle in [c1, c2]:
        for i in range(len(cycle)):
            cost += dist[cycle[i], cycle[(i + 1) % len(cycle)]]
    return cost

def local_search(distance_matrix, cycle1, cycle2):
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

def generate_initial_population(distance_matrix, pop_size=20):
    population = []
    for _ in range(pop_size):
        cycle1, cycle2 = random_cycles(distance_matrix)
        cycle1, cycle2, cost = local_search(distance_matrix, cycle1, cycle2)
        population.append((cycle1, cycle2, cost))
    return population

def is_duplicate(candidate_cost, population):
    return any(abs(candidate_cost - indiv[2]) < 1e-6 for indiv in population)

def get_worst_index(population):
    return max(range(len(population)), key=lambda i: population[i][2])

def select_two_parents(population):
    return random.sample(population, 2)


def recombine(c1_a, c2_a, c1_b, c2_b, distance_matrix):
    def edges(cycle):
        return {(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))} | \
               {(cycle[(i + 1) % len(cycle)], cycle[i]) for i in range(len(cycle))}

    # Wybierz rodzica bazowego
    if random.random() < 0.5:
        base_c1, base_c2 = c1_a[:], c2_a[:]
        other_c1, other_c2 = c1_b, c2_b
    else:
        base_c1, base_c2 = c1_b[:], c2_b[:]
        other_c1, other_c2 = c1_a, c2_a

    # Zbiór krawędzi drugiego rodzica
    ref_edges = edges(other_c1) | edges(other_c2)

    # Usuń krawędzie niewspólne z bazowego rozwiązania
    def filter_cycle(cycle):
        new_cycle = []
        n = len(cycle)
        for i in range(n):
            u, v = cycle[i], cycle[(i + 1) % n]
            if (u, v) in ref_edges or (v, u) in ref_edges:
                new_cycle.append(u)
        return list(dict.fromkeys(new_cycle))  # usuń duplikaty

    base_c1 = filter_cycle(base_c1)
    base_c2 = filter_cycle(base_c2)

    # Zbierz wierzchołki pozostałe do wstawienia
    all_nodes = set(range(len(distance_matrix)))
    used_nodes = set(base_c1 + base_c2)
    remaining_nodes = list(all_nodes - used_nodes)
    random.shuffle(remaining_nodes)

    # Naprawa – greedy insertion
    def insert_best(node, cycle):
        best_pos = None
        best_increase = float("inf")
        for i in range(len(cycle)):
            prev, nxt = cycle[i], cycle[(i + 1) % len(cycle)]
            increase = (
                distance_matrix[prev][node]
                + distance_matrix[node][nxt]
                - distance_matrix[prev][nxt]
            )
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1
        cycle.insert(best_pos, node)

    # Wstawiamy każdy node do najkrótszego cyklu
    while remaining_nodes:
        node = remaining_nodes.pop()
        if len(base_c1) < len(base_c2):
            insert_best(node, base_c1)
        else:
            insert_best(node, base_c2)

    # Upewnij się, że cykle są równo liczne
    while len(base_c1) > len(base_c2):
        insert_best(base_c1.pop(), base_c2)
    while len(base_c2) > len(base_c1):
        insert_best(base_c2.pop(), base_c1)

    return base_c1, base_c2

def mutate(c1, c2, mutation_rate=0.1):
    for cycle in (c1, c2):
        if random.random() < mutation_rate:
            choice = random.choice(['swap', 'invert', 'relocate'])
            n = len(cycle)
            if choice == 'swap':
                i, j = random.sample(range(n), 2)
                cycle[i], cycle[j] = cycle[j], cycle[i]
            elif choice == 'invert' and n > 2:
                i, j = sorted(random.sample(range(n), 2))
                cycle[i:j+1] = reversed(cycle[i:j+1])
            elif choice == 'relocate':
                i = random.randrange(n)
                node = cycle.pop(i)
                j = random.randrange(n)
                cycle.insert(j, node)
    return c1, c2

def run_single_instance(filename, T, pop_size, use_local_search, use_mutation, mutation_rate):
    distance_matrix, coords = load_tsplib_instance(filename)
    population = generate_initial_population(distance_matrix, pop_size=pop_size)

    start_time = time.time()
    best_cost = min(population, key=lambda x: x[2])[2]
    best_solution = min(population, key=lambda x: x[2])

    iterations = 0

    while time.time() - start_time < T:
        iterations += 1
        parent1, parent2 = select_two_parents(population)

        child_c1, child_c2 = recombine(
            parent1[0], parent1[1],
            parent2[0], parent2[1],
            distance_matrix
        )

        if use_mutation:
            child_c1, child_c2 = mutate(child_c1, child_c2, mutation_rate)

        if use_local_search:
            child_c1, child_c2, child_cost = local_search(
                distance_matrix, child_c1, child_c2
            )
        else:
            child_cost = calc_total_cost(child_c1, child_c2, distance_matrix)

        worst_idx = get_worst_index(population)
        if child_cost < population[worst_idx][2] and not is_duplicate(child_cost, population):
            population[worst_idx] = (child_c1, child_c2, child_cost)
            if child_cost < best_cost:
                best_cost = child_cost
                best_solution = (child_c1, child_c2, child_cost)

    elapsed_time = time.time() - start_time
    return best_solution, elapsed_time, iterations



def main_repeated_runs(filename, T=60, pop_size=20, use_local_search=True, use_mutation=False, mutation_rate=0.1, runs=10):
    costs = []
    times = []
    iterations_list = []
    best_overall = None

    for i in range(runs):
        print(f"\n🔁 Run {i + 1}/{runs}")
        best_solution, elapsed, iterations = run_single_instance(
            filename, T, pop_size,
            use_local_search, use_mutation, mutation_rate
        )
        cost = best_solution[2]
        costs.append(cost)
        times.append(elapsed)
        iterations_list.append(iterations)

        print(f"   ✅ Koszt: {cost:.2f} | Czas: {elapsed:.2f}s | Iteracje: {iterations}")

        if best_overall is None or cost < best_overall[2]:
            best_overall = best_solution

    # Statystyki
    print("\n   Statystyki po 10 uruchomieniach:")
    print(f"   ➤ Średni koszt:      {np.mean(costs):.2f}")
    print(f"   ➤ Min koszt:         {np.min(costs):.2f}")
    print(f"   ➤ Max koszt:         {np.max(costs):.2f}")
    print(f"   ➤ Średni czas:       {np.mean(times):.2f}s")
    print(f"   ➤ Min czas:          {np.min(times):.2f}s")
    print(f"   ➤ Max czas:          {np.max(times):.2f}s")
    print(f"   ➤ Średnia liczba iteracji HEA: {np.mean(iterations_list):.1f}")
    print(f"   ➤ Min iteracji:      {np.min(iterations_list)}")
    print(f"   ➤ Max iteracji:      {np.max(iterations_list)}")

    # Rysowanie najlepszego
    distance_matrix, coords = load_tsplib_instance(filename)
    plot_solution(
        coords,
        best_overall[0], best_overall[1],
        best_overall[2],
        title="Najlepsze rozwiązanie"
    )



if __name__ == "__main__":
    main_repeated_runs(
        filename="kroA200.tsp",
        T=451,
        pop_size=20,
        use_local_search=True,
        use_mutation=False,
        mutation_rate=0.1,
        runs=10
    )


