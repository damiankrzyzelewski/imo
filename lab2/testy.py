import numpy as np
import time
import random
from imo_lab2 import (
    weighted_regret_heuristic,
    load_tsplib_instance,
    plot_solution,
    local_search,
    random_cycles,
)


def calculate_cost(distance_matrix, cycle1, cycle2):
    cost = 0
    for cycle in [cycle1, cycle2]:
        for i in range(len(cycle)):
            cost += distance_matrix[cycle[i], cycle[(i + 1) % len(cycle)]]
    return cost


def random_walk(distance_matrix, start_solution, max_time):
    """
    Algorytm losowego błądzenia
    Parametry:
      - distance_matrix: macierz odległości
      - start_solution: krotka (cycle1, cycle2) – podział wierzchołków na dwa cykle
      - max_time: maksymalny czas działania algorytmu
    """
    cycle1, cycle2 = start_solution
    best_solution = (cycle1[:], cycle2[:])
    best_cost = calculate_cost(distance_matrix, cycle1, cycle2)
    current_cost = best_cost

    start_time = time.time()

    while time.time() - start_time < max_time:
        # Losowo wybieramy rodzaj ruchu
        move_type = random.choice(["swap_between", "swap_inside", "edge_swap"])
        # Kopie bieżących cykli
        new_cycle1, new_cycle2 = cycle1[:], cycle2[:]
        delta = 0

        if move_type == "swap_between":
            # Ruch między cyklami = swap
            if len(new_cycle1) > 1 and len(new_cycle2) > 1:
                i = random.randint(0, len(new_cycle1) - 1)
                j = random.randint(0, len(new_cycle2) - 1)
                a, b = new_cycle1[i], new_cycle2[j]
                # Obliczamy deltę oddzielnie dla każdego cyklu
                for ni, cycle, node in [(i, new_cycle1, a), (j, new_cycle2, b)]:
                    prev = cycle[ni - 1]
                    nxt = cycle[(ni + 1) % len(cycle)]
                    old_cost = distance_matrix[prev, node] + distance_matrix[node, nxt]
                    new_node = b if cycle is new_cycle1 else a
                    new_cost = distance_matrix[prev, new_node] + distance_matrix[new_node, nxt]
                    delta += new_cost - old_cost
                # Wykonujemy zamianę
                new_cycle1[i], new_cycle2[j] = new_cycle2[j], new_cycle1[i]

        elif move_type == "swap_inside":
            # Ruch wewnątrz cyklu
            cycle = random.choice([new_cycle1, new_cycle2])
            n = len(cycle)
            if n < 2:
                continue
            i, j = random.sample(range(n), 2)
            if i > j:
                i, j = j, i
            a, b = cycle[i], cycle[j]
            prev_i, next_i = cycle[(i - 1) % n], cycle[(i + 1) % n]
            prev_j, next_j = cycle[(j - 1) % n], cycle[(j + 1) % n]
            if (i + 1) % n != j and (j + 1) % n != i:
                delta -= distance_matrix[prev_i, a] + distance_matrix[a, next_i]
                delta -= distance_matrix[prev_j, b] + distance_matrix[b, next_j]
                delta += distance_matrix[prev_i, b] + distance_matrix[b, next_i]
                delta += distance_matrix[prev_j, a] + distance_matrix[a, next_j]
            else:
                if (i + 1) % n == j:
                    delta -= distance_matrix[prev_i, a] + distance_matrix[a, b] + distance_matrix[b, next_j]
                    delta += distance_matrix[prev_i, b] + distance_matrix[b, a] + distance_matrix[a, next_j]
                else:
                    delta -= distance_matrix[prev_j, b] + distance_matrix[b, a] + distance_matrix[a, next_i]
                    delta += distance_matrix[prev_j, a] + distance_matrix[a, b] + distance_matrix[b, next_i]
            # Wykonujemy zamianę wierzchołków
            cycle[i], cycle[j] = cycle[j], cycle[i]

        elif move_type == "edge_swap":
            # Ruch edge swap w obrębie jednego cyklu
            cycle = random.choice([new_cycle1, new_cycle2])
            if len(cycle) > 3:
                i, j = sorted(random.sample(range(len(cycle)), 2))
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                c, d = cycle[j], cycle[(j + 1) % len(cycle)]
                delta = -distance_matrix[a, b] - distance_matrix[c, d] \
                        + distance_matrix[a, c] + distance_matrix[b, d]
                # Odwracamy fragment cyklu
                cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
            else:
                continue  # Za mało wierzchołków, by wykonać edge swap

        # Aktualizujemy koszt
        current_cost += delta
        cycle1, cycle2 = new_cycle1, new_cycle2

        if current_cost < best_cost:
            best_cost = current_cost
            best_solution = (cycle1[:], cycle2[:])

    return best_solution, best_cost


def run_experiments(distance_matrix, coords, start_choice, search_choice, intra_move_type, runs=100):
    costs = []
    times = []
    best_solution = None
    best_cost = float("inf")

    for i in range(runs):
        print(i)
        if start_choice == 1:
            cycle1, cycle2 = random_cycles(distance_matrix)
        else:
            cycle1, cycle2, _ = weighted_regret_heuristic(distance_matrix)

        start_time = time.time()
        cycle1, cycle2, cost = local_search(distance_matrix, cycle1, cycle2, search_choice, intra_move_type)
        end_time = time.time()

        costs.append(cost)
        times.append(end_time - start_time)

        if cost < best_cost:
            best_cost = cost
            best_solution = (cycle1[:], cycle2[:])

    avg_max_time = np.mean(times)

    print("\nWyniki dla funkcji celu:")
    print(f"Średnia: {np.mean(costs):.2f}, Min: {np.min(costs)}, Max: {np.max(costs)}")
    print("\nWyniki dla czasu obliczeń:")
    print(f"Średnia: {np.mean(times):.4f}s, Min: {np.min(times):.4f}s, Max: {np.max(times):.4f}s")

    # Uruchamiamy losowe błądzenie przez średni czas najwolniejszej metody
    print(f"\nUruchamiam losowe błądzenie przez {avg_max_time:.4f}s...")
    if start_choice == 1:
        rwcycle1, rwcycle2 = random_cycles(distance_matrix)
    else:
        rwcycle1, rwcycle2, _ = weighted_regret_heuristic(distance_matrix)
    best_random_solution, best_random_cost = random_walk(distance_matrix, (rwcycle1, rwcycle2), avg_max_time)

    print(f"Najlepsze rozwiązanie znalezione przez losowe błądzenie: {best_random_cost}")

    # Rysujemy najlepsze rozwiązanie z lokalnego przeszukiwania
    if best_solution:
        plot_solution(coords, best_solution[0], best_solution[1], best_cost, "Best Found Solution - Local Search")

    # Rysujemy najlepsze rozwiązanie z losowego błądzenia
    if best_random_solution:
        plot_solution(coords, best_random_solution[0], best_random_solution[1], best_random_cost,
                      "Best Found Solution - Random Walk")


if __name__ == "__main__":
    filename = "kroB200.tsp"
    distance_matrix, coords = load_tsplib_instance(filename)
    print("Wybierz rozwiązanie początkowe:")
    print("1 - Losowe cykle")
    print("2 - Heurystyka 2-żal")
    start_choice = int(input("Podaj numer: "))
    print("Wybierz metodę lokalnego przeszukiwania:")
    print("1 - Steepest Local Search")
    print("2 - Greedy Local Search")
    search_choice = int(input("Podaj numer: "))
    print("Wybierz typ ruchu wewnątrztrasowego:")
    print("1 - Zamiana krawędzi")
    print("2 - Zamiana wierzchołków")
    intra_move_choice = int(input("Podaj numer: "))
    intra_move_type = "edge_swap" if intra_move_choice == 1 else "vertex_swap"

    run_experiments(distance_matrix, coords, start_choice, search_choice, intra_move_type)
