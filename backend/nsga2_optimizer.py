"""
Manual NSGA-II Multi-Objective Optimizer (Layer 5)
====================================================
Full manual implementation of NSGA-II with:
  - Non-dominated sorting
  - Crowding distance
  - SBX crossover + Gaussian mutation
  - Hypervolume convergence tracking
  - Parallel evaluation via ThreadPoolExecutor
Decision variables: Motor_Speed, Temperature, Pressure, Flow_Rate, Hold_Time
Objectives: minimize Energy, minimize Carbon, maximize Quality
"""

import numpy as np
import os
import random
from concurrent.futures import ThreadPoolExecutor

MAX_WORKERS = min(8, os.cpu_count() or 4)

# Industrial bounds for decision variables
BOUNDS = {
    "Motor_Speed": (1200.0, 1800.0),
    "Temperature": (60.0, 90.0),
    "Pressure": (3.0, 5.5),
    "Flow_Rate": (15.0, 30.0),
    "Hold_Time": (10.0, 25.0),
}
VAR_NAMES = list(BOUNDS.keys())
N_VARS = len(VAR_NAMES)

CARBON_INTENSITY = 0.82  # kg CO2 per kWh


# ═══════════════════════════════════════════════════════════════════════
# NSGA-II Core
# ═══════════════════════════════════════════════════════════════════════

def dominates(obj1, obj2):
    """Check if obj1 dominates obj2 (all objectives minimized)."""
    better_in_all = all(obj1[i] <= obj2[i] for i in range(len(obj1)))
    better_in_one = any(obj1[i] < obj2[i] for i in range(len(obj1)))
    return better_in_all and better_in_one


def fast_non_dominated_sort(objectives):
    """Rank solutions into non-dominated fronts."""
    n = len(objectives)
    domination_counts = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominated_solutions[i].append(j)
                domination_counts[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_solutions[j].append(i)
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for p in fronts[k]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        k += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def assign_crowding_distance(front_indices, objectives):
    """Assign crowding distance for diversity preservation."""
    n = len(front_indices)
    if n <= 2:
        return [float("inf")] * n

    distances = [0.0] * n
    num_obj = len(objectives[0])

    for m in range(num_obj):
        sorted_idx = sorted(range(n), key=lambda i: objectives[front_indices[i]][m])

        distances[sorted_idx[0]] = float("inf")
        distances[sorted_idx[-1]] = float("inf")

        obj_range = (
            objectives[front_indices[sorted_idx[-1]]][m]
            - objectives[front_indices[sorted_idx[0]]][m]
        )
        if obj_range < 1e-10:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                objectives[front_indices[sorted_idx[i + 1]]][m]
                - objectives[front_indices[sorted_idx[i - 1]]][m]
            ) / obj_range

    return distances


def calculate_hypervolume(pareto_objectives, reference_point):
    """Approximate hypervolume (higher = better Pareto coverage)."""
    if not pareto_objectives:
        return 0.0

    volume = 0.0
    for obj in pareto_objectives:
        contrib = 1.0
        for i in range(len(obj)):
            diff = reference_point[i] - obj[i]
            if diff <= 0:
                contrib = 0
                break
            contrib *= diff
        volume += max(0, contrib)
    return volume


# ═══════════════════════════════════════════════════════════════════════
# Genetic Operators
# ═══════════════════════════════════════════════════════════════════════

def initialize_population(pop_size):
    """Random initialization within bounds."""
    population = []
    for _ in range(pop_size):
        individual = []
        for var in VAR_NAMES:
            lo, hi = BOUNDS[var]
            individual.append(random.uniform(lo, hi))
        population.append(individual)
    return population


def sbx_crossover(p1, p2, eta=20):
    """Simulated Binary Crossover."""
    child1, child2 = list(p1), list(p2)

    for i in range(N_VARS):
        if random.random() > 0.5:
            continue
        lo, hi = BOUNDS[VAR_NAMES[i]]

        if abs(p1[i] - p2[i]) < 1e-10:
            continue

        beta = 1.0 + (2.0 * min(p1[i] - lo, p2[i] - lo) / (abs(p1[i] - p2[i]) + 1e-10))
        alpha = 2.0 - beta ** (-(eta + 1))

        u = random.random()
        if u <= 1.0 / alpha:
            betaq = (u * alpha) ** (1.0 / (eta + 1))
        else:
            betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1))

        child1[i] = 0.5 * ((1 + betaq) * p1[i] + (1 - betaq) * p2[i])
        child2[i] = 0.5 * ((1 - betaq) * p1[i] + (1 + betaq) * p2[i])

        child1[i] = max(lo, min(hi, child1[i]))
        child2[i] = max(lo, min(hi, child2[i]))

    return child1, child2


def gaussian_mutation(individual, mutation_strength=0.1):
    """Gaussian mutation with bounds enforcement."""
    mutant = list(individual)
    for i in range(N_VARS):
        if random.random() < 0.2:  # per-variable mutation probability
            lo, hi = BOUNDS[VAR_NAMES[i]]
            sigma = (hi - lo) * mutation_strength
            mutant[i] += random.gauss(0, sigma)
            mutant[i] = max(lo, min(hi, mutant[i]))
    return mutant


def binary_tournament_selection(population, fronts, crowding_distances, n_select):
    """Select parents via binary tournament using rank + crowding distance."""
    # Build rank map & distance map
    rank_map = {}
    dist_map = {}
    for rank, front in enumerate(fronts):
        dists = crowding_distances[rank]
        for idx_in_front, pop_idx in enumerate(front):
            rank_map[pop_idx] = rank
            dist_map[pop_idx] = dists[idx_in_front]

    selected = []
    pop_indices = list(range(len(population)))

    for _ in range(n_select):
        i, j = random.sample(pop_indices, 2)
        # Prefer lower rank, then higher crowding distance
        if rank_map.get(i, 999) < rank_map.get(j, 999):
            selected.append(population[i])
        elif rank_map.get(i, 999) > rank_map.get(j, 999):
            selected.append(population[j])
        elif dist_map.get(i, 0) > dist_map.get(j, 0):
            selected.append(population[i])
        else:
            selected.append(population[j])

    return selected


# ═══════════════════════════════════════════════════════════════════════
# Main Optimizer
# ═══════════════════════════════════════════════════════════════════════

def evaluate_objectives(individual, predictor, feature_cols, base_features):
    """
    Map decision variables to predictions via ensemble, then compute objectives.
    base_features: dict of "typical" feature values to fill non-decision features.
    """
    # Build feature vector: start from base, override decision vars
    features = dict(base_features)

    # Map decision variables to relevant features
    var_map = {
        "Motor_Speed": "Mean_Motor_Speed",
        "Temperature": "Mean_Temperature",
        "Pressure": "Mean_Pressure",
        "Flow_Rate": "Mean_Flow_Rate",
        "Hold_Time": "Phase1_Processing_Duration",
    }

    for var_name, feat_name in var_map.items():
        idx = VAR_NAMES.index(var_name)
        if feat_name in features:
            features[feat_name] = individual[idx]

    # Build array in correct column order
    X = np.array([[features.get(c, 0.0) for c in feature_cols]])

    pred_mean, _ = predictor.predict_with_uncertainty(X)

    energy = pred_mean[0, 4]      # Total_Energy_kWh
    carbon = energy * CARBON_INTENSITY
    quality = pred_mean[0, 0]     # Hardness (higher = better)

    # All objectives minimized → negate quality
    return [float(energy), float(carbon), float(-quality)]


def nsga2_optimize(initial_params, predictor, feature_cols, base_features,
                   pop_size=40, n_generations=25, crossover_rate=0.8, mutation_rate=0.1):
    """
    Run NSGA-II optimization.
    Returns: (pareto_solutions, pareto_objectives, hypervolume_history)
    """
    random.seed(42)

    # Initialize population
    population = initialize_population(pop_size)

    # Insert initial params as first individual
    if initial_params:
        ind0 = [initial_params.get(v, (BOUNDS[v][0] + BOUNDS[v][1]) / 2) for v in VAR_NAMES]
        population[0] = ind0

    hypervolume_history = []
    reference_point = [800, 700, 0]  # worst case for hypervolume calc

    for gen in range(n_generations):
        # Parallel objective evaluation
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            objectives = list(executor.map(
                lambda ind: evaluate_objectives(ind, predictor, feature_cols, base_features),
                population,
            ))

        # Non-dominated sorting
        fronts = fast_non_dominated_sort(objectives)

        # Crowding distance for each front
        crowding_distances = []
        for front in fronts:
            dists = assign_crowding_distance(front, objectives)
            crowding_distances.append(dists)

        # Hypervolume
        pareto_objs = [objectives[i] for i in fronts[0]]
        hv = calculate_hypervolume(pareto_objs, reference_point)
        hypervolume_history.append(hv)

        # Selection
        parents = binary_tournament_selection(population, fronts, crowding_distances, pop_size)

        # Crossover
        offspring = []
        random.shuffle(parents)
        for i in range(0, len(parents) - 1, 2):
            if random.random() < crossover_rate:
                c1, c2 = sbx_crossover(parents[i], parents[i + 1])
                offspring.extend([c1, c2])
            else:
                offspring.extend([list(parents[i]), list(parents[i + 1])])

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = gaussian_mutation(offspring[i])

        # Elitism: combine parent + offspring, select best N
        combined = population + offspring
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            combined_obj = list(executor.map(
                lambda ind: evaluate_objectives(ind, predictor, feature_cols, base_features),
                combined,
            ))

        combined_fronts = fast_non_dominated_sort(combined_obj)
        new_pop = []
        for front in combined_fronts:
            if len(new_pop) + len(front) <= pop_size:
                for idx in front:
                    new_pop.append(combined[idx])
            else:
                dists = assign_crowding_distance(front, combined_obj)
                ranked = sorted(range(len(front)), key=lambda i: -dists[i])
                for i in ranked:
                    if len(new_pop) >= pop_size:
                        break
                    new_pop.append(combined[front[i]])
                break

        population = new_pop[:pop_size]

    # Final evaluation
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        final_objectives = list(executor.map(
            lambda ind: evaluate_objectives(ind, predictor, feature_cols, base_features),
            population,
        ))

    final_fronts = fast_non_dominated_sort(final_objectives)

    # Extract Pareto front solutions
    pareto_solutions = []
    pareto_objectives_list = []
    for idx in final_fronts[0]:
        sol = {VAR_NAMES[i]: round(population[idx][i], 2) for i in range(N_VARS)}
        sol["energy"] = round(final_objectives[idx][0], 2)
        sol["carbon"] = round(final_objectives[idx][1], 2)
        sol["quality"] = round(-final_objectives[idx][2], 2)  # un-negate
        pareto_solutions.append(sol)
        pareto_objectives_list.append(final_objectives[idx])

    return pareto_solutions, pareto_objectives_list, hypervolume_history


def select_balanced_solution(pareto_solutions):
    """Select the most balanced solution from the Pareto front."""
    if not pareto_solutions:
        return None

    # Normalize each objective to [0,1] then pick the one closest to ideal
    energies = [s["energy"] for s in pareto_solutions]
    carbons = [s["carbon"] for s in pareto_solutions]
    qualities = [s["quality"] for s in pareto_solutions]

    def normalize(vals):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-10:
            return [0.5] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    norm_e = normalize(energies)
    norm_c = normalize(carbons)
    norm_q = [1 - q for q in normalize(qualities)]  # higher quality = lower score

    scores = [0.4 * ne + 0.3 * nc + 0.3 * nq for ne, nc, nq in zip(norm_e, norm_c, norm_q)]
    best_idx = scores.index(min(scores))

    return pareto_solutions[best_idx]
