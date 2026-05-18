"""

VARIANTS IMPLEMENTED
--------------------
A. ACCEPTANCE PROBABILITY
   A1. Boltzmann (standard): P = exp(-delta / T)
   A2. Linear simplification: P = max(0, 1 - delta / T)
   A3. Discrete step-based:   P = 0.8 -> 0.6 -> 0.4 by iteration block

B. NEIGHBORHOOD OPERATOR
   B1. 2-opt swap (reverse a sub-segment)
   B2. Inversion (same as 2-opt for permutations -- alias)
   B3. Transposition (swap two positions)
   B4. Displacement (remove element, reinsert elsewhere)

C. COOLING SCHEDULE
   C1. Geometric:    T_{k+1} = T_k * alpha
   C2. Logarithmic:  T_k = T_0 * ln(2) / ln(k)
   C3. Lundy-Mees:   T_k = T_{k-1} / (1 + gamma * T_{k-1})
   C4. Fixed temp:   T stays constant (no cooling)
   C5. Reheating:    cool on accept, heat on reject

"""

import math
import random
import itertools
import time
import matplotlib.pyplot as plt


DIST = [
    [   0,  98,  62,  91,  59,  80,  94,  67,  66,  98],  # Node 1
    [  98,   0,  63,  84,  68,  94,  65,  56,  55,  56],  # Node 2
    [  62,  63,   0,  80,  82,  81,  53,  76,  75,  97],  # Node 3
    [  91,  84,  80,   0,  88,  73,  64,  68,  56,  79],  # Node 4
    [  59,  68,  82,  88,   0,  83,  83,  92,  74,  63],  # Node 5
    [  80,  94,  81,  73,  83,   0,  81,  82,  65,  96],  # Node 6
    [  94,  65,  53,  64,  83,  81,   0,  73,  75,  50],  # Node 7
    [  67,  56,  76,  68,  92,  82,  73,   0,  86,  85],  # Node 8
    [  66,  55,  75,  56,  74,  65,  75,  86,   0,  92],  # Node 9
    [  98,  56,  97,  79,  63,  96,  50,  85,  92,   0],  # Node 10
]
N = len(DIST)



def tour_distance(tour):
    """Total closed-tour distance."""
    return sum(DIST[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def format_tour(tour):
    """Render a 0-based tour as '1 -> X -> ... -> 1'."""
    return " -> ".join(str(n + 1) for n in tour) + f" -> {tour[0] + 1}"


def print_section(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def exhaustive_enumeration():
    """Enumerate all 9! tours; return the certified global optimum."""
    remaining = list(range(1, N))
    best_dist, best_tour = math.inf, None
    for perm in itertools.permutations(remaining):
        candidate = [0] + list(perm)
        d = tour_distance(candidate)
        if d < best_dist:
            best_dist, best_tour = d, candidate[:]
    return best_dist, best_tour


# All operators preserve index 0 (Node 1) as the fixed start.
# They each return a new tour list (do not mutate the input).

def neighbor_2opt(tour):
    """B1/B2. Reverse a random sub-segment between positions i and j (i<j)."""
    new_tour = tour[:]
    i = random.randint(1, N - 2)
    j = random.randint(i + 1, N - 1)
    new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
    return new_tour


def neighbor_transposition(tour):
    """B3. Swap two random positions (other than index 0)."""
    new_tour = tour[:]
    i, j = random.sample(range(1, N), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def neighbor_displacement(tour):
    """B4. Remove one node and reinsert it at a different position."""
    new_tour = tour[:]
    i = random.randint(1, N - 1)
    node = new_tour.pop(i)
    # Choose a new insert position different from i, but always after index 0
    j = random.randint(1, N - 1)
    while j == i:
        j = random.randint(1, N - 1)
    new_tour.insert(j, node)
    return new_tour


NEIGHBORHOODS = {
    "2opt":          neighbor_2opt,
    "transposition": neighbor_transposition,
    "displacement":  neighbor_displacement,
}


# Each returns a probability in [0, 1] for accepting a WORSE move (delta > 0).
# Better moves (delta < 0) are accepted unconditionally outside these.

def accept_boltzmann(delta, T, iteration, max_iter):
    """A1. Standard Metropolis: P = exp(-delta / T)."""
    if T <= 0:
        return 0.0
    return math.exp(-delta / T)


def accept_linear(delta, T, iteration, max_iter):
    """A2. Linear simplification: P = max(0, 1 - delta/T). Slide formula."""
    if T <= 0:
        return 0.0
    return max(0.0, 1.0 - delta / T)


def accept_discrete(delta, T, iteration, max_iter):
    """
    A3. Discrete step-based: probability depends only on iteration progress.
    Slide example was 100/200/300; we scale to max_iter so it applies to
    any run length. Three equal bands: 0.8, 0.6, 0.4.
    """
    progress = iteration / max(1, max_iter)
    if progress < 1 / 3:
        return 0.8
    elif progress < 2 / 3:
        return 0.6
    else:
        return 0.4


ACCEPTANCES = {
    "boltzmann": accept_boltzmann,
    "linear":    accept_linear,
    "discrete":  accept_discrete,
}


# Each function returns the next temperature given the current temperature,
# step index k (>=1 typically), and a params dict.

def cool_geometric(T, k, params):
    """C1. Geometric: T_{k+1} = T_k * alpha."""
    return T * params["alpha"]


def cool_logarithmic(T, k, params):
    """C2. Logarithmic: T_k = T_0 * ln(2) / ln(k+1).  Note k+1 to avoid ln(1)=0."""
    return params["T0"] * math.log(2) / math.log(k + 2)


def cool_lundy_mees(T, k, params):
    """C3. Lundy & Mees (1986): T_{k+1} = T_k / (1 + gamma * T_k)."""
    return T / (1.0 + params["gamma"] * T)


def cool_fixed(T, k, params):
    """C4. Fixed temperature: never change T."""
    return T


COOLINGS = {
    "geometric":   cool_geometric,
    "logarithmic": cool_logarithmic,
    "lundy_mees":  cool_lundy_mees,
    "fixed":       cool_fixed,
}


def simulated_annealing(
    neighborhood_name,
    acceptance_name,
    cooling_name,
    initial_temp        = 1000.0,
    min_temp            = 1e-3,
    iterations_per_temp = 50,
    max_total_iter      = 200_000,
    cooling_params      = None,
    reheating           = False,
    reheat_beta         = 0.2,
    reheat_gamma        = 0.1,
    seed                = None,
):
    
    if seed is not None:
        random.seed(seed)
    cooling_params = cooling_params or {}
    cooling_params.setdefault("T0", initial_temp)

    neighbor_fn   = NEIGHBORHOODS[neighborhood_name]
    accept_fn     = ACCEPTANCES[acceptance_name]
    cool_fn       = COOLINGS[cooling_name]

    # Initialise
    inner = list(range(1, N))
    random.shuffle(inner)
    current_tour = [0] + inner

    #calculate distance of current tour
    current_dist = tour_distance(current_tour)

    # Global best tracking
    best_tour, best_dist = current_tour[:], current_dist
    iters_to_best = 0
    convergence_history = [(0, best_dist)]

    T = initial_temp
    total_iters = 0
    k = 0  # temperature step counter

    # Outer loop: temperature levels
    while T > min_temp and total_iters < max_total_iter:
        for _ in range(iterations_per_temp):
            total_iters += 1

            # Neighbor generation
            neighbor = neighbor_fn(current_tour)
            # calculate distance of neighbor tour
            neighbor_dist = tour_distance(neighbor)
            # calculate delta of neighbor and the current tour
            delta = neighbor_dist - current_dist

            accepted = False
            if delta < 0:
                # Always accept improving moves because the neighbor is better
                current_tour, current_dist = neighbor, neighbor_dist
                accepted = True
            else:
                # Worse move (the neighbor is worse): probabilistic acceptance via chosen formula
                p = accept_fn(delta, T, total_iters, max_total_iter)
                if random.random() < p:
                    current_tour, current_dist = neighbor, neighbor_dist
                    accepted = True

            # Update global best if the current distance is better than the best found so far
            if current_dist < best_dist:
                best_dist     = current_dist
                best_tour     = current_tour[:]
                iters_to_best = total_iters
                convergence_history.append((total_iters, best_dist))

            # C5 reheating: adjust T immediately based on acceptance outcome
            if reheating:
                if accepted:
                    T = T / (1.0 + reheat_beta)
                else:
                    T = T / (1.0 - reheat_gamma)
                # Safety: prevent runaway heating
                T = min(T, initial_temp * 10)

            if total_iters >= max_total_iter:
                break

        # Per-temperature-level cooling (skipped for reheating mode and fixed)
        if not reheating:
            k += 1
            T = cool_fn(T, k, cooling_params)

    return {
        "best_dist":           best_dist,
        "best_tour":           best_tour,
        "iters_to_best":       iters_to_best,
        "total_iters":         total_iters,
        "convergence_history": convergence_history,
    }


# Each "variant" is a complete SA configuration: a combination of mechanics +
# parameters. We hold most parameters constant so we can isolate the effect
# of changing one knob at a time.

# Shared "base" parameters used across most variants
BASE_T0       = 1000.0
BASE_TMIN     = 1e-3
BASE_L        = 50
BASE_MAXITER  = 200_000

# Each entry: (label, kwargs for simulated_annealing)
VARIANTS = [
    # Reference: standard SA
    ("V0  Standard (Boltzmann + 2opt + Geometric)", {
        "neighborhood_name":   "2opt",
        "acceptance_name":     "boltzmann",
        "cooling_name":        "geometric",
        "cooling_params":      {"alpha": 0.995},
    }),

    # A. Acceptance probability variants
    ("V1  Acceptance: Linear  (P = 1 - d/T)", {
        "neighborhood_name":   "2opt",
        "acceptance_name":     "linear",
        "cooling_name":        "geometric",
        "cooling_params":      {"alpha": 0.995},
    }),
    ("V2  Acceptance: Discrete (0.8/0.6/0.4)", {
        "neighborhood_name":   "2opt",
        "acceptance_name":     "discrete",
        "cooling_name":        "geometric",
        "cooling_params":      {"alpha": 0.995},
    }),

    # B. Neighborhood operator variants
    ("V3  Neighborhood: Transposition (swap)", {
        "neighborhood_name":   "transposition",
        "acceptance_name":     "boltzmann",
        "cooling_name":        "geometric",
        "cooling_params":      {"alpha": 0.995},
    }),
    ("V4  Neighborhood: Displacement (insertion)", {
        "neighborhood_name":   "displacement",
        "acceptance_name":     "boltzmann",
        "cooling_name":        "geometric",
        "cooling_params":      {"alpha": 0.995},
    }),

    # C. Cooling schedule variants
    ("V5  Cooling: Logarithmic", {
        "neighborhood_name":   "2opt",
        "acceptance_name":     "boltzmann",
        "cooling_name":        "logarithmic",
        "cooling_params":      {},
    }),
    ("V6  Cooling: Lundy-Mees", {
        "neighborhood_name":   "2opt",
        "acceptance_name":     "boltzmann",
        "cooling_name":        "lundy_mees",
        "cooling_params":      {"gamma": 0.001},
    }),
    ("V7  Cooling: Fixed Temperature (T=50)", {
        "neighborhood_name":   "2opt",
        "acceptance_name":     "boltzmann",
        "cooling_name":        "fixed",
        "cooling_params":      {},
        "initial_temp":        50.0,   # override -- pick a "good" fixed T
    }),
    ("V8  Reheating (cool on accept, heat on reject)", {
        "neighborhood_name":   "2opt",
        "acceptance_name":     "boltzmann",
        "cooling_name":        "geometric",   # ignored when reheating=True
        "cooling_params":      {"alpha": 0.995},
        "reheating":           True,
        "reheat_beta":         0.2,
        "reheat_gamma":        0.1,
    }),
]


def run_variant(label, kwargs, n_runs):
    """Execute n_runs independent SA runs for one variant configuration."""
    # Fill in shared defaults that the variant didn't override
    kwargs = dict(kwargs)
    kwargs.setdefault("initial_temp",        BASE_T0)
    kwargs.setdefault("min_temp",            BASE_TMIN)
    kwargs.setdefault("iterations_per_temp", BASE_L)
    kwargs.setdefault("max_total_iter",      BASE_MAXITER)

    distances, iters_list, tours, histories, times = [], [], [], [], []
    for run in range(n_runs):
        t0 = time.perf_counter()
        result = simulated_annealing(seed=run, **kwargs)
        elapsed = time.perf_counter() - t0
        distances.append(result["best_dist"])
        iters_list.append(result["iters_to_best"])
        tours.append(result["best_tour"])
        histories.append(result["convergence_history"])
        times.append(elapsed)

    n      = len(distances)
    mean_d = sum(distances) / n
    std_d  = math.sqrt(sum((d - mean_d) ** 2 for d in distances) / n)
    best_i = distances.index(min(distances))

    return {
        "label":            label,
        "kwargs":           kwargs,
        "distances":        distances,
        "iters_list":       iters_list,
        "tours":            tours,
        "histories":        histories,
        "times":            times,
        "best_dist":        min(distances),
        "worst_dist":       max(distances),
        "mean_dist":        mean_d,
        "std_dist":         std_d,
        "mean_iters":       sum(iters_list) / n,
        "mean_time":        sum(times) / n,
        "best_tour":        tours[best_i],
        "best_run_history": histories[best_i],
    }



def plot_variants(results, opt_dist, save_path):
    """Two-panel comparison plot covering all variants."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = plt.cm.tab10.colors

    # ----- Left: best-run convergence per variant (step plot) -----
    ax1 = axes[0]
    for idx, res in enumerate(results):
        h = res["best_run_history"]
        xs = [e[0] for e in h] + [BASE_MAXITER]
        ys = [e[1] for e in h] + [h[-1][1]]
        ax1.step(xs, ys, where="post", color=colors[idx % 10], linewidth=1.8,
                 label=f"{res['label'][:42]}  ({res['best_dist']})")

    ax1.axhline(y=opt_dist, color="green", linestyle="--", linewidth=2,
                label=f"Global Optimum = {opt_dist}", zorder=10)
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Best-So-Far Tour Distance", fontsize=11)
    ax1.set_title("Convergence -- Best Run of Each Variant", fontsize=12)
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("linear")

    # ----- Right: bar chart of mean +/- std distance per variant -----
    ax2 = axes[1]
    labels    = [r["label"][:42] for r in results]
    means     = [r["mean_dist"] for r in results]
    stds      = [r["std_dist"]  for r in results]
    bests     = [r["best_dist"] for r in results]
    bar_colors = [colors[i % 10] for i in range(len(results))]

    y_pos = list(range(len(results)))
    bars = ax2.barh(y_pos, means, xerr=stds, color=bar_colors, alpha=0.7,
                    edgecolor="black", linewidth=0.8, capsize=4,
                    error_kw={"linewidth": 1.2})
    # Mark best-of-N with a black tick
    for i, b in enumerate(bests):
        ax2.scatter(b, i, color="black", marker="|", s=200, zorder=5,
                    label="Best of 20" if i == 0 else "")

    ax2.axvline(x=opt_dist, color="green", linestyle="--", linewidth=2,
                label=f"Global Optimum = {opt_dist}", zorder=10)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Tour Distance  (mean +/- sigma; black bar = best)", fontsize=11)
    ax2.set_title("Solution Quality Across Variants  (20 runs each)", fontsize=12)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Simulated Annealing Variants -- TSP Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Graph saved to: {save_path}")


def main():
    print("  TSP SOLVER -- SIMULATED ANNEALING VARIANTS COMPARISON")
    print("  10-Node Symmetric Problem  |  9 SA variants  |  20 runs each")

    #Global optimum (reference)
    print_section("PART A: GLOBAL OPTIMUM  (Exhaustive Enumeration)")
    print(f"\n  Enumerating {math.factorial(9):,} tours ...")
    t0 = time.perf_counter()
    opt_dist, opt_tour = exhaustive_enumeration()
    t_enum = time.perf_counter() - t0
    print(f"  Time     : {t_enum:.3f} s")
    print(f"  Tour     : {format_tour(opt_tour)}")
    print(f"  Distance : {opt_dist}")

    print_section("PART B: RUNNING ALL VARIANTS  (20 independent runs each)")
    N_RUNS = 20
    results = []
    t0 = time.perf_counter()
    for label, kwargs in VARIANTS:
        print(f"  Running {label} ...")
        res = run_variant(label, kwargs, N_RUNS)
        results.append(res)
    t_all = time.perf_counter() - t0
    print(f"\n  Total time across all variants: {t_all:.1f} s")

    # Comparison table
    print_section("PART C: COMPARATIVE RESULTS TABLE")
    print()
    hdr = f"  {'Variant':<48}{'Best':>6}{'Mean':>9}{'Worst':>7}{'Sigma':>8}{'Iter@Best':>11}{'Gap%':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for res in results:
        gap_pct = (res["mean_dist"] - opt_dist) / opt_dist * 100
        print(f"  {res['label']:<48}"
              f"{res['best_dist']:>6}"
              f"{res['mean_dist']:>9.2f}"
              f"{res['worst_dist']:>7}"
              f"{res['std_dist']:>8.2f}"
              f"{res['mean_iters']:>11,.0f}"
              f"{gap_pct:>6.2f}%")

    # Best tour per variant
    print_section("PART D: BEST TOUR PER VARIANT")
    for res in results:
        match = "  *** OPTIMUM ***" if res["best_dist"] == opt_dist else ""
        print(f"\n  {res['label']}{match}")
        print(f"    {format_tour(res['best_tour'])}   dist = {res['best_dist']}")

    # Per-category rankings
    print_section("PART E: WHICH VARIANT WINS EACH CRITERION?")
    by_best   = sorted(results, key=lambda r: r["best_dist"])
    by_mean   = sorted(results, key=lambda r: r["mean_dist"])
    by_sigma  = sorted(results, key=lambda r: r["std_dist"])
    by_speed  = sorted(results, key=lambda r: r["mean_iters"])

    print("\n  Best-of-20 Distance  (lower is better)")
    for r in by_best[:3]:
        print(f"    {r['best_dist']:>4}  {r['label']}")

    print("\n  Mean Distance  (lower is better -- the 'average quality' winner)")
    for r in by_mean[:3]:
        print(f"    {r['mean_dist']:>7.2f}  {r['label']}")

    print("\n  Consistency  (lower sigma is better -- 'will I get the same result?')")
    for r in by_sigma[:3]:
        print(f"    {r['std_dist']:>5.2f}  {r['label']}")

    print("\n  Convergence Speed  (fewer iterations to reach best is better)")
    for r in by_speed[:3]:
        print(f"    {r['mean_iters']:>8,.0f}  {r['label']}")

    # Convergence graph
    print_section("PART F: CONVERGENCE GRAPH")
    plot_variants(results, opt_dist,
                  save_path="./variants_comparison.png")

    standard_label = "V0  Standard (Boltzmann + 2opt + Geometric)"
    standard_result = next((r for r in results if r["label"] == standard_label), None)
    if standard_result is not None:
        plot_variants([standard_result], opt_dist=opt_dist,
                      save_path="./standard_convergence.png")


if __name__ == "__main__":
    main()