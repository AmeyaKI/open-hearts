"""Headline result 1: how good are the guesses, at each evidence level?

Simulates heuristic-vs-heuristic games from a fixed master seed, records them
to a flat file, then scores belief tables against the true deal at every
completed-trick boundary. Writes results/guessing.txt and results/guessing.png.
"""
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from openhearts.belief.table import Level  # noqa: E402
from openhearts.engine.game import deal, play_game  # noqa: E402
from openhearts.eval import guessing  # noqa: E402
from openhearts.eval.records import (  # noqa: E402
    read_records, record_from, write_records,
)
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402

MASTER_SEED = 2026
NUM_GAMES = 2000
RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
RECORDS_PATH = os.path.join(RESULTS, "heuristic_games.txt")
TABLE_PATH = os.path.join(RESULTS, "guessing.txt")
PLOT_PATH = os.path.join(RESULTS, "guessing.png")

METRICS = [
    ("meanP", "mean P(truth)"),
    ("NLL", "mean -ln P(truth)"),
    ("top1", "top-1 accuracy"),
]
LEVELS = [Level.UNIFORM, Level.VOIDS, Level.FULL]


def simulate(num_games):
    recs = []
    for i in range(num_games):
        seed = MASTER_SEED + i
        state = deal(np.random.default_rng(seed))
        initial = tuple(state.hands)
        final = play_game(state, [HeuristicPlayer() for _ in range(4)])
        recs.append(record_from(seed=seed, initial_hands=initial,
                                final_state=final))
    return recs


def write_table(path, results, num_games):
    with open(path, "w") as f:
        f.write(f"# master_seed={MASTER_SEED} games={num_games}\n")
        f.write("# trick k = belief state entering trick k "
                "(k=1 is the fresh deal)\n")
        f.write("level trick meanP NLL top1\n")
        for level in LEVELS:
            for k in range(guessing.NUM_TRICKS):
                p, nll, top1 = results[level][k]
                f.write(f"{level.value} {k + 1} "
                        f"{p:.6f} {nll:.6f} {top1:.6f}\n")


def plot(path, results, num_games):
    tricks = np.arange(1, guessing.NUM_TRICKS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for j, (short, label) in enumerate(METRICS):
        ax = axes[j]
        for level in LEVELS:
            ax.plot(tricks, results[level][:, j], marker="o",
                    label=level.value)
        ax.set_xlabel("trick number (belief state entering trick)")
        ax.set_ylabel(label)
        ax.set_title(short)
        ax.grid(alpha=0.3)
        ax.legend(title="evidence level")
    fig.suptitle(
        f"Guessing quality by evidence level "
        f"({num_games} heuristic games, master seed {MASTER_SEED})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_GAMES
    os.makedirs(RESULTS, exist_ok=True)

    t0 = time.time()
    recs = simulate(num_games)
    write_records(RECORDS_PATH, recs)
    recs = read_records(RECORDS_PATH)  # evaluate exactly what was written
    print(f"simulated+recorded {len(recs)} games in {time.time() - t0:.1f}s",
          flush=True)

    results = {}
    for level in LEVELS:
        t1 = time.time()
        results[level] = guessing.run(recs, level)
        print(f"  {level.value}: {time.time() - t1:.1f}s", flush=True)

    write_table(TABLE_PATH, results, len(recs))
    plot(PLOT_PATH, results, len(recs))
    print(f"total {time.time() - t0:.1f}s -> {TABLE_PATH}, {PLOT_PATH}")

    for level in LEVELS:
        print(f"\n{level.value}:  trick   meanP      NLL      top1")
        for k in range(guessing.NUM_TRICKS):
            p, nll, top1 = results[level][k]
            print(f"         {k + 1:>6} {p:8.4f} {nll:8.4f} {top1:8.4f}")


if __name__ == "__main__":
    main()
