# Successive Bipartite Matching

C++ implementations and empirical analysis comparing exact and greedy/hybrid solvers for **successive maximum-weight bipartite matching** (finding $k$ edge-disjoint matchings / $k$-th maximum matching costs on random weighted bipartite graphs).

## Contents

- `kmax_network_simplex.cpp` — Network Simplex min-cost flow solver for computing $k$ maximum-weight matchings.
- `mcmf_vs_hung.cpp` — Head-to-head C++ benchmark comparing min-cost max-flow against successive Hungarian matching.
- `compare_hybrid_greedy.cpp` — Comparison of exact vs. hybrid/greedy matching heuristics.
- `match.cpp`, `match.py`, `deviation.cpp` — Simulation drivers for Uniform and Exponential edge-weight distributions and cost deviation measurements.
- `match.ipynb`, `compare.ipynb`, `analyze_compare_hybrid_greedy.ipynb` — Notebooks and generated empirical distribution plots (`K22`–`K77`, expected $k$-th matching cost, standard deviation, and skew/gradient analyses).

## Building (C++)

```bash
g++ -O3 -std=c++20 kmax_network_simplex.cpp -o kmax_network_simplex
g++ -O3 -std=c++20 mcmf_vs_hung.cpp -o mcmf_vs_hung
g++ -O3 -std=c++20 compare_hybrid_greedy.cpp -o compare_hybrid_greedy
```
