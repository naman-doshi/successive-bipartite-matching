#include <bits/stdc++.h>
using namespace std;

enum class Distribution { Uniform, Exponential };

// Hungarian algorithm for square cost matrix (min-cost assignment).
// Returns pair(min_cost, assignment), assignment[i] = column index matched to row i.
static pair<double, vector<int>> hungarian_min_cost(const vector<vector<double>>& a) {
    int n = (int)a.size();
    if (n == 0) return {0.0, {}};
    int m = (int)a[0].size();
    // assume square
    vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    vector<int> p(m + 1, 0), way(m + 1, -1);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<double> minv(m + 1, numeric_limits<double>::infinity());
        vector<char> used(m + 1, false);
        way.assign(m + 1, -1);
        while (true) {
            used[j0] = true;
            int i0 = p[j0];
            double delta = numeric_limits<double>::infinity();
            int j1 = 0;
            // relax edges
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            // update potentials
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if (p[j0] == 0) break;
        }
        // augment
        while (true) {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if (j0 == 0) break;
        }
    }
    vector<int> assignment(n, -1);
    for (int j = 1; j <= m; ++j) {
        if (p[j] != 0) assignment[p[j] - 1] = j - 1;
    }
    double min_cost = -v[0];
    return {min_cost, assignment};
}

// Build an n x n cost matrix with negative weights drawn from the chosen distribution.
// Maximization of total weight is turned into minimization by negating the draws.
static vector<vector<double>> random_matrix(int n, mt19937_64& rng, Distribution dist, double rate) {
    vector<vector<double>> mat(n, vector<double>(n));
    if (dist == Distribution::Uniform) {
        uniform_real_distribution<double> uni(0.0, 1.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                mat[i][j] = -uni(rng);
    } else if (dist == Distribution::Exponential) {
        if (!(rate > 0.0)) throw invalid_argument("rate must be > 0 for exponential distribution");
        exponential_distribution<double> expo(rate);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                mat[i][j] = -expo(rng);
    } else {
        throw invalid_argument("unknown distribution");
    }
    return mat;
}

// Compute: mean_max(n) - predicted_n, where predicted_n comes from least-squares
// fit on points (x, mean_max(x)) for x = 1..(n-1), extrapolated to x=n.
double deviation_of_max_matching_at_n(int n,
                                      int trials = 10000,
                                      uint64_t seed = std::random_device{}(),
                                      Distribution dist = Distribution::Uniform,
                                      double rate = 1.0) {
    if (n < 1) throw invalid_argument("n must be >= 1");
    mt19937_64 rng(seed);

    vector<double> means(n, 0.0);
    for (int m = 1; m <= n; ++m) {
        double sum = 0.0;
        for (int t = 0; t < trials; ++t) {
            auto mat = random_matrix(m, rng, dist, rate);
            auto res = hungarian_min_cost(mat);
            double total_weight = -res.first; // negate min-cost to get max total weight
            sum += total_weight;
        }
        means[m - 1] = sum / trials;
    }

    if (n == 1) return 0.0;

    int m = n - 1;
    double xbar = (m + 1) / 2.0; // mean of 1..m
    double ybar = 0.0;
    for (int x = 1; x <= m; ++x) ybar += means[x - 1];
    ybar /= m;

    double sxx = 0.0;
    for (int x = 1; x <= m; ++x) {
        double dx = x - xbar;
        sxx += dx * dx;
    }

    double slope = 0.0;
    if (m >= 2 && sxx > 0.0) {
        double sxy = 0.0;
        for (int x = 1; x <= m; ++x) {
            sxy += (x - xbar) * (means[x - 1] - ybar);
        }
        slope = sxy / sxx;
    }
    double intercept = ybar - slope * xbar;
    double pred_n = intercept + slope * n;
    return means[n - 1] - pred_n;
}

// Optional CLI for quick testing.
// Build with: g++ -O2 -std=c++17 deviation.cpp -o deviation
// Usage: ./deviation n trials [uniform|exponential] [rate] [seed]
int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "usage: " << argv[0] << " n trials [uniform|exponential] [rate] [seed]\n";
        cerr << "example: " << argv[0] << " 20 5000 uniform\n";
        return 1;
    }
    int n = stoi(argv[1]);
    int trials = stoi(argv[2]);
    string d = (argc >= 4 ? argv[3] : "uniform");
    double rate = (argc >= 5 ? stod(argv[4]) : 1.0);
    uint64_t seed = (argc >= 6 ? stoull(argv[5]) : std::random_device{}());

    Distribution dist = Distribution::Uniform;
    if (d == "uniform") dist = Distribution::Uniform;
    else if (d == "exponential") dist = Distribution::Exponential;
    else {
        cerr << "unknown distribution: " << d << "\n";
        return 2;
    }

    double dev = deviation_of_max_matching_at_n(n, trials, seed, dist, rate);
    cout.setf(std::ios::fixed); cout << setprecision(9) << dev << "\n";
    return 0;
}
