// computing k matchings with maximum weight
// using network simplex algorithm

#include <random>
#pragma GCC optimize("O3,unroll-loops")
#include <bits/stdc++.h>
 
using namespace std;
using ll = long long;
using db = double;
const int inf = 1e9;
mt19937 mt(random_device{}());

struct NetworkSimplex {
    using Flow = int;
    using Cost = ll;
    struct Edge {
        int nxt, to;
        Flow cap;
        Cost cost;
    };
    vector<Edge> edges;
    vector<int> head, fa, fe, mark, cyc;
    vector<Cost> dual;
    int ti;

    NetworkSimplex(int n)
            : head(n, 0), fa(n), fe(n), mark(n), cyc(n + 1), dual(n), ti(0) {
        edges.push_back({0, 0, 0, 0});
        edges.push_back({0, 0, 0, 0});
    }

    int ae(int u, int v, Flow cap, Cost cost) {
        assert(size(edges) % 2 == 0);
        int e = size(edges);
        edges.push_back({head[u], v, cap, cost});
        head[u] = e;
        edges.push_back({head[v], u, 0, -cost});
        head[v] = e + 1;
        return e;
    }

    void init_tree(int x) {
        mark[x] = 1;
        for (int i = head[x]; i; i = edges[i].nxt) {
            int v = edges[i].to;
            if (!mark[v] and edges[i].cap) {
                fa[v] = x, fe[v] = i;
                init_tree(v);
            }
        }
    }

    Cost phi(int x) {
        if (mark[x] == ti) return dual[x];
        return mark[x] = ti, dual[x] = phi(fa[x]) - edges[fe[x]].cost;
    }

    void push_flow(int e, Cost &cost) {
        int pen = edges[e ^ 1].to, lca = edges[e].to;
        ti++;
        while (pen) mark[pen] = ti, pen = fa[pen];
        while (mark[lca] != ti) mark[lca] = ti, lca = fa[lca];

        int e2 = 0, f = edges[e].cap, path = 2, clen = 0;
        for (int i = edges[e ^ 1].to; i != lca; i = fa[i]) {
            cyc[++clen] = fe[i];
            if (edges[fe[i]].cap < f)
                f = edges[fe[e2 = i] ^ (path = 0)].cap;
        }
        for (int i = edges[e].to; i != lca; i = fa[i]) {
            cyc[++clen] = fe[i] ^ 1;
            if (edges[fe[i] ^ 1].cap <= f)
                f = edges[fe[e2 = i] ^ (path = 1)].cap;
        }
        cyc[++clen] = e;

        for (int i = 1; i <= clen; ++i) {
            edges[cyc[i]].cap -= f, edges[cyc[i] ^ 1].cap += f;
            cost += edges[cyc[i]].cost * f;
        }
        if (path == 2)  return;

        int laste = e ^ path, last = edges[laste].to, cur = edges[laste ^ 1].to;
        while (last != e2) {
            mark[cur]--;
            laste ^= 1;
            swap(laste, fe[cur]);
            swap(last, fa[cur]); swap(last, cur);
        }
    }

    Cost compute() {
        Cost cost = 0;
        init_tree(0);
        mark[0] = ti = 2, fa[0] = cost = 0;
        int ncnt = size(edges) - 1;
        for (int i = 2, pre = ncnt; i != pre; i = i == ncnt ? 2 : i + 1)
            if (edges[i].cap && edges[i].cost < phi(edges[i ^ 1].to) - phi(edges[i].to))
                push_flow(pre = i, cost);
        ti++;
        for (int u = 0; u < size(dual); ++u)
            phi(u);
        return cost;
    }
};
 
int main() {
    cin.tie(0)->sync_with_stdio(0);signed
 
    const int n = 50, k = 20;
    auto next_double = [&] () { 
        return uniform_real_distribution<db>(0, 1)(mt); 
    };
    const int reps = 100;
    db tot;
    for (int _ = 0; _ < reps; _++) {
        vector mat(n, vector<db>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mat[i][j] = -next_double();
            }
        }
        int s = 2 * n, t = 2 * n + 1;
        
        NetworkSimplex sim(2 * n + 2);
        for (int i = 0; i < n; i++) {
            sim.ae(s, i, k, 0);
            sim.ae(i + n, t, k, 0);
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sim.ae(i, j + n, 1, mat[i][j] * inf);
            }
        }
        sim.ae(t, s, n * k, -inf);
        db cost = (sim.compute() + 1ll * n * k * inf) * 1.0 / inf;
        tot += -cost;
    }   
    cout << setprecision(12) << fixed << tot / reps << '\n';
}