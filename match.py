import math
import random

def hungarian(a):
	if not a:
		return 0.0, []
	n = len(a)
	m = len(a[0]) if n else 0
	u = [0.0] * (n + 1)
	v = [0.0] * (m + 1)
	p = [0] * (m + 1)
	for i in range(1, n + 1):
		p[0] = i
		j0 = 0
		minv = [float('inf')] * (m + 1)
		used = [False] * (m + 1)
		way = [-1] * (m + 1)
		while True:
			used[j0] = True
			i0 = p[j0]
			delta = float('inf')
			j1 = 0
			for j in range(1, m + 1):
				if not used[j]:
					cur = a[i0 - 1][j - 1] - u[i0] - v[j]
					if cur < minv[j]:
						minv[j] = cur
						way[j] = j0
					if minv[j] < delta:
						delta = minv[j]
						j1 = j
			for j in range(m + 1):
				if used[j]:
					u[p[j]] += delta
					v[j] -= delta
				else:
					minv[j] -= delta
			j0 = j1
			if p[j0] == 0:
				break
		while True:
			j1 = way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0:
				break
	ans = [-1] * n
	for j in range(1, m + 1):
		if p[j]:
			ans[p[j] - 1] = j - 1
	return -v[0], ans

def expected_kth_values(n: int, trials: int = 10000, seed: int | None = None):
	# Expected total weight of the k-th max matching, k=1..n
	LARGE = 1e18
	avg = [0.0] * n
	rng = random.Random(seed) if seed is not None else random
	for _ in range(trials):
		mat = [[-rng.random() for _ in range(n)] for _ in range(n)]  # negate for max
		for k in range(n):
			cost, sol = hungarian(mat)
			avg[k] += -cost  # convert back to max-sum
			for i in range(n):
				mat[i][sol[i]] = LARGE  # forbid used edge
	return [x / trials for x in avg]

def kth_max_value_from_matrix(mat, k: int) -> float:
	LARGE = 1e18
	val = 0.0
	for _ in range(k):
		cost, sol = hungarian(mat)
		val = -cost
		for i in range(len(mat)):
			mat[i][sol[i]] = LARGE
	return val

def plot_kth_distribution(n: int, k: int, trials: int = 10000, bins: int = 50, seed: int | None = None):
	vals = []
	rng = random.Random(seed) if seed is not None else random
	for _ in range(trials):
		mat = [[-rng.random() for _ in range(n)] for _ in range(n)]
		vals.append(kth_max_value_from_matrix(mat, k))
	mean = sum(vals) / len(vals)
	std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
	print(f"k={k}, n={n}, trials={trials}: mean={mean:.6f}, std={std:.6f}")
	try:
		import matplotlib.pyplot as plt
		plt.figure()
		# Histogram (density)
		plt.hist(vals, bins=bins, density=True, alpha=0.75, edgecolor='black')
		# Range for overlays
		x_min, x_max = (min(vals), max(vals)) if vals else (mean - 3 * std, mean + 3 * std)
		# Overlay Normal PDF using the observed mean/std
		if std > 0 and x_max > x_min:
			num = 512
			step = (x_max - x_min) / (num - 1)
			xs = [x_min + i * step for i in range(num)]
			inv = 1.0 / (std * math.sqrt(2.0 * math.pi))
			pdf = [inv * math.exp(-0.5 * ((x - mean) / std) ** 2) for x in xs]
			plt.plot(xs, pdf, 'r-', lw=2, label=f"Normal(mean={mean:.3f}, std={std:.3f})")
		# Overlay Gamma(k, θ) PDF (fit via method-of-moments on positive samples)
		pos_vals = [x for x in vals if x > 0]
		if len(pos_vals) > 1:
			m = sum(pos_vals) / len(pos_vals)
			var = (sum((x - m) ** 2 for x in pos_vals) / len(pos_vals))
			if m > 0 and var > 0:
				k_shape = (m * m) / var
				theta = var / m
				if x_max > max(x_min, 0.0):
					num = 512
					gx_min = max(x_min, 0.0)
					step2 = (x_max - gx_min) / (num - 1) if x_max > gx_min else 1.0
					xs2 = [gx_min + i * step2 for i in range(num)]
					den = math.gamma(k_shape) * (theta ** k_shape)
					gpdf = [
						((x ** (k_shape - 1)) * math.exp(-x / theta) / den) if x > 0 else 0.0
						for x in xs2
					]
					plt.plot(xs2, gpdf, 'g--', lw=2, label=f"Gamma(k={k_shape:.3f}, θ={theta:.3f})")
		plt.legend()
		plt.title(f"Distribution of total weight of the {k}-th max matching\nK({n},{n}), Uniform[0,1], trials={trials}")
		plt.xlabel("Total weight")
		plt.ylabel("Density")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()
	except Exception:
		pass

def plot_k_range_distributions(n: int, k_start: int, k_end: int, trials: int = 10000, bins: int = 50, seed: int | None = None, alpha: float = 0.4):
	"""
	Overlay histograms (density) of total weights for k in [k_start, k_end].
	Prints mean/std per k and uses a shared bin range for visual comparability.
	Also overlays a Normal(mean_k, std_k) curve for each k.
	"""
	assert 1 <= k_start <= k_end <= n, "Require 1 <= k_start <= k_end <= n"
	vals_by_k: dict[int, list[float]] = {}
	mean_by_k: dict[int, float] = {}
	std_by_k: dict[int, float] = {}
	rng = random.Random(seed) if seed is not None else random

	min_v = float("inf")
	max_v = float("-inf")
	for k in range(k_start, k_end + 1):
		vals = []
		for _ in range(trials):
			mat = [[-rng.random() for _ in range(n)] for _ in range(n)]
			vals.append(kth_max_value_from_matrix(mat, k))
		vals_by_k[k] = vals
		if vals:
			min_v = min(min_v, min(vals))
			max_v = max(max_v, max(vals))
		mean = sum(vals) / len(vals)
		std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
		mean_by_k[k] = mean
		std_by_k[k] = std
		print(f"k={k}, n={n}, trials={trials}: mean={mean:.6f}, std={std:.6f}")

	try:
		import matplotlib.pyplot as plt
		plt.figure()
		# Draw histograms
		for k in range(k_start, k_end + 1):
			vals = vals_by_k[k]
			plt.hist(
				vals,
				bins=bins,
				range=(min_v, max_v) if min_v < max_v else None,
				density=True,
				alpha=alpha,
				label=f"k={k}",
				edgecolor="black",
				linewidth=0.5,
			)
		# Overlay Normal PDFs for each k on the same range
		x_min, x_max = (min_v, max_v) if min_v < max_v else (0.0, 1.0)
		if x_max > x_min:
			num = 512
			step = (x_max - x_min) / (num - 1)
			xs = [x_min + i * step for i in range(num)]
			for k in range(k_start, k_end + 1):
				mean = mean_by_k[k]
				std = std_by_k[k]
				if std <= 0:
					continue
				inv = 1.0 / (std * math.sqrt(2.0 * math.pi))
				pdf = [inv * math.exp(-0.5 * ((x - mean) / std) ** 2) for x in xs]
				plt.plot(xs, pdf, lw=2, label=f"Normal k={k} (μ={mean:.3f}, σ={std:.3f})")
		plt.title(f"Distributions of total weight for k-th max matching\nK({n},{n}), U[0,1], trials={trials}")
		plt.xlabel("Total weight")
		plt.ylabel("Density")
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()
	except Exception:
		pass

# New: compute mean and std for k=1..n in a single simulation pass
def compute_kth_mean_std(n: int, trials: int = 10000, seed: int | None = None):
	LARGE = 1e18
	sumv = [0.0] * n
	sumsq = [0.0] * n
	rng = random.Random(seed) if seed is not None else random
	for _ in range(trials):
		mat = [[-rng.random() for _ in range(n)] for _ in range(n)]
		for k in range(n):
			cost, sol = hungarian(mat)
			val = -cost
			sumv[k] += val
			sumsq[k] += val * val
			for i in range(n):
				mat[i][sol[i]] = LARGE
	means = [sv / trials for sv in sumv]
	stds = []
	for k in range(n):
		var = (sumsq[k] / trials) - (means[k] * means[k])
		if var < 0 and var > -1e-12:
			var = 0.0
		stds.append(math.sqrt(var) if var > 0 else 0.0)
	return means, stds

# New: plot expected value vs k
def plot_expected_vs_k(n: int, trials: int = 10000, seed: int | None = None):
	means, _ = compute_kth_mean_std(n, trials=trials, seed=seed)
	try:
		import matplotlib.pyplot as plt
		xs = list(range(1, n + 1))
		plt.figure()
		plt.plot(xs, means, marker='o', lw=2)
		plt.title(f"Expected total weight vs k\nK({n},{n}), U[0,1], trials={trials}")
		plt.xlabel("k")
		plt.ylabel("E[total weight]")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()
	except Exception:
		pass

# New: plot std vs k
def plot_std_vs_k(n: int, trials: int = 10000, seed: int | None = None):
	_, stds = compute_kth_mean_std(n, trials=trials, seed=seed)
	try:
		import matplotlib.pyplot as plt
		xs = list(range(1, n + 1))
		plt.figure()
		plt.plot(xs, stds, marker='o', lw=2, color='orange')
		plt.title(f"Std of total weight vs k\nK({n},{n}), U[0,1], trials={trials}")
		plt.xlabel("k")
		plt.ylabel("Std[total weight]")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()
	except Exception:
		pass