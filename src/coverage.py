from __future__ import division
import random
from plot_utils import *
from plotly_utils import plot_multiple_line_plotly

def gen_rand_distribution(n):
    res = []
    sum = 0
    for i in range(n):
        p = random.random()
        res.append(p)
        sum += p
    for i in range(n):
        res[i] /= sum
    return res

def compute_coverage(distribution, num_samples):
    n = len(distribution)
    c_distribution = [0 for i in range(n+1)]
    for i in range(1,n+1):
        c_distribution[i] = c_distribution[i-1] + distribution[i-1]
    covered = [False for i in range(n)]
    for i in range(num_samples):
        p = random.random()
        for j in range(1,n+1):
            if p >= c_distribution[j-1] and p < c_distribution[j]:
                covered[j-1] = True
                break
    res = 0
    for i in range(n):
        if covered[i]:
            res += distribution[i]
    return res

def min_samples(coverage, num_clusters):
    r = 1
    while True:
        p = 1 - num_clusters / (r + 1) * (1 - 1 / (r + 1))**r
        if p >= coverage:
            return r
        r += 1
    return None

if __name__ == '__main__':
    xs = []
    ys = []
    legends = []
    num_distributions = 100
    num_trials = 100
    for num_categories in [2, 5, 10, 20, 50, 100]:#range(2, 20, 4):
        y = []
        xs.append(range(15, 205, 15))
        coverages = [0 for i in range(len(xs[-1]))]
        for distribution in range(num_distributions):
            if (distribution + 1) % 10 == 0:
                print num_categories, distribution
            dis = gen_rand_distribution(num_categories)
            for trial in range(num_trials):
                for num_samples_index in range(len(xs[-1])):
                    num_samples = xs[-1][num_samples_index]
                    coverage = compute_coverage(dis, num_samples)
                    coverages[num_samples_index] += coverage
        ys.append([c*1.0/(num_trials*num_distributions) for c in coverages])
        legends.append('#categories = {}'.format(num_categories))
    fig_name = 'num_items_clustering_phase_simulation'
    plot_multiple_line(
        xs, ys, None, '# samples', 'Coverage', legends, fig_name)
    plot_multiple_line_plotly(
        xs, ys, None, '# samples', 'Coverage', legends, fig_name)
    raise SystemExit
