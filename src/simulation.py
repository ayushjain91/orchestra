from random import randint, random as rand
from os.path import isfile
from numpy.random import multinomial, dirichlet
from numpy import nonzero, unique
from classes import *
from graph import *
from evaluation_utils import *
from read_utils import *
from collections import defaultdict
from plot_utils import *


def generate_items(num_items=35, hit_num=1,
                   num_perspectives=4, num_categories=8, overwrite=False):
    hit_filename = "../data/simulation/hit{}.txt".format(hit_num)
    if isfile(hit_filename) and not overwrite:
        raise NameError("Trying to overwrite simulation items." +
                        " Please ensure you have a copy")
    with open(hit_filename, "w") as hit_file:
        for item in ranged(num_items):
            item_name = str(item) + "_"
            for perspective in range(num_perspectives):
                leaf_num = randint(7, 14)
                item_name += "perspective{}leaf{}_".format(
                    perspective, str(leaf_num).zfill(2))
                if perspective == 0:
                    item_name = str(leaf_num) + '/' + item_name
            item_name = item_name[:-1]
            hit_file.write(item_name + "\n")


def generate_worker_response(
        num_responses=15, affinities=[0.7, 0.1, 0.1, 0.1],
        hit_num=1, overwrite=False, poor_worker_prob=0.0,
        poor_worker_quality=0.2):
    # Steps:
    # 1. Choose perspective
    # 2. Choose granularity:
    #   (a) Start with children of root in list
    #   (b) For node in list, decide if node is to be split or not.
    #       If split, remove it and add its children to list
    #   (c) For every node that is not split,
    #           create cluster
    #           For every descendant of node
    #               For every item belonging to descendant
    #                   add to current cluster

    hit_filename = "../data/simulation/hit{}.txt".format(hit_num)
    string = ''

    for worker in range(num_responses):
        perspective = nonzero(multinomial(1, affinities))[0][0]
        nodes = [1, 2]
        clustering = []
        while nodes:
            current_node = nodes.pop()
            to_be_split = False
            if current_node * 2 + 1 < 15:
                if rand() > 0.5:
                    to_be_split = True
                    nodes.append(current_node * 2 + 1)
                    nodes.append(current_node * 2 + 2)
            if to_be_split:
                continue
            descendant_cands = [current_node]
            descendants = []
            while descendant_cands:
                descendant_cand = descendant_cands.pop()
                if descendant_cand * 2 + 1 >= 15:
                    descendants.append(descendant_cand)
                else:
                    descendant_cands.append(descendant_cand * 2 + 1)
                    descendant_cands.append(descendant_cand * 2 + 2)

            cluster = []
            with open(hit_filename, "r") as hit_file:
                all_lines = hit_file.readlines()
                for item_line in all_lines:
                    item = item_line.strip()
                    item_id = int(item.split('_')[0].split('/')[1])
                    item_leaf_num = int(item.split('_')[perspective + 1][-2:])
                    if item_leaf_num in descendants:
                        cluster.append(item_id + 1)
            clustering.append(cluster)

        if rand() < poor_worker_prob:
            items = []
            for i in range(len(clustering)):
                for it in clustering[i]:
                    items.append((it, i))
            for it, cluster_index in items:
                if rand() >= poor_worker_quality:
                    continue
                new_cluster = randint(0, len(clustering) - 1)
                if new_cluster >= cluster_index:
                    new_cluster += 1
                clustering[cluster_index].remove(it)
                clustering[new_cluster].append(it)

        clustering_str = '[]'
        for cluster in clustering:
            cluster_string = '['
            cluster_string += ','.join([str(item) for item in cluster])
            cluster_string += ']'
            clustering_str += cluster_string
        string += str(hit_num) + "," + clustering_str + ",1000\n"
    result_filename = "../data/simulation/results/results.txt"
    if isfile(result_filename) and not overwrite:
        raise NameError("Worker responses already constructed")
    result_file = open("../data/simulation/results/results.txt", "w")
    result_file.write(string)
    result_file.close()


def run_process(hits_directory,
                sample_responses=None):

    num_clustering_hits, items_map, items = read_hits(hits_directory)

    results_file = hits_directory + 'results/results.txt'
    curr_hierarchy = None
    for hit_num in range(1, num_clustering_hits + 1):
        clusterings = read_worker_answers(
            results_file, hit_num, items_map, num_hits_reqd=sample_responses)
        graph = Graph(clusterings)
        cliques, res2 = graph.find_maximal_cliques()
        if curr_hierarchy is None:
            curr_hierarchy = Hierarchy(cliques[0])
        else:
            curr_hierarchy.merge_hierarchy(Hierarchy(cliques[0]))

    frontier = curr_hierarchy.maxLikelihoodFrontier()

    # Assign clusters to items
    for node in frontier:
        for item in node.item_array:
            item.cluster = node

    all_items = list(items)
    urls, clusters, ground_truth_labels, metrics = evaluate(
        all_items)
    metrics['clique_size'] = len(cliques[0])
    return urls, clusters, ground_truth_labels, metrics


def evaluate(all_items):
    items = [item.url for item in all_items]
    clusters = [item.cluster for item in all_items]
    ground_truth_labels = [item.ground_truth_label for item in all_items]

    precision = calc_precision(clusters, ground_truth_labels)
    recall = calc_recall(clusters, ground_truth_labels)
    accuracy = calc_accuracy(clusters, ground_truth_labels)

    unclustered_items = len(
        [item for item in all_items if item.cluster is None])

    metrics = {'Pr': precision, 'Re': recall,
               'Ac': accuracy, 'Ui': unclustered_items}
    return items, clusters, ground_truth_labels, metrics


def num_workers_simulation():
    dirichlet_priors = [0.3, 0.5, 0.7]  # [0.3, 0.6, 0.9]
    vals = [3, 6, 9, 12, 15, 18]
    num_trials = 200
    hits_directory = '../data/simulation/'

    xs = []
    ys = []
    legends = []

    for prior in dirichlet_priors:
        probability_dominant_pers = defaultdict(int)
        for trial in range(num_trials):
            if (trial + 1) % 10 == 0:
                print prior, trial
            # affinities = list(dirichlet([prior]*4))
            affinities = [prior] + [(1.0 - prior) / 3] * 3
            affinities.sort(reverse=True)
            for num_responses in vals:
                generate_worker_response(num_responses=num_responses,
                                         overwrite=True,
                                         affinities=affinities)
                items, clusters, labels, metrics = run_process(hits_directory)
                if metrics['Ac'] == 1.0 and metrics['Re'] == 1.0:
                    probability_dominant_pers[num_responses] += 1
        keys = probability_dominant_pers.keys()
        keys.sort()
        xs.append(keys)
        ys.append([probability_dominant_pers[key] * 1.0 / 100 for key in keys])
        legends.append('d = ' + str(prior))
        print xs, ys

    fig_name = 'num_workers_simulation'
    plot_multiple_line(
        xs, ys, None, '# workers', 'Pr(T_{max}=T_{ml})', legends, fig_name)


def worker_errors_simulation():
    poor_worker_probs = [0, 0.2, 0.4, 0.6, 0.8, 1]
    poor_worker_qualities = [0, 0.1, 0.3, 0.5, 0.7]
    hits_directory = '../data/simulation/'

    xs = []
    accuracies = []
    clique_sizes = []
    legends = []
    num_trials = 100

    for prob in poor_worker_probs:
        accuracy = []
        clique_size = []
        for quality in poor_worker_qualities:
            accuracy.append(0)
            clique_size.append(0)
            for trial in range(num_trials):
                if (trial + 1) % 10 == 0:
                    print prob, quality, trial
                generate_worker_response(overwrite=True,
                                         poor_worker_prob=prob,
                                         poor_worker_quality=quality)
                items, clusters, labels, metrics = run_process(hits_directory)
                accuracy[-1] += metrics['Ac']
                clique_size[-1] += metrics['clique_size']
            accuracy[-1] = accuracy[-1]*1.0/ num_trials
            clique_size[-1] = clique_size[-1]*1.0/num_trials
        accuracies.append(accuracy)
        clique_sizes.append(clique_size)
        xs.append(poor_worker_qualities)
        legends.append('P_w = {}'.format(prob))

    fig_name = 'worker_errors_simulation_Ac'
    plot_multiple_line(
        xs, accuracies, None, 'Error rate of inaccurate workers', 'Ac', legends, fig_name)

    fig_name = 'worker_errors_simulation_clique_size'
    plot_multiple_line(xs, clique_sizes, 
        None, 'Error rate of inaccurate workers', 'Size of max clique', legends, fig_name)




# num_workers_simulation()
worker_errors_simulation()
