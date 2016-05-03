"""
Runs Orchestra's clustering and categorization phases
"""
from classes import *
from graph import *
import argparse
import numpy
from evaluation_utils import *
from read_write_utils import *
from collections import defaultdict


def aggregate_categorization(responses):
    """Aggregate categorization responses from multiple workers for each item

    Args:
        responses (list(list(int))): List of clusterings, where each clustering
            organizes items labelled from integers 1 to num_items

    Returns:
        list: Cluster indices for every item from 1 to num_items
    """
    num_items = max([max(r) for r in responses[0]])
    num_clusters = max([len(r) for r in responses])  # len(pivot_clusters)
    vote_matrix = numpy.zeros((num_clusters, num_items))
    for response in responses:
        for cluster_idx in range(len(response)):
            for item in response[cluster_idx]:
                vote_matrix[cluster_idx, item - 1] += 1

    cluster_numbers = []
    for item_idx in range(num_items):
        cluster_idx = numpy.argmax(vote_matrix[:, item_idx])
        cluster_numbers.append(cluster_idx)
    return cluster_numbers


def evaluate(all_items, conf_matrix_path=None):
    """Evaluates the quality of clustering with respect to ground truth labels

    Args:
        all_items (List(Item)): The list of items to be organized
        conf_matrix_path (str, optional): If specified, stores a
                visualization of clusters in the form of a confusion
                matrix in .eps and .png format at this path.
                The path should include the name of the file
                (.eps and .png are auto appended)

    Returns:
        dict(string) to float: mapping of all metrics to their values
    """
    items = [item.url for item in all_items]
    clusters = [item.cluster for item in all_items]
    ground_truth_labels = [item.ground_truth_label for item in all_items]

    precision = calc_precision(clusters, ground_truth_labels)
    recall = calc_recall(clusters, ground_truth_labels)
    accuracy = calc_accuracy(clusters, ground_truth_labels)
    if conf_matrix_path:
        conf_matrix_plot(clusters, ground_truth_labels, conf_matrix_path)

    unclustered_items = len(
        [item for item in all_items if item.cluster is None])

    metrics = {'Pr': precision, 'Re': recall,
               'Ac': accuracy, 'Ui': unclustered_items}
    return items, clusters, ground_truth_labels, metrics


def run_individual_HITs(hits_directory, sample_responses=None,
                        items_to_remove=0):
    """Runs Orchestra on each HIT indidually in the given directory

    Args:
        hits_directory (str): The directory containing the HITs data
        sample_responses (int, optional): The number of worker responses
                to be sampled randomly
        items_to_remove (int, optional): The number of items to be removed
                randomly from each HIT

    Returns:
        dict of string to list: A dictionary mapping all metrics to their
                values for each HIT
    """
    num_clustering_hits, items_map, items = read_hits(hits_directory)

    results_file = hits_directory + 'results/results.txt'

    metrics = defaultdict(list)

    for hit_num in range(1, num_clustering_hits + 1):
        clusterings = read_worker_answers(results_file, hit_num, items_map,
                                          num_hits_reqd=sample_responses,
                                          items_to_remove=items_to_remove)
        graph = Graph(clusterings)
        cliques, res2 = graph.find_maximal_cliques()
        hierarchy = Hierarchy(cliques[0])
        frontier = hierarchy.maxLikelihoodFrontier()

        metrics['max_likelihood_frontier_size'].append(len(frontier))
        metrics['num_maximal_cliques'].append(len(cliques))
        metrics['size_max_clique'].append(len(cliques[0]))

        hit_items = []
        for node in frontier:
            for item in node.item_array:
                item.cluster = node
                hit_items.append(item)

        _, _, _, hit_metrics = evaluate(hit_items)
        for m in hit_metrics.keys():
            metrics[m].append(hit_metrics[m])
    return metrics


def run_process(hits_directory, includeCategorization,
                sample_responses=None, conf_matrix_fname=None):
    """Runs Orchestra on HITs in the given directory and returns results

    Args:
        hits_directory (str): The directory containing the HITs data
        includeCategorization (boolean): Specifies whether to include
                categorization data in running Orchestra
        sample_responses (int, optional): The number of worker responses
                to be sampled randomly
        conf_matrix_fname (str, optional): Filename for the cluster
                visualizations (.eps and .png are auto appended)

    Returns:
        (urls, clusters, ground_truth_labels, metrics)
        urls: list of the urls/identifiers of all items
        clusters: list of cluster identifiers for all items, in the same order
            as in urls
        ground_truth_labels: Ground truth labels for all items, in the same
            order as in urls
        metrics: dict(string) mapping all metrics to their values
    """
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

    if includeCategorization:
        categorization_directory = hits_directory + 'categorization/'
        num_categorization_hits, c_items_map, c_items = read_hits(
            categorization_directory)
        pivot_clusters = read_pivots(categorization_directory, items)

        results_file = categorization_directory + 'results/results.txt'
        for hit_num in range(1, num_categorization_hits + 1):
            raw_answers = read_worker_answers(
                results_file, hit_num, None, is_categorization=True)
            cluster_assignments = aggregate_categorization(
                raw_answers)
            for item_idx in range(1, len(cluster_assignments) + 1):
                if cluster_assignments[item_idx - 1] == len(pivot_clusters):
                    c_items_map[(hit_num, item_idx)].cluster = None
                else:
                    c_items_map[(hit_num, item_idx)].cluster = pivot_clusters[
                        cluster_assignments[item_idx - 1]]
        all_items.extend(c_items)

    conf_matrix_path = None
    if conf_matrix_fname:
        plot_dir = hits_directory
        if includeCategorization:
            plot_dir = categorization_directory
        conf_matrix_path = plot_dir + conf_matrix_fname

    urls, clusters, ground_truth_labels, metrics = evaluate(
        all_items, conf_matrix_path)
    for m in metrics:
        print m, '=', metrics[m]

    return urls, clusters, ground_truth_labels, metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str,
                        help="Path to directory containing " +
                        "hit data: relative to ../data",
                        default="scenes")
    parser.add_argument('--categorization', help='Include this if' +
                        ' you want to include the categorization phase',
                        action='store_true')
    arguments = parser.parse_args()

    hits_directory = './../data/' + arguments.path + '/'

    run_process(hits_directory, arguments.categorization,
                conf_matrix_fname='cluster_visualization')
