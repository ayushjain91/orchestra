"""Provides utilities to read HITs data
"""
import ast
import glob
import re
from classes import Cluster, Clustering, Item
import random


def random_combination(iterable, r):
    """Returns a random combination of r objects from iterable

    Args:
        iterable (iterable): The collection from which to return objects
        r (int): Size of the returned combination

    Returns:
        TYPE: Description
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(xrange(n), r))
    return [pool[i] for i in indices]


def process_clustering_string(line):
    """Processes a line from the hit file to return a list representation
       of the clustering

    Args:
        line (str): A string representing clustering
                    e.g. '1, [1,2,3][4,5,6][7,8], 2345'

    Returns:
        list(list(int)): Reads and evaluates this as a string
    """
    clustering = '['

    m = re.findall(r"""\[[0-9,]+\]""", line.rstrip())
    for patt in m:
        clustering += patt
        clustering += ','

    clustering = clustering[:-1] + ']'
    return ast.literal_eval(clustering)


def create_clustering_obj(hit_id, items_map, clustering_list):
    """Creates a clustering object for a clustering

    Args:
        hit_id (int): Hit #
        items_map (dict of tuple to Item): Maps items from occurances in hits
                to respective Item class object
        clustering_list (list(int)): A clustering of items (represented by
                indices of their occuance in hit)

    Returns:
        Clustering: Object of class clustering
    """
    clustering_obj = Clustering()
    for cluster in clustering_list:
        cluster_obj = Cluster()
        for item_id in cluster:
            cluster_obj.add_item(items_map[(hit_id, item_id)])
        clustering_obj.add_cluster(cluster_obj)
    return clustering_obj


def read_worker_clustering(results_file, hit_num, items_map=None,
                           num_hits_reqd=None, is_categorization=False,
                           items_to_remove=0):
    clustering_set = []
    item_set = None
    with open(results_file, 'r+') as f:
        for line in f:
            hit_num_str = str(hit_num) + ','
            if line[0:len(hit_num_str)] != hit_num_str:
                continue
            line = line[len(hit_num_str):]
            clustering_list = process_clustering_string(line)

            # Remove Items
            if item_set is None:
                item_set = [
                    item for cluster in clustering_list for item in cluster]

                if items_to_remove > 0:
                    item_sample = random_combination(
                        item_set, len(item_set) - items_to_remove)

            if items_to_remove > 0:
                for i in range(len(clustering_list)):
                    clustering_list[i] = list(
                        set(clustering_list[i]) & set(item_sample))
            if not is_categorization:
                clustering_obj = create_clustering_obj(
                    hit_num, items_map, clustering_list)
            else:
                clustering_obj = clustering_list
            clustering_set.append(clustering_obj)

    if num_hits_reqd is not None and num_hits_reqd < len(clustering_set):
        return random_combination(clustering_set, num_hits_reqd)
    return clustering_set


def read_pivots(categorization_folder, clustering_items):
    pivots_urls = []
    with open(categorization_folder + "pivots.txt", 'r+') as fin:
        for line in fin:
            pivots_urls.append(extract_from_url(line.strip()))

    pivots_cluster_assignments = []
    with open(categorization_folder + "pivots_ordering.txt", 'r+') as fin:
        for line in fin:
            index = map(int, line.strip()[1:-1].split(','))[0] - 1
            for item in clustering_items:
                if item.url == pivots_urls[index]:
                    pivots_cluster_assignments.append(item.cluster)
                    break

    return pivots_cluster_assignments


def read_worker_answers(results_file, hit_num, items_map,
                        num_hits_reqd=None, is_categorization=False,
                        items_to_remove=0):
    return read_worker_clustering(results_file, hit_num,
                                  items_map, num_hits_reqd, is_categorization,
                                  items_to_remove=items_to_remove)


def read_hit(hit_file):
    res = []
    with open(hit_file, 'rb') as f:
        for line in f:
            res.append(line.rstrip())
    return res


def extract_from_url(url):
    return '/'.join(url.strip().split('/')[-2:])


def read_hits(directory="scenesHits/run2/", hit_file_regex="hit*.txt"):
    items_map = {}
    all_items = []
    hit_files = glob.glob(directory + hit_file_regex)
    for hit_file in hit_files:
        hit_name = hit_file.split('/')[-1]
        hit_num = int(hit_name[3:-4])
        hit_items = read_hit(hit_file)
        cnt = 1
        for url in hit_items:
            item_obj = Item()
            item_obj.url = extract_from_url(url)
            idx = -1
            if item_obj in all_items:
                idx = all_items.index(item_obj)
            if idx == -1:
                item_obj.ground_truth_label = url.split('/')[-2]
                item_obj.hitID_array.append(hit_num)
                all_items.append(item_obj)
                items_map[(hit_num, cnt)] = item_obj
            else:
                all_items[idx].hitID_array.append(hit_num)
                items_map[(hit_num, cnt)] = all_items[idx]

            cnt += 1
    return (len(hit_files), items_map, all_items)
