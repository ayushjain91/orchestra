from __future__ import division
import random
import scipy.io as sio


class Clustering(object):
    def __init__(self, file_name=None):
        self.clustering_array = []
        if file_name is not None:
            self.load_mat(file_name)

    def add_cluster(self, cluster):
        self.clustering_array.append(cluster)

    def __getitem__(self, index):
        return self.clustering_array[index]

    def consistent_binary(self, other):
        for cluster1 in self.clustering_array:
            for cluster2 in other.clustering_array:
                intersect = set(cluster1.item_array) & set(cluster2.item_array)
                if len(intersect) > 0:
                    if not(cluster1 <= cluster2 or cluster1 >= cluster2):
                        return False
        return True

    def load_mat(self, file_name):
        self.clustering_array = []
        data = sio.loadmat(file_name)
        cluster_numbers = data['clusters']
        item_urls = data['items']
        num_clusters = 0
        for n in cluster_numbers:
            num_clusters = max(num_clusters, n[0])
        for i in range(num_clusters):
            self.clustering_array.append(Cluster())

        for i in range(len(cluster_numbers)):
            cluster_number = cluster_numbers[i][0] - 1
            item_url = item_urls[i].rstrip()
            item = Item(item_url)
            item.ground_truth_label = item_url.split("/")[-1]
            self.clustering_array[cluster_number].add_item(item)


class Cluster(object):
    def __init__(self):
        self.item_array = []

    def add_item(self, item):
        self.item_array.append(item)

    def __getitem__(self, index):
        return self.item_array[index]

    def __le__(self, other):
        if isinstance(other, Cluster):
            return set(self.item_array) <= set(other.item_array)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Cluster):
            return set(self.item_array) >= set(other.item_array)
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Cluster):
            return len(set(self.item_array) - set(self.item_array)) == 0
        return NotImplemented

    def __hash__(self):
        sorted_array = sorted(self.item_array)
        str = ''
        for item in sorted_array:
            str += item.url
        return hash(str)


class Item(object):
    def __init__(self, url=""):
        self.url = url
        self.ground_truth_label = ""
        self.hitID_array = []
        self.cluster = None

    def add_to_cluster(self, cluster=None):
        self.cluster = cluster

    def __eq__(self, other):
        if isinstance(other, Item):
            return self.url == other.url
        return NotImplemented

    def __hash__(self):
        return hash(self.url)

    def get_item_id(self):
        tokens = self.url.split("/")
        return tokens[-1]


class HITResult(object):
    def __init__(self):
        self.hitID = ""
        self.workerID = ""
        self.clustering = None
        self.time_taken = 0


class HITDict(object):
    def __init__(self):
        self.dict = {}


class Hierarchy(object):
    def __init__(self, clique=None):
        self.root = None
        self.nodes = []
        if clique is not None:
            self.construct_hierarchy(clique)
        self.node_worker = {}

    def construct_hierarchy(self, clique=None):
        clusters = []
        cluster_worker = {}
        for clustering in clique:
            for cluster in clustering:
                if cluster not in cluster_worker:
                    cluster_worker[cluster] = 1
                else:
                    cluster_worker[cluster] += 1
                clusters.append(cluster)

        clusters = list(set(clusters))
        items = []
        for cluster in clusters:
            items.extend(cluster.item_array)
        items = list(set(items))
        self.root = HierarchyNode()
        self.root.item_array = list(items)

        for cluster in clusters:
            node = HierarchyNode()
            node.item_array = list(cluster.item_array)
            node.numSplit += cluster_worker[cluster]
            self.nodes.append(node)

        for node1 in self.nodes:
            parent = None
            for node2 in self.nodes:
                if node1 == node2:
                    continue
                if set(node1.item_array) <= set(node2.item_array):
                    if parent is None:
                        parent = node2
                    else:
                        if len(parent.item_array) > len(node2.item_array):
                            parent = node2
            if parent is None:
                node1.parent = self.root
                self.root.children.append(node1)
            else:
                node1.parent = parent
                parent.children.append(node1)

    def print_all_leaves(self):
        for node in self.nodes:
            if len(node.children) != 0:
                continue
            l = []
            for item in node.item_array:
                l.append(item.url)
            print str(list(set(l)))

    def get_items_not_in_leaves(self):
        items_in_leaves = []
        for node in self.nodes:
            if (len(node.children)) != 0:
                continue
            items_in_leaves.extend(node.item_array)
        res = list(set(self.root.item_array) - set(items_in_leaves))
        return res

    def print_all_nodes(self):
        for node in self.nodes:
            # , [a.url for a in node.item_array]
            print str(node.get_ground_truth_labels()) + ' ' + str(len(node.item_array)), node.numSplit

    def print_max_likelihood_frontier(self):
        frontier = self.maxLikelihoodFrontier()
        self.print_frontier(frontier)

    def print_frontier(self, frontier):
        import random
        for node in frontier:
            urls = [a.url for a in node.item_array]
            random.shuffle(urls)
            urls = urls[0:10]
            print str(node.get_ground_truth_labels()), urls
        print

    def get_leaves(self):
        res = []
        for node in self.nodes:
            if len(node.children) != 0:
                continue
            res.append(node)
        return res

    def merge_hierarchy(self, other):
        common_items = self.get_common_items(other)
        leaves = other.get_leaves()
        for leaf in leaves:
            anc = other.get_lowest_anc_common(leaf, common_items)
            if anc is None:
                anc = other.root
            common = list(set(anc.item_array) & set(common_items))
            self.add_leaf(leaf, common, anc != leaf)

    def get_all_items(self):
        res = []
        for node in self.nodes:
            res.extend(node.item_array)
        return list(set(res))

    def get_common_items(self, other):
        return list(set(self.get_all_items()) & set(other.get_all_items()))

    def add_leaf(self, leaf, common, attach):
        self._add_leaf(self.root, leaf, common, attach)

    def _add_leaf(self, curr_node, leaf, common, attach):
        # add the leaf node items to current node
        curr_node.item_array.extend(leaf.item_array)

        # get rid of duplicates
        curr_node.item_array = list(set(curr_node.item_array))
        traversed = False
        for child in curr_node.children:
            if set(common) <= set(child.item_array):
                self._add_leaf(child, leaf, common, attach)
                traversed = True
        if not traversed:
            curr_node.numSplit += leaf.numSplit

        if not traversed and attach:
            curr_node.children.append(leaf)
            leaf.parent = curr_node
            self.nodes.append(leaf)

    def get_lowest_anc_common(self, leaf, common_items):
        res = None
        for node in self.nodes:
            if set(leaf.item_array) <= set(node.item_array) and len(set(node.item_array) & set(common_items)) > 0:
                if res is None or len(res.item_array) > len(node.item_array):
                    res = node
        return res

    def get_num_items(self):
        return len(self.root.item_array)

    def sample_from_leaves(self, sample_num, kernels):
        urls = []
        res = []
        a = 0
        b = 0
        for node in self.nodes:
            if len(node.children) != 0:
                continue
            l = []
            for item in node.item_array:
                l.append(item.url)
            inter = list(set(l) & set(kernels))
            if len(inter) > 0:
                urls.append(inter)
                a += 1
            else:
                urls.append(l)
            b += 1
        print str(a) + ' / ' + str(b)
        cnt = 0
        while cnt < sample_num:
            for i in range(len(urls)):
                if len(urls[i]) == 0:
                    continue
                random.shuffle(urls[i])
                res.append(urls[i][0])
                urls[i] = urls[i][1:]
                cnt += 1
                if cnt == sample_num:
                    break
        return res

    def _copy(self, h, node, node_copy):
        h.nodes.append(node_copy)
        node_copy.item_array = list(node.item_array)
        for child in node.children:
            child_copy = HierarchyNode()
            child_copy.parent = node_copy
            child_copy.item_array = list(child.item_array)
            child_copy.numSplit = child.numSplit
            node_copy.children.append(child_copy)
            self._copy(h, child, child_copy)

    def copy(self):
        h = Hierarchy()
        h.root = HierarchyNode()
        self._copy(h, self.root, h.root)
        return h

    def printItemsInRoot(self):
        A = set([a.url for a in self.root.item_array])
        b = []
        for child in self.root.children:
            b.extend([a.url for a in child.item_array])
        b = set(b)
        return A - b

    def maxLikelihoodFrontier(self):
        self.root.calcNumSplitsInSubtree()
        return self.root.getMaxLikelihoodFrontierInSubtree()[0]

    def getAllFrontiers(self, size):
        return self.root.getAllFrontiers(size)

    def findFrontierLikelihood(self, frontier):
        self.maxLikelihoodFrontier()
        return self.root.findFrontierLikelihood(frontier)


class HierarchyNode(object):
    def __init__(self):
        self.item_array = []
        self.children = []
        self.parent = None
        self.numSplit = 0  # Number of workers who split this node
        self.numSplitInSubtree = 0
        self.probNotSplit = 0

    def get_ground_truth_labels(self):
        l = []
        for item in self.item_array:
            l.append(item.ground_truth_label)
        return list([(a, l.count(a)) for a in set(l)])

    def get_item_ids(self):
        l = []
        for item in self.item_array:
            l.append(item.get_item_id())
        return l

    def calcNumSplitsInSubtree(self):
        n = self.numSplit
        for child in self.children:
            n += child.calcNumSplitsInSubtree()
        self.numSplitInSubtree = n
        return n

    def getMaxLikelihoodFrontierInSubtree(self):
        frontier = []
        split = False

        if not self.children:
            frontier = [self]
            return (frontier, 1)

        prob = 1 - (self.numSplit * 1.0 / self.numSplitInSubtree)
        self.probNotSplit = prob
        front = []
        for child in self.children:
            (best_frontier,
             frontier_prob) = child.getMaxLikelihoodFrontierInSubtree()
            prob = prob * frontier_prob
            front.extend(best_frontier)

        prob_notSplitting = (self.numSplit * 1.0 / self.numSplitInSubtree)
        if prob <= prob_notSplitting:
            frontier = [self]
            return (frontier, prob_notSplitting)

        return (front, prob)

    def getAllFrontiers(self, size):

        num_children = len(self.children)
        if size == 1:
            return [[self]]

        elif num_children == 0:
            return []

        all_ways_splitting = splitItems(size, num_children)[1]
        all_frontiers = []
        # print size, num_children
        for split in all_ways_splitting:
            idx = 0
            frontiers = [[]]
            flag = True
            for child in self.children:

                child_frontiers = child.getAllFrontiers(split[idx])
                idx += 1
                if child_frontiers == []:
                    flag = False
                    break
                _frontiers = []
                for frontier in frontiers:
                    for child_frontier in child_frontiers:
                        frontier.extend(child_frontier)
                        _frontiers.append(frontier)
                frontiers = _frontiers

            if flag:
                all_frontiers.extend(frontiers)

        # print all([len(a)==size for a in all_frontiers])
        return all_frontiers

    def findFrontierLikelihood(self, frontier):
        if self in frontier:
            return 1 - self.probNotSplit

        prob = self.probNotSplit
        for child in self.children:
            prob = prob * child.findFrontierLikelihood(frontier)

        return prob


def splitItems(numItems, numBins):
    splits = []
    if numBins == 0 and numItems != 0:
        return False, None
    elif numBins == 0:
        return True, None
    for i in range(1, numItems - numBins + 2):
        s_tup = splitItems(numItems - i, numBins - 1)
        if s_tup[0]:
            if s_tup[1] != None:
                s = s_tup[1]
                for _s in s:
                    _scopy = _s
                    _scopy.extend([i])
                    splits.append(_scopy)
            else:
                splits.append([i])

    return len(splits) != 0, splits


def custom_metric(clustering1, clustering2):
    score = 0
    items1 = []
    items2 = []
    for cluster in clustering1:
        items1.extend(cluster.item_array)
    for cluster in clustering2:
        items2.extend(cluster.item_array)
    items = list(set(items1) & set(items2))
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            together1 = False
            for cluster in clustering1:
                if items[i] in cluster.item_array and items[j] in cluster.item_array:
                    together1 = True
                    break
            together2 = False
            for cluster in clustering2:
                if items[i] in cluster.item_array and items[j] in cluster.item_array:
                    together2 = True
                    break

            if together1 == together2:
                score += 1
            else:
                score -= 1
    return score / (n * (n - 1) / 2)


if __name__ == '__main__':

    raise SystemExit

    clustering1 = Clustering('./../../tmp/results/cc_ground_truth.mat')
    clustering2 = Clustering('./../../tmp/results/orchestra.mat')
    clustering3 = Clustering('./../../tmp/results/crowdclustering.mat')

    print custom_metric(clustering1, clustering2)
    print custom_metric(clustering1, clustering3)
