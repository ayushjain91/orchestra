from graph import Graph
from hitting_set import HittingSet
from classes import *

# build hitting set for 2-consistency
def build_hitting_set_2(clustering1, clustering2):
    res = HittingSet()
    for cluster1 in clustering1:
        for cluster2 in clustering2:
            if cluster1 <= cluster2 or cluster1 >= cluster2:
                continue
            a = set(cluster1.item_array) - set(cluster2.item_array)
            b = set(cluster1.item_array) & set(cluster2.item_array)
            c = set(cluster2.item_array) - set(cluster1.item_array)

            for item_a in a:
                for item_b in b:
                    for item_c in c:
                        res.add_set(frozenset([item_a, item_b, item_c]))
    return res

# build hitting set for n-consistency
def build_hitting_set_n(clusterings):
    res = HittingSet()
    for clustering1 in clusterings:
        for clustering2 in clusterings:
            res += build_hitting_set_2(clustering1, clustering2)
    return res

if __name__ == "__main__":
    c1 = Cluster([Item("a"), Item("b")])
    c2 = Cluster([Item("c")])
    c3 = Cluster([Item("d"), Item("e")])

    c4 = Cluster([Item("a")])
    c5 = Cluster([Item("b"), Item("c")])
    c6 = Cluster([Item("d"), Item("e")])

    C1 = Clustering()
    C2 = Clustering()
    C1.add_cluster(c1)
    C1.add_cluster(c2)
    C1.add_cluster(c3)

    C2.add_cluster(c4)
    C2.add_cluster(c5)
    C2.add_cluster(c6)

    res = build_hitting_set_n([C1, C2])
    print res.hitting_sets