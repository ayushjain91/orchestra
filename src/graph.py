class Graph:
    def __init__(self, clusterings=None):
        self.vertices = []
        self.edges = {}
        if clusterings is not None:
            self.construct_graph(clusterings)

    def construct_graph(self, clusterings):
        N = len(clusterings)
        for clustering in clusterings:
            self.add_vertex(clustering)
        for i in range(N):
            for j in range(i + 1, N):
                consistent = clusterings[i].consistent_binary(clusterings[j])
                if consistent:
                    self.add_edge(clusterings[i], clusterings[j])

    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices.append(vertex)
            self.edges[vertex] = []

    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.edges:
            self.edges[vertex1].append(vertex2)
        else:
            self.edges[vertex1] = [vertex2]

        if vertex2 in self.edges:
            self.edges[vertex2].append(vertex1)
        else:
            self.edges[vertex2] = [vertex1]

    def mask_compare(self, a, b):
        cnt1 = bin(a).count("1")
        cnt2 = bin(b).count("1")
        if cnt1 > cnt2:
            return -1
        if cnt1 == cnt2:
            return 0
        return 1

    def find_maximal_cliques(self):
        N = len(self.vertices)
        used = [False for i in range(1 << N)]
        masks = [i for i in range(1 << N)]
        masks.sort(cmp=self.mask_compare)
        res = []
        res2 = []
        # iterate all subsets
        for mask in masks:
            if used[mask]:
                continue
            indices = []
            for i in range(N):
                if (mask & (1 << i)) > 0:
                    indices.append(i)
            clique = True
            for i in indices:
                vertex1 = self.vertices[i]
                neighbors = self.edges[vertex1]
                for j in indices:
                    if i == j:
                        continue
                    vertex2 = self.vertices[j]
                    if vertex2 not in neighbors:
                        clique = False
                        break
                if not clique:
                    break

            # clique satisfying the clique constraint found
            if clique:
                # ignore all subsets of the found clique in the future
                for mask2 in range(1 << len(indices)):
                    submask = 0
                    for i in range(len(indices)):
                        if (mask2 & (1 << i)) > 0:
                            submask |= (1 << indices[i])
                    used[submask] = True

                subset = []
                for index in indices:
                    subset.append(self.vertices[index])
                res.append(subset)
                res2.append(indices)
        return (res, res2)
