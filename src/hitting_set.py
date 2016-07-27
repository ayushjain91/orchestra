from collections import defaultdict

class HittingSet:
    def __init__(self):
        self.hitting_sets = set()

    def add_set(self, hitting_set):
        self.hitting_sets.add(hitting_set)

    def __add__(self, other):
        res = HittingSet()
        res.hitting_sets = self.hitting_sets | other.hitting_sets
        return res

    # k-approximation algorithm for hitting set where k is the maximum size of the sets.
    def solve_approx(self):
        res = set()
        removed = defaultdict(lambda: 0)
        for s in self.hitting_sets:
            removed_count = reduce(lambda x, y: removed[x] + removed[y], s)
            if removed_count == 0:
                for item in s:
                    removed[item] = 1
                    res.add(item)

        return list(res)

if __name__ == "__main__":
    a = HittingSet()
    a.add_set(frozenset([1, 2, 3]))
    a.add_set(frozenset([3, 4, 5]))

    b = HittingSet()
    b.add_set(frozenset([6, 7, 8]))
    b.add_set(frozenset([0, 1, 2]))

    a += b
    print a.hitting_sets
    print a.solve_approx()