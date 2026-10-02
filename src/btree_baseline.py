"""Baselines the learned index must beat: a real B-Tree and plain binary search."""
from bisect import bisect_left

from BTrees.OOBTree import OOBTree  # type: ignore  


class BTreeIndex:
    """Classic B-Tree (from the BTrees package). Maps key -> position."""

    name = "BTree"

    def __init__(self, keys):
        self.tree = OOBTree()
        self.tree.update(dict(zip(keys.tolist(), range(len(keys)))))

    def lookup(self, key):
        return self.tree.get(key, -1)

    def size_bytes(self):
        # Pickling a big B-Tree hits Python's recursion limit, so we ESTIMATE the
        # size of a compact on-disk B-Tree instead (standard textbook model):
        #   leaf level : every key (8 B) + position (8 B), pages ~69% full
        #   inner nodes: one key + one pointer (16 B) per child, fan-out 32
        n = len(self.tree)
        leaf = n * 16 / 0.69
        inner = leaf / 31  # geometric sum of the upper levels: 1/32 + 1/32^2 + ...
        return int(leaf + inner)

    def params(self):
        return ""


class BinarySearchIndex:
    """No index structure at all: binary search over the sorted list."""

    name = "BinarySearch"

    def __init__(self, keys):
        self.keys = keys.tolist()

    def lookup(self, key):
        i = bisect_left(self.keys, key)
        if i < len(self.keys) and self.keys[i] == key:
            return i
        return -1

    def size_bytes(self):
        return 0  # no extra memory beyond the data itself

    def params(self):
        return ""
