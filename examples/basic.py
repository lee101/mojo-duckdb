import numpy as np

import mojo_duckdb as mdb


values = np.ma.array([1.0, 2.0, 99.0, 4.0], mask=[0, 0, 1, 0])

print(mdb.sum(values))
print(mdb.var_pop(values))
print(mdb.filter(values, [True, False, True, True]))

groups = mdb.group_by([10, 20, 10, 20], [1.0, 2.0, 3.0, None])
print(sorted((int(k), float(v)) for k, v in zip(groups.keys, groups.sum)))

pairs = mdb.hash_join([7, 8, 8], [8, 8, 9])
print(sorted((int(i), int(j)) for i, j in zip(pairs.left, pairs.right)))
