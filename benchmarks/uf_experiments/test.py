from proglearn import UncertaintyForest

X = [[1, 2, 3, 4], [1, 2, 3, 4]]*4
y = [0, 1] * 4

uf = UncertaintyForest()
uf.fit(X, y)

print('done')