from proglearn import UncertaintyForest
import numpy as np

X = np.random.normal(0, 1, (100, 4))
y = (X[:, 0] > 0).astype(int)

print(y)
uf = UncertaintyForest()
uf.fit(X, y)
print(uf.predict(X))

print('done')