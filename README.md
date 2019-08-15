Decision Tree Impelemented in Numpy

---

# What's in

- Regression Tree with exhaustive search and randomized search 
- Both Train and Inference
- Pure numpy

# Some performance highlight

---

## Regression with MSE

### 2 times faster than sklearn's cython version

```
In [1]: from decision_tree.one import build_regression_tree

In [2]: from sklearn.metrics import regression

In [3]: from sklearn.datasets import make_regression

In [4]: from sklearn.tree import DecisionTreeRegressor

In [5]: from decision_tree.utils import inference

In [6]: x, y = make_regression(n_samples=10000)

In [7]: clf = DecisionTreeRegressor(max_depth=1)

In [8]: t = build_regression_tree(x, y, max_depth=1)

In [9]: clf.fit(x, y)
Out[9]:
DecisionTreeRegressor(criterion='mse', max_depth=1, max_features=None,
                      max_leaf_nodes=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      presort=False, random_state=None, splitter='best')

In [10]: regression.mean_squared_error(inference(x, t), y)
Out[10]: 16909.89923684728

In [11]: regression.mean_squared_error(inference(x, clf), y)
Out[11]: 16909.89923684728

In [12]: clf = DecisionTreeRegressor(max_depth=10).fit(x, y)

In [13]: t = build_regression_tree(x, y, max_depth=10)

In [14]: regression.mean_squared_error(inference(x, t), y)
Out[14]: 3385.9081612808072

In [15]: regression.mean_squared_error(inference(x, clf), y)
Out[15]: 3377.270646863389

In [16]: %timeit DecisionTreeRegressor(max_depth=10).fit(x, y)
654 ms ± 63.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [17]: %timeit build_regression_tree(x, y, max_depth=10)
390 ms ± 6.78 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
---
## Regression With MAE

### 4 times fatser than sklearn's cython version

```
In [13]: from decision_tree.strategy_l1 import build_regression_tree

In [14]: x, y = make_regression(n_samples=10000)

In [15]: t =  build_regression_tree(x, y, max_depth=1)

In [16]: clf.fit(x, y)

Out[16]:
DecisionTreeRegressor(criterion='mae', max_depth=1, max_features=None,
                      max_leaf_nodes=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      presort=False, random_state=None, splitter='best')

In [17]:

In [17]: regression.mean_absolute_error(y, clf.predict(x))
Out[17]: 151.3417502835549

In [18]: regression.mean_absolute_error(y, inference(x, t))
Out[18]: 151.3417502835549

In [19]: %timeit t =  build_regression_tree(x, y, max_depth=1)
5.59 s ± 87.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [20]: %timeit clf.fit(x, y)
22.5 s ± 116 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

### 20 timese faster with cummedian implemented in cython

```
In [25]: from decision_tree.strategy_l1 import build_regression_tree_v2

In [26]: x, y = make_regression(n_samples=10000)

In [27]: %timeit build_regression_tree_v2(x, y, max_depth=1)
1.19 s ± 15.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [28]: clf = DecisionTreeRegressor(criterion='mae', max_depth=1)

In [29]: %timeit clf.fit(x, y)
22.2 s ± 140 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [30]: t = build_regression_tree_v2(x, y, max_depth=1)

In [31]: regression.mean_absolute_error(y, inference(x, t))
Out[31]: 152.5162889311546

In [32]: clf.fit(x, y)
Out[32]:
DecisionTreeRegressor(criterion='mae', max_depth=1, max_features=None,
                      max_leaf_nodes=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      presort=False, random_state=None, splitter='best')

In [33]: regression.mean_absolute_error(y, clf.predict(x))
Out[33]: 152.5162889311546
```

## To do

- add multi-class classification
- add multi-label classification
- maybe tensorflow version
