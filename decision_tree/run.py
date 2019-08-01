import numpy as np
from sklearn.metrics import regression
from sklearn.datasets import make_regression
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor

from decision_tree.utils import timeit, inference
from decision_tree.tree_builder import build_tree, random_split, random_split_v2, greedy_split


@timeit
def main():
    np.random.seed(20)
    x, y = make_regression(n_samples=10000, n_informative=30)

    t = build_tree(x, y, max_depth=10,
                   max_feature=100, min_improvement=0.000001, split_method=random_split)
    y_hat = inference(x, t)
    print("my result random", regression.mean_squared_error(y_hat, y))

    t = build_tree(x, y, max_depth=10,
                   max_feature=100, min_improvement=0.000001, split_method=random_split_v2)
    y_hat = inference(x, t)
    print("my result random v2", regression.mean_squared_error(y_hat, y))

    t = build_tree(x, y, max_depth=10,
                   max_feature=100, min_improvement=0.000001, split_method=greedy_split)
    y_hat = inference(x, t)
    print("my result", regression.mean_squared_error(y_hat, y))

    @timeit
    def fit(x, y):
        clf = DecisionTreeRegressor(
            #clf = DecisionTreeRegressor(
            max_depth=10, max_features=100, min_impurity_decrease=0.000001)
        clf.fit(x, y)
        return clf

    clf = fit(x, y)
    #print ("sklearn predicted", clf.predict(x))
    print("greedy sklearn result",
          regression.mean_squared_error(inference(x, clf), y))

    @timeit
    def fit(x, y):
        clf = ExtraTreeRegressor(
            #clf = DecisionTreeRegressor(
            max_depth=10, max_features=100, min_impurity_decrease=0.000001)
        clf.fit(x, y)
        return clf

    clf = fit(x, y)
    #print ("sklearn predicted", clf.predict(x))
    print("random sklearn result",
          regression.mean_squared_error(inference(x, clf), y))
    return x, y, clf, t


if __name__ == "__main__":
    main()
