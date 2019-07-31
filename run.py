import numpy as np
from sklearn.metrics import regression
from sklearn.datasets import make_regression
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor

from utils import timeit, inference
from tree_builder import build_tree


@timeit
def main():
    np.random.seed(20)
    x, y = make_regression(n_samples=10000, n_informative=40)
    t = build_tree(x, y, max_depth=10,
                   max_feature=100, min_improvement=0.000001)
    y_hat = inference(x, t)
    #print(t)
    #print("truth", y)
    #print("we predicted", y_hat)
    print("my result", regression.mean_squared_error(y_hat, y))

    @timeit
    def fit(x, y):
        #clf = ExtraTreeRegressor(
        clf = DecisionTreeRegressor(
            max_depth=10, max_features=100, min_impurity_decrease=0.000001)
        clf.fit(x, y)
        return clf

    clf = fit(x, y)
    #print ("sklearn predicted", clf.predict(x))
    print("sklearn result", regression.mean_squared_error(clf.predict(x), y))
    print("sklearn result", regression.mean_squared_error(inference(x, clf), y))
    return x, y, clf, t


if __name__ == "__main__":
    main()
