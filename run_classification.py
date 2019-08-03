import numpy as np
from sklearn.metrics import classification
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from decision_tree.utils import timeit, inference
from decision_tree.tree_builder import build_classification_tree, random_split, random_split_v2, greedy_split
from decision_tree.strategy import greedy_classification, random_classify


@timeit
def main():
    np.random.seed(20)
    x, y = make_classification(n_samples=1000, n_classes=4, n_informative=10)

    t = build_classification_tree(x, y, max_depth=10,
                                  max_feature=20, min_improvement=0.000001, max_classes=4, split_method=greedy_classification)
    y_hat = inference(x, t)
    print("my result greedy", classification.accuracy_score(y_hat, y))
    t = build_classification_tree(x, y, max_depth=10,
                                  max_feature=20, min_improvement=0.000001, max_classes=4, split_method=random_classify)
    y_hat = inference(x, t)
    print("my result random", classification.accuracy_score(y_hat, y))

    @timeit
    def fit(x, y):
        clf = DecisionTreeClassifier(
            #clf = DecisionTreeRegressor(
            max_depth=10, max_features=20, min_impurity_decrease=0.000001)
        clf.fit(x, y)
        return clf

    clf = fit(x, y)
    #print ("sklearn predicted", clf.predict(x))
    print("greedy sklearn result",
          classification.accuracy_score(clf.predict(x), y))

    @timeit
    def fit(x, y):
        clf = ExtraTreeClassifier(
            #clf = DecisionTreeRegressor(
            max_depth=10, max_features=20, min_impurity_decrease=0.000001)
        clf.fit(x, y)
        return clf

    clf = fit(x, y)
    print("random sklearn result",
          classification.accuracy_score(clf.predict(x), y))
    return x, y, clf, t


if __name__ == "__main__":
    main()
