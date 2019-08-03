import numpy as np
from sklearn.metrics import classification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from decision_tree.utils import timeit, inference
from decision_tree.tree_builder import build_classification_tree, random_split, random_split_v2, greedy_split
from decision_tree.strategy import greedy_classification, random_classify, greedy_classification_p_at_k, random_classify_p_at_k


@timeit
def main():
    np.random.seed(20)
    x, y = make_classification(n_samples=1000, n_classes=4, n_informative=10)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    t = build_classification_tree(X_train, y_train, max_depth=10,
                                  max_feature=20, min_improvement=0.000001, max_classes=4, split_method=greedy_classification)
    y_hat = inference(X_test, t)
    print("my result greedy", classification.classification_report(y_test, y_hat))

    t = build_classification_tree(X_train, y_train, max_depth=10,
                                  max_feature=20, min_improvement=0.000001, max_classes=4, split_method=greedy_classification_p_at_k)
    y_hat = inference(X_test, t)
    print("my result greedy p@1",
          classification.classification_report(y_test, y_hat))

    t = build_classification_tree(X_train, y_train, max_depth=10,
                                  max_feature=20, min_improvement=0.000001, max_classes=4, split_method=random_classify)
    y_hat = inference(X_test, t)
    print("my result random", classification.classification_report(y_test, y_hat))

    t = build_classification_tree(X_train, y_train, max_depth=10,
                                  max_feature=20, min_improvement=0.000001, max_classes=4, split_method=random_classify_p_at_k)
    y_hat = inference(X_test, t)
    print("my result random p@1",
          classification.classification_report(y_test, y_hat))

    @timeit
    def fit(x, y):
        clf = DecisionTreeClassifier(
            #clf = DecisionTreeRegressor(
            max_depth=10, max_features=20, min_impurity_decrease=0.000001)
        clf.fit(x, y)
        return clf

    clf = fit(X_train, y_train)
    #print ("sklearn predicted", clf.predict(x))
    print("greedy sklearn result",
          classification.classification_report(y_test, clf.predict(X_test)))

    @timeit
    def fit(x, y):
        clf = ExtraTreeClassifier(
            #clf = DecisionTreeRegressor(
            max_depth=10, max_features=20, min_impurity_decrease=0.000001)
        clf.fit(x, y)
        return clf

    clf = fit(X_train, y_train)
    #print ("sklearn predicted", clf.predict(x))
    print("random sklearn result",
          classification.classification_report(y_test, clf.predict(X_test)))
    return x, y, clf, t


if __name__ == "__main__":
    main()
