import numpy as np
from sklearn.metrics import classification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from functools import partial
from sklearn import preprocessing
from sklearn.datasets import fetch_kddcup99

from sklearn.model_selection import cross_val_score
from decision_tree.utils import timeit, inference
from decision_tree.tree_builder import build_classification_tree, random_split, random_split_v2, greedy_split
from decision_tree.strategy import greedy_classification, random_classify, greedy_classification_p_at_k, random_classify_p_at_k
from sklearn import preprocessing
import sklearn.datasets as datasets


class Mree:
    def __init__(self, max_depth=10, max_feature=20, min_improvement=1e-7, max_classes=23, split_method=greedy_classification):
        self.params = dict(
            max_depth=max_depth,
            max_feature=max_feature,
            min_improvement=min_improvement,
            max_classes=max_classes,
            split_method=split_method,
        )
        self.method = partial(build_classification_tree,
                              **self.params
                              )

    def get_params(self, *args, **kwargs):
        return self.params

    def fit(self, x, y):
        self.t = self.method(x, y)

    def predict(self, x):
        return inference(x, self.t)


@timeit
def main():
    np.random.seed(20)

    def scorer(est, x, y):
        y_hat = est.predict(x)
        return classification.accuracy_score(y, y_hat)

    #x, y = make_classification(n_samples=1000, n_classes=4, n_informative=10)
    x, y = fetch_kddcup99(return_X_y=True)
    x = np.array(x[:, 4:], dtype=np.float32)
    y = preprocessing.LabelEncoder().fit_transform(y)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    myclf = Mree(split_method=greedy_classification)
    score = cross_val_score(myclf, x, y, cv=5, scoring=scorer)
    print("Mine greedy classification result",
          score,
          np.mean(score))
    myclf = Mree(split_method=greedy_classification_p_at_k)
    score = cross_val_score(myclf, x, y, cv=5, scoring=scorer)
    print("Mine greedy p@k classification result",
          score,
          np.mean(score))

    myclf = Mree(split_method=random_classify_p_at_k)
    score = cross_val_score(myclf, x, y, cv=5, scoring=scorer)
    print("Mine random p@k classification result",
          score,
          np.mean(score))

    clf = DecisionTreeClassifier(
        max_depth=10, max_features=20, min_impurity_decrease=0.000001)
    score = cross_val_score(clf, x, y, cv=5, scoring=scorer)
    print("Sklearn greedy classification result",
          score,
          np.mean(score))
    clf = ExtraTreeClassifier(
        max_depth=10, max_features=20, min_impurity_decrease=0.000001)
    score = cross_val_score(clf, x, y, cv=5, scoring=scorer)
    print("sklearn random classification result",
          score,
          np.mean(score))


if __name__ == "__main__":
    main()
