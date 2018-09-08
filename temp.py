from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.metrics import recall_score
from sklearn.datasets import load_iris
import sklearn.svm as svm
iris = load_iris()
scoring = ['accuracy','precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
                        cv=5)
sorted(scores.keys())
print(scores)
scores['test_recall_macro']
