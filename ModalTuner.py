import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class ModelTuner:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def tune_decision_tree(self):
        param_gridDT = {
            'criterion': ['gini', 'entropy'],  # Quality of a split criterion
            'max_depth': np.arange(1, 21),  # Maximum depth of the tree
            'min_samples_split': np.arange(2, 21),  # Minimum number of samples required to split an internal node
            'min_samples_leaf': np.arange(1, 21)  # Minimum number of samples required to be at a leaf node
        }
        dt = DecisionTreeClassifier(random_state=42)
        grid_searchDT = GridSearchCV(dt, param_gridDT, cv=5, scoring='accuracy', n_jobs=-1)
        grid_searchDT.fit(self.X_train, self.y_train)
        return grid_searchDT.best_params_

    def tune_knn(self):
        param_gridKNN = {
            'n_neighbors': np.arange(1, 31),  # Number of neighbors to consider (k)
            'weights': ['uniform', 'distance'],  # Weight function used in prediction
            'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
        }
        knn = KNeighborsClassifier()
        grid_searchKNN = GridSearchCV(knn, param_gridKNN, cv=5, scoring='accuracy', n_jobs=-1)
        grid_searchKNN.fit(self.X_train, self.y_train)
        return grid_searchKNN.best_params_

    def tune_random_forest(self):
        param_gridRF = {
            'n_estimators': [10, 50, 100, 200],
            'max_features': ['sqrt'],
            'max_depth': [10, 20, 30, 40, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        grid_searchRF = GridSearchCV(RandomForestClassifier(random_state=42), param_gridRF, cv=5, scoring='accuracy',
                                     n_jobs=-1, verbose=0)
        grid_searchRF.fit(self.X_train, self.y_train)
        return grid_searchRF.best_params_

    def tune_svm(self):
        param_gridSVM = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],
            'gamma': ['scale', 'auto']
        }
        svm1 = SVC()
        grid_searchSVM = GridSearchCV(svm1, param_gridSVM, cv=5, n_jobs=-1)
        grid_searchSVM.fit(self.X_train, self.y_train)
        return grid_searchSVM.best_params_
