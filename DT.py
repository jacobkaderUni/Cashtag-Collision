import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, matthews_corrcoef)

class DecisionTreeModel:

    def __init__(self, best_params=None):
        if best_params:
            self.model = DecisionTreeClassifier(**best_params, random_state=42)
        else:
            self.model = DecisionTreeClassifier()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        # Calculate performance metrics
        accuracy = accuracy_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        # F1 scores
        f1_macro = f1_score(y_pred, y_test, average="macro")
        f1_micro = f1_score(y_pred, y_test, average="micro")
        f1_weighted = f1_score(y_pred, y_test, average="weighted")

        # Precision
        precision_macro = precision_score(y_test, y_pred, average='macro')
        precision_micro = precision_score(y_test, y_pred, average='micro')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')

        # Recall
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_micro = recall_score(y_test, y_pred, average='micro')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Write results to an external file
        with open('results/DTresults.txt', 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")

            f.write("Accuracy: {}\n".format(accuracy))
            f.write("F1 Score: {}\n".format(f1))
            f.write("Precision: {}\n".format(precision))
            f.write("Recall: {}\n".format(recall))
            f.write("Mcc: {}\n".format(mcc))

            f.write("\nF1 Score macro: {}\n".format(f1_macro))
            f.write("F1 Score micro: {}\n".format(f1_micro))
            f.write("F1 Score weighted: {}\n".format(f1_weighted))

            f.write("\nPrecision macro: {}\n".format(precision_macro))
            f.write("Precision micro: {}\n".format(precision_micro))
            f.write("Precision weighted: {}\n".format(precision_weighted))

            f.write("\nRecall macro: {}\n".format(recall_macro))
            f.write("Recall micro: {}\n".format(recall_micro))
            f.write("Recall weighted: {}\n".format(recall_weighted))

        # Plot confusion matrix
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        display.plot()

        # Save the plot as an image file
        plt.savefig('results/DT_confusion_matrix.png')
        # Close the plot to release memory
        plt.close()

    def plot_feature_importances(self, X_train, n_features_to_plot=20):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Load feature labels from the file
        with open("results/feature_labels.txt", "r", encoding="utf-8") as f:
            feature_labels = [line.strip() for line in f.readlines()]

        # Get feature names corresponding to the indices
        feature_names = [feature_labels[index] for index in indices[:n_features_to_plot]]

        plt.figure(figsize=(20, 6))
        plt.title("Feature importances")
        plt.bar(range(n_features_to_plot), importances[indices[:n_features_to_plot]], align="center")
        plt.xticks(range(n_features_to_plot), feature_names,
                   rotation=30)  # Replace indices with feature_names and add rotation
        plt.xlim([-1, n_features_to_plot])

        # Save the plot as an image file
        plt.savefig("results/feature_importance.png")
        plt.close()