from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    f1_score, precision_score, recall_score, matthews_corrcoef)
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
class NaiveBayesModel:

    def __init__(self):
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # F1 Score in depth
        f1_macro = f1_score(y_pred, y_test, average="macro")
        f1_micro = f1_score(y_pred, y_test, average="micro")
        f1_weighted = f1_score(y_pred, y_test, average="weighted")

        # Precision in depth
        precision_macro = precision_score(y_test, y_pred, average='macro')
        precision_micro = precision_score(y_test, y_pred, average='micro')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')

        # Recall in depth
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_micro = recall_score(y_test, y_pred, average='micro')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')

        # Plot confusion matrix
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        display.plot()

        # Save the plot as an image file
        plt.savefig('results/NB_confusion_matrix.png')
        # Close the plot to release memory
        plt.close()

        with open('results/NBresults.txt', 'w', encoding='utf-8') as f:
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