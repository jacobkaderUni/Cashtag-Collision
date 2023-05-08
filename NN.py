from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam

from matplotlib import pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, matthews_corrcoef)

class NeuralNetworkModel:
    def __init__(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=input_dim, activation='relu'))  # Input layer
        self.model.add(Dense(16, activation='relu'))  # Hidden layer
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        self.history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        scores = self.model.evaluate(X_test, y_test)
        accuracy = scores[1] * 100
        return accuracy

    def predict(self, X_test):
        return (self.model.predict(X_test) > 0.5).astype("int32")

    def get_performance_metrics(self, y_test, y_pred, accuracy):
        print("test")
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_micro = f1_score(y_test, y_pred, average="micro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        precision_macro = precision_score(y_test, y_pred, average='macro')
        precision_micro = precision_score(y_test, y_pred, average='micro')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_micro = recall_score(y_test, y_pred, average='micro')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')

        # Write results to an external file
        with open('results/NNresults.txt', 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")

            f.write("Accuracy: {}\n".format(accuracy/100))
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
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        display.plot()
        # Save the plot as an image file
        plt.savefig('results/NN_confusion_matrix.png')
        # Close the plot to release memory
        plt.close()


