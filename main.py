# -*- coding: utf-8 -*-


""" IMPORTS NEEDED """
import matplotlib.ticker as mtick
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import time
from ModalTuner import ModelTuner
from FeatureExtraction import FeatureExtractor
from NB import NaiveBayesModel
from DT import DecisionTreeModel
from KNN import KNeighborsModel
from RF import RandomForestModel
from SVM import SVMmodel
from NN import NeuralNetworkModel
import time
""" END """


""" EXTRACTING THR FEATURES"""
def extract_features():
    # START TIME
    features_start_time = time.perf_counter()
    feature_extractor = FeatureExtractor('tweets.json', 10)
    # Call the methods to extract features
    X = feature_extractor.getFeatures()
    # Extract labels
    y = feature_extractor.getLabels()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    # END TIME
    features_end_time = time.perf_counter()
    features_runtime = features_end_time - features_start_time
    print(f"The runtime of feature extraction is {features_runtime: .5f} seconds.")
    return X_train, X_test, y_train, y_test
""" END """


""" NAIVE BAYES """
def naive_bayes(X_train, y_train, X_test, y_test):
    #
    # START TIME
    nb_start_time = time.perf_counter()
    # Instantiate the NaiveBayesModel class
    nb_model = NaiveBayesModel()
    # Train the model
    nb_model.train(X_train, y_train)
    # Make predictions
    NBpred = nb_model.predict(X_test)
    # Evaluate the model
    nb_model.evaluate(y_test, NBpred)
    # END TIME
    nb_end_time = time.perf_counter()
    nb_runtime = nb_end_time - nb_start_time
    print(f"The total runtime of NB classifier is {nb_runtime: .5f} seconds.")
    # write the runtime to the file
    with open('results/NBresults.txt', 'a') as f:
        f.write(f"\nRuntime: {nb_runtime}\n")
    #
    return nb_model
""" END """


""" DECISION TREES """
def decision_trees(tuner, X_train, y_train, X_test, y_test):
    #
    # START TIME
    dt_start_time = time.perf_counter()
    # find the best hyper param
    best_paramsDT = tuner.tune_decision_tree()
    print("Best hyper parameters Decision Trees:", best_paramsDT)
    # Instantiate the DecisionTreeModel class
    dt_model = DecisionTreeModel(best_params=best_paramsDT)
    # Train the model
    dt_model.fit(X_train, y_train)
    # Make predictions
    DTpred = dt_model.predict(X_test)
    # Evaluate the model
    dt_model.evaluate(y_test, DTpred)
    # Plot feature importances and save the plot as an image
    dt_model.plot_feature_importances(X_train)
    # END TIME
    dt_end_time = time.perf_counter()
    dt_runtime = dt_end_time - dt_start_time
    print(f"The total runtime of DT classifier is {dt_runtime: .5f} seconds.")
    # write the runtime to the file
    with open('results/DTresults.txt', 'a') as f:
        f.write(f"\nRuntime: {dt_runtime}\n")
        f.write(f"\nHyper param: {best_paramsDT}\n")
    #
    return dt_model
""" END """


""" K NEAREST NEIGHBORS """
def k_nearest_neighbhors(tuner, X_train, y_train, X_test, y_test):
    #
    # START TIME
    knn_start_time = time.perf_counter()
    # find the best hyper param
    best_paramsKNN = tuner.tune_knn()
    print("Best hyperparameters KNN:", best_paramsKNN)
    # Instantiate the DecisionTreeModel class
    knn_model = KNeighborsModel(best_params=best_paramsKNN)
    # Train the model
    knn_model.fit(X_train, y_train)
    # Make predictions
    KNNpred = knn_model.predict(X_test)
    # Evaluate the model
    knn_model.evaluate(y_test, KNNpred)
    # END TIME
    knn_end_time = time.perf_counter()
    knn_runtime = knn_end_time - knn_start_time
    print(f"The total runtime of KNN classifier is {knn_runtime: .5f} seconds.")
    # write the runtime to the file
    with open('results/KNNresults.txt', 'a') as f:
        f.write(f"\nRuntime: {knn_runtime}\n")
        f.write(f"\nHyper param: {best_paramsKNN}\n")
    #
    return knn_model
""" END """


""" RANDOM FOREST """
def random_forest(tuner, X_train, y_train, X_test, y_test):
    #
    # START TIME
    rf_start_time = time.perf_counter()
    # find hyper param
    best_paramsRF = tuner.tune_random_forest()
    print("Best hyperparameters Random Forest:", best_paramsRF)
    # Instantiate the DecisionTreeModel class
    rf_model = RandomForestModel(best_params=best_paramsRF)
    # Train the model
    rf_model.fit(X_train, y_train)
    # Make predictions
    RFpred = rf_model.predict(X_test)
    # Evaluate the model
    rf_model.evaluate(y_test, RFpred)
    # END TIME
    rf_end_time = time.perf_counter()
    rf_runtime = rf_end_time - rf_start_time
    print(f"The total runtime of RF classifier is {rf_runtime: .5f} seconds.")
    # write the runtime to the file
    with open('results/RFresults.txt', 'a') as f:
        f.write(f"\nRuntime: {rf_runtime}\n")
        f.write(f"\nHyper param: {best_paramsRF}\n")
    #
    return rf_model
""" END """


""" SUPPORT VECTOR CLASSIFICATION """
def support_vector_classification(tuner, X_train, y_train, X_test, y_test):
    #
    # START TIME
    svm_start_time = time.perf_counter()
    # Best hyper param
    #
    best_paramsSVM = tuner.tune_svm()
    print("Best hyperparameters SVM: ", best_paramsSVM)
    # Instantiate the DecisionTreeModel class
    svm_model = SVMmodel(best_params=best_paramsSVM)
    # Train the model
    svm_model.fit(X_train, y_train)
    # Make predictions
    SVMpred = svm_model.predict(X_test)
    # Evaluate the model
    svm_model.evaluate(y_test, SVMpred)
    # END TIME
    svm_end_time = time.perf_counter()
    svm_runtime = svm_end_time - svm_start_time
    print(f"The total runtime of SVM classifier is {svm_runtime: .5f} seconds.")
    # write the runtime to the file
    with open('results/SVMresults.txt', 'a') as f:
        f.write(f"\nRuntime: {svm_runtime}\n")
        f.write(f"\nHyper param: {best_paramsSVM}\n")
    #
    return svm_model
""" END """


""" NEURAL NETWORKS """
def neural_networks(X_train, y_train, X_test, y_test):
    #
    # START TIME
    nn_start_time = time.perf_counter()
    # convert dataset into vstack
    my_Xtrain = np.vstack(X_train)
    my_Xtest = np.vstack(X_test)
    my_Ytrain = np.vstack(y_train)
    my_Ytest = np.vstack(y_test)
    # initiate the model
    nn_classifier = NeuralNetworkModel(input_dim=my_Xtrain.shape[1])
    # train the model
    nn_classifier.train(my_Xtrain, my_Ytrain, my_Xtest, my_Ytest, epochs=10, batch_size=32)
    nn_accuracy = nn_classifier.evaluate(my_Xtest, my_Ytest)
    nn_pred = nn_classifier.predict(my_Xtest)
    # evaluate the model
    performance_metrics = nn_classifier.get_performance_metrics(my_Ytest, nn_pred, nn_accuracy)
    # END TIME
    nn_end_time = time.perf_counter()
    nn_runtime = nn_end_time - nn_start_time
    print(f"The total runtime of NN classifier is {nn_runtime: .5f} seconds.")
    # write the runtime to the file
    with open('results/NNresults.txt', 'a') as f:
        f.write(f"\nRuntime: {nn_runtime}\n")
    #
    return nn_classifier
""" END """

""" GET METRICS """
def get_metric(filename, metric_name):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(metric_name):
                metric_value = line.split(":")[-1].strip()
                return float(metric_value)
        return None
""" END """

""" PLOT METRICS """
def plot_metric(data_file, metric_name, output_file):
    # Extract the data from the file using the metric name
    data = [get_metric(data_file[i], metric_name) for i in range(len(data_file))]
    # Convert the data to percentages
    data = [i * 100 for i in data]
    print(data)
    # Create horizontal bar chart
    classifiers = ('NB', 'DT', 'KNN', 'RF', 'SVM', 'NN')
    y_pos = np.arange(len(classifiers))
    plt.barh(y_pos, data)
    plt.yticks(y_pos, classifiers)

    plt.title(f"Classifier {metric_name}")
    plt.xlabel(f"{metric_name} (%)")  # Indicate that x axis is in percentages
    plt.ylabel("classifiers")

    # Add the values at the end of the bars
    for i, v in enumerate(data):
        plt.text(v + 1, i - 0.1, f"{round(v, 1)}%")  # Display the values as percentages

    # Format x-axis to display as percentage
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())

    # Save the chart to a file
    plt.savefig(os.path.join('results', output_file))
    # Clear the plot
    plt.clf()
    # Return the data
    return data
""" END """


def main():
    if not os.path.exists('results'):
        os.makedirs('results')
    X_train, X_test, y_train, y_test = extract_features()
    tuner = ModelTuner(X_train, y_train)
    
    naive_bayes(X_train, y_train, X_test, y_test)
    decision_trees(tuner, X_train, y_train, X_test, y_test)
    k_nearest_neighbhors(tuner, X_train, y_train, X_test, y_test)
    random_forest(tuner, X_train, y_train, X_test, y_test)
    support_vector_classification(tuner, X_train, y_train, X_test, y_test)
    neural_networks(X_train, y_train, X_test, y_test)

    files = ["results/NBresults.txt", "results/DTresults.txt", "results/KNNresults.txt", "results/RFresults.txt",
             "results/SVMresults.txt", "results/NNresults.txt"]
    accuracy_values = plot_metric(files, "Accuracy", "accuracy.png")
    precision_values = plot_metric(files, "Precision", "precision.png")
    F1_values = plot_metric(files, "F1 Score", "F1_score.png")
    recall_values = plot_metric(files, "Recall", "recall.png")
    Mcc = plot_metric(files, "Mcc", "mcc.png")

if __name__ == "__main__":
    main()
    print('DONE')
