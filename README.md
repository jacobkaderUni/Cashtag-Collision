# Cashtag-Collision on the London Stock Exchange group

Training different supervised classifiers to identify which tweets belong to the LSE.

## Cashtags

A Cashtag is a company ticker symbol preceded by the U.S. dollar sign, e.g. $TWTR. When you click on a cashtag, you’ll see other tweets mentioning that same ticket symbol.

## What is Cashtag-Collision?

One of the principal issues with including Cashtags in tweets is a phenomenon known as a 'Cashtag-Collision', which can occur when two different companies or entities share the same ticker symbol on Twitter, resulting in confusion among analysts and investor. This could also potentially affect stock prices. Investors must therefore be very careful when researching a ticker symbol.

## Application

6 supervised classifiers will be built and trained(fitted) using features which have carefully been extracted from the Tweets.JSON data set.

- NAIVE BAYES
- DECISION TREES (inc. hyperparam)
- K NEAREST NEIGHBHORS (inc. hyperparam)
- RANDOM FOREST (inc. hyperparam)
- SUPPORT VECTOR MACHINE (inc. hyperparam)
- NEURAL NETWORKS

4 of the classifiers included hyperparameters to optmise their performance. The classifiers were trained on several features:

- BoW, tweet text converted into sparse vectors.
- Crypto tweet, checking of the tweet is promoting a crypto currency.
- Verified user, check if the user is verified.
- Tweet Language.
- User language.
- Time tweeted, the time of the tweet can tell us various things such as which Stock exchange is being tweeted about.
- Tweet reply count.
- Tweet retweet count.
- Tweet favorite count.

During the process the time taken was calculated and used to evaluate the performance of the classifiers along with several other metrics:

- Matthew's correlation coefficient
- Accuracy
- F1 Score
- Precision
- Recall

## Results

Results for this project can be viewed in the results folder.

## Technologies used for this project

Python 3.10.9 was used for this project.

Link: https://www.anaconda.com/download

### IDE used

Pycharm

Link: https://www.jetbrains.com/pycharm/download/#section=mac

### Libraries and packages used

Using pip

Scikit-learn

$ pip install -U scikit-learn

Matplotlib

$ pip install -U matplotlib

Numpy

$ pip install numpy

Pandas

$ pip install pandas

Tensorflow

$ pip install tensorflow

(If you struglle to get tensor installed used Anaconda packet manager instead)

