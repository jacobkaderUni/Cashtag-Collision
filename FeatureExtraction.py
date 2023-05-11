import json
from datetime import datetime, time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class FeatureExtractor:

    def __init__(self, pathIn, minDF):
        self.pathIn = pathIn
        self.minDF = minDF

    def getBagOfWords(self, myArray):
        vectorizer = CountVectorizer(min_df=self.minDF)
        vectorizer.fit(myArray)
        bow = vectorizer.transform(myArray).toarray()
        # NLT K, to get rid of stop words
        # Save the feature labels to a file
        with open('results/feature_labels.txt', 'w', encoding='utf-8') as f:
            for label in vectorizer.get_feature_names_out():
                f.write(label + '\n')

        return bow

    def getTextInArrayFromJson2(self):
        features = []

        with open(self.pathIn, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    if 'text' in data:
                        if 'extended_tweet' in data:
                            # feature = data['extended_tweet']['full_text']
                            features.append(data['extended_tweet']['full_text'])
                        else:
                            # feature = data['text']
                            features.append(data['text'])
                    else:
                        features.append('')
                except json.JSONDecodeError:
                    pass

        return features

    def getCryptoTweets(self):
        features = []
        alertWord = "coin"
        with open(self.pathIn, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    text = data.get('text', '') + (data.get('user', {}).get('description', '') or '')
                    if alertWord in text:
                        features.append(1)
                    else:
                        features.append(0)
                except json.JSONDecodeError:
                    pass

        return features

    def getVerificationCheck(self):
        features = []
        with open(self.pathIn, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    if 'user' in data and 'verified' in data['user']:
                        if data['user']['verified']:
                            features.append(1)
                        else:
                            features.append(0)
                    else:
                        features.append('')
                except json.JSONDecodeError:
                    pass

        return features

    def getTweetLang(self):
        features = []
        with open(self.pathIn, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    if data.get('lang', '') == 'en':
                        features.append(1)
                    else:
                        features.append(0)
                except json.JSONDecodeError:
                    pass
        return features

    def getUserLang(self):
        features = []
        with open(self.pathIn, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    if data.get('user', {}).get('lang', '') == 'en':
                        features.append(1)
                    else:
                        features.append(0)
                except json.JSONDecodeError:
                    pass
        return features

    def getTimeTweeted(self):
        features = []
        with open(self.pathIn, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    created_at = data.get('user', {}).get('created_at', '')
                    # convert the string to a datetime object
                    tweet_time = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y')

                    # extract the time
                    tweet_time = tweet_time.time()

                    if time(7, 30) <= tweet_time <= time(17, 30):
                        features.append(1)
                    else:
                        features.append(0)
                except json.JSONDecodeError:
                    pass

        return features

    def getTweetReplyCount(self):
        features = []
        with open(self.pathIn, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    if data.get('reply_count') > 0:
                        features.append(1)
                    else:
                        features.append(0)

                except json.JSONDecodeError:
                    pass
        return features

    def getTweetRetweetCount(self):
        features = []
        with open(self.pathIn, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    if data.get('retweet_count') > 0:
                        features.append(1)
                    else:
                        features.append(0)

                except json.JSONDecodeError:
                    pass
        return features

    def getTweetFavouriteCount(self):
        features = []
        with open(self.pathIn, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    if data.get('favorite_count') > 0:
                        features.append(1)
                    else:
                        features.append(0)

                except json.JSONDecodeError:
                    pass
        return features

    def getLabels(self):
        labels = []
        with open(self.pathIn, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    if 'label' in data:
                        label = data['label']
                        labels.append(label)
                    else:
                        label = None
                        labels.append(label)
                except json.JSONDecodeError:
                    pass

        return labels

    def getFeatures(self):
        X = []
        feature1 = self.getBagOfWords(self.getTextInArrayFromJson2())
        feature2 = self.getCryptoTweets()
        feature3 = self.getVerificationCheck()
        feature4 = self.getTweetLang()
        feature5 = self.getUserLang()
        feature6 = self.getTimeTweeted()
        feature7 = self.getTweetReplyCount()
        feature8 = self.getTweetRetweetCount()
        feature9 = self.getTweetFavouriteCount()
        for i in range(len(feature2)):
            combined_feature = np.append(np.append(np.append(np.append(
                np.append(np.append(np.append(np.append(feature1[i], feature2[i]), feature3[i]), feature4[i]),
                          feature5[i]),
                feature6[i]), feature7[i]), feature8[i]), feature9[i])
            X.append(combined_feature)
        return X
