import json
import nltk

from senpy.plugins import SentimentPlugin
from senpy.models import Response, Opinion, Entry


class BayesPlugin(SentimentPlugin):

    def activate(self):
        tweets = [
            (['love', 'this', 'car'], 'marl:Positive'),
            (['this', 'view', 'amazing'], 'marl:Positive'),
            (['feel', 'great', 'this', 'morning'], 'marl:Positive'),
            (['excited', 'about', 'the', 'concert'], 'marl:Positive'),
            (['best', 'friend'], 'marl:Positive'),
            (['not', 'like', 'this', 'car'], 'marl:Negative'),
            (['this', 'view', 'horrible'], 'marl:Negative'),
            (['feel', 'tired', 'this', 'morning'], 'marl:Negative'),
            (['not', 'looking', 'forward', 'the', 'concert'], 'marl:Negative'),
            (['enemy'], 'marl:Negative')]

        def get_words_in_tweets(tweets):
            all_words = []
            for (words, sentiment) in tweets:
                all_words.extend(words)
            return all_words

        def get_word_features(wordlist):
            wordlist = nltk.FreqDist(wordlist)
            word_features = wordlist.keys()
            return word_features

        self._word_features = get_word_features(get_words_in_tweets(tweets))


        training_set = nltk.classify.apply_features(self.extract_features, tweets)

        self._classifier = nltk.NaiveBayesClassifier.train(training_set)

    def get_sentiment(self, text):
        return self._classifier.classify(self.extract_features(text.split()))

    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in self._word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    def analyse(self, **params):
        response = Response()
        text = params['input']
        polarity = self.get_sentiment(text)
        entry = Entry(text=text)
        opinion = Opinion(hasPolarity=polarity)
        opinion["prov:wasGeneratedBy"] = self.id
        entry.opinions.append(opinion)
        response.entries.append(entry)
        return response
