import json
import random

from senpy.plugins import SentimentPlugin
from senpy.models import Response, Opinion, Entry


class SmileysPlugin(SentimentPlugin):

    def get_sentiment(self, text):
        if ':)' in text:
            return 'marl:Positive'
        elif ':(' in text:
            return 'marl:Negative'
        return 'marl:Neutral'

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
