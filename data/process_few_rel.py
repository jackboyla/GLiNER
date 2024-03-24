from datasets import load_dataset
import json


dataset = load_dataset("few_rel")    # features: ['relation', 'tokens', 'head', 'tail', 'names'],
data = dataset['train_wiki'].select(range(50))
data = data.to_dict()

'''
{
    'relation': ['P931', 'P931', 'P931'], 
    'tokens': [['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', '(', 'TJQ', ')', '.'], ['The', 'name', 'was', 'at', 'one', 'point', 'changed', 'to', 'Nottingham', 'East', 'Midlands', 'Airport', 'so', 'as', 'to', 'include', 'the', 'name', 'of', 'the', 'city', 'that', 'is', 'supposedly', 'most', 'internationally', 'recognisable', ',', 'mainly', 'due', 'to', 'the', 'Robin', 'Hood', 'legend', '.'], [
        'It', 'is', 'a', 'four', '-', 'level', 'stack', 'interchange', 'near', 'Fort', 'Lauderdale', '-', 'Hollywood', 'International', 'Airport', 'in', 'Fort', 'Lauderdale', ',', 'Florida', '.']], 
        'head': [{'text': 'tjq', 'type': 'Q1331049', 'indices': [[16]]}, {'text': 'east midlands airport', 'type': 'Q8977', 'indices': [[9, 10, 11]]}, {'text': 'fort lauderdale-hollywood international airport', 'type': 'Q635361', 'indices': [[9, 10, 11, 12, 13, 14]]}], 
        'tail': [{'text': 'tanjung pandan', 'type': 'Q3056359', 'indices': [[13, 14]]}, {'text': 'nottingham', 'type': 'Q41262', 'indices': [[8]]}, {'text': 'fort lauderdale, florida', 'type': 'Q165972', 'indices': [[16, 17, 18, 19]]}], 'names': [['place served by transport hub', 'territorial entity or entities served by this transport hub (airport, train station, etc.)'], ['place served by transport hub', 'territorial entity or entities served by this transport hub (airport, train station, etc.)'], ['place served by transport hub', 'territorial entity or entities served by this transport hub (airport, train station, etc.)']]}
'''

import ipdb;ipdb.set_trace()

