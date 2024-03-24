from datasets import load_dataset
import json

'''

# DocRED paper: https://arxiv.org/pdf/1906.06127.pdf

all entities in DocRED are grouped together by co-reference resolution, 
and each group is treated as a single entity.

We want to split that up so that each mention is a separate entity.
This involves duplicating the relationships shared across the mentions of the same entity.

Also getting the rest of the data in the right format for GLiNER training (ner, tokenized_text)

'''

def map_entities(doc):
    # Create a mapping for the entities
    entity_mapping = {}
    for entity_id, mentions in enumerate(doc['vertexSet']):
        for mention in mentions:
            entity_mapping[(entity_id, mention['sent_id'])] = mention

    '''
    {(0, 4): {'name': 'Skai TV', 'sent_id': 4, 'pos': [0, 2], 'type': 'ORG'},
    (0, 0): {'name': 'Skai TV', 'sent_id': 0, 'pos': [0, 2], 'type': 'ORG'},
    '''
    return entity_mapping


def expand_labels(doc, entity_mapping):
    # Expand the labels to individual mentions
    expanded_labels = []
    labels = doc['labels']
    for i, relation in enumerate(zip(labels['head'], labels['tail'], labels['relation_id'], labels['relation_text'], labels['evidence'])):
        head_id, tail_id, relation_id, relation_text, evidence = relation
        # Expand the head and tail mentions
        for head_mention in doc['vertexSet'][head_id]:
            for tail_mention in doc['vertexSet'][tail_id]:
                expanded_relation = {
                    'head': head_mention,
                    'tail': tail_mention,
                    'relation_id': relation_id,
                    'relation_text': relation_text,
                    'evidence': evidence
                }
                expanded_labels.append(expanded_relation)

    return expanded_labels

def get_ner_from_entity_mapping(entity_mapping):
    ner = []
    for entity_id, mention in entity_mapping.items():
        ner.append([mention['pos'][0], mention['pos'][1], mention['type']])
    return ner

#
dataset = load_dataset("docred")    # features: ['relation', 'tokens', 'head', 'tail', 'names'],

data = []
for i in range(10):
    doc = dataset['validation'][i]
    example_row = {}

    entity_mapping = map_entities(doc)

    example_row['ner'] = get_ner_from_entity_mapping(entity_mapping)  #  "ner": [[3,6,"LOC"], [7,9,"PER"]]
    example_row['relations'] = expand_labels(doc, entity_mapping)

    # Combine the list of sentences into one list of tokens
    example_row['tokenized_text'] = [token for sentence in doc['sents'] for token in sentence]

    data.append(example_row)

# # Output the expanded labels in a JSON format for the user to verify
# expanded_labels_json = print(json.dumps(data[1], indent=2))

# save to a jsonl file
with open('docred_expanded.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + "\n")