from collections import Counter
import json

import tensorflow as tf
import jsonlines

from ..parameters import LABELMAPPINGS, NLABELS


def flatten_list(input_list):
    return [item for sublist in input_list for item in sublist]


def load_jsonl(jsonldir):
    data_list=[]
    with jsonlines.open(jsonldir) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


def save_json(data,savedir):
    with open(savedir, 'w') as f:
        json.dump(data, f,ensure_ascii=True, indent=4, sort_keys=False)    


def load_json(data_dir):
    with open(data_dir,encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_jsonlines(data,savedir):
    with jsonlines.open(savedir, mode='w') as writer:
        for element in data:
            writer.write(element)


def read_jsonlines(data_tok_dir,position):
    counter=0
    with jsonlines.open(data_tok_dir) as reader:
        for obj in reader:
            if counter == position:
                key = list(obj.keys())[0]
                return obj[key]
            counter+=1


def compute_class_weights(encoded_labels):
    N = len(encoded_labels)
    occurences = Counter(encoded_labels)

    class_weights={}
    for key in occurences:
        class_weights[key] = N/occurences[key]
        
    return class_weights


def onehot(encoded_labels):
    '''
    One hot encoded labels
    '''
    return tf.keras.utils.to_categorical(encoded_labels, num_classes=NLABELS, dtype="float32")


def encode_labels(label_list):
    '''
    Encode labels from categorical to numerical
    '''

    labels_encoded=[]
    for lbl in label_list:
        encoded_label = LABELMAPPINGS[lbl]
        labels_encoded.append(encoded_label)
    
    return labels_encoded


def get_id_startend_mappings(ids):
    id_positions={}
    N = len(ids)

    reversed_list = ids[::-1]
    for key in set(ids):
        start = ids.index(key)
        end = N - reversed_list.index(key)
        
        id_positions[key] = (start,end)
    return id_positions