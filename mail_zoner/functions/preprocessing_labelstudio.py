from operator import itemgetter
from collections import Counter
import warnings
import random

import pandas as pd

from ..util.util import load_json

NEWLINELABEL='<empty>'

def email_length(ids):
    return Counter(ids)


def split_dataset(dataset_flattened,split_percentage):
    '''
    dataset_flattened: keys are IDs 
    '''
    keys = list(dataset_flattened.keys())
    N = len(keys)

    N_sample = int(N*split_percentage)
    keys_sampled = random.sample(keys,N_sample)

    dataset_sampled = {k: dataset_flattened[k] for k in keys_sampled}

    for key in keys_sampled:
        del dataset_flattened[key]
        
    return dataset_flattened,dataset_sampled


class Preprocess_labelstudio:
    def __init__(self,jsondir,reduced=False,N_reduced=15):
        self.jsondir = jsondir #path to json file downloaded from Labelstudio
        self.reduced = reduced #reduce the number of data to process
        self.N_reduced = N_reduced # Number of samples to process


    def reorder_dataset(self,data_dir):
        '''
        Reorder structure coming from labelstudio to {id:{'text':str,'label_info':list(dict(start,end,label))}}
        '''
        data_json = load_json(data_dir)

        compact_dataset={}
        for mail in data_json: #list of #emails coming from LabelStudio
            identifier = mail['id'] #each sample has an unique id
            compact_dataset[identifier] = {}

            raw_text = mail['data']['ner'] #raw text as str
            compact_dataset[identifier]['text'] = raw_text
            
            labelList = mail['annotations'][0]['result']  #label list for one email i.e. each line is one label pair

            positions_labels_list=[] #save all in one list as tuple: (start,end,label)
            for lbl in labelList:
                start = lbl['value']['start']
                end = lbl['value']['end']
                label = lbl['value']['labels'][0]
                
                assert end > start, f"Label strategy wrong as end < start of string with if: {identifier}"
                assert len(lbl['value']['labels']) ==1, f"ID {identifier} not labeleded properly. Overlapping labels"


                positions_labels_list.append( (start,end,label ) ) 

            positions_labels_list_sorted = sorted(positions_labels_list, key=itemgetter(0)) #sort (start, end ,label) by start

            compact_dataset[identifier]['label_info'] = positions_labels_list_sorted
            
        return compact_dataset


    def flatten_dataset_instance(self,email,identifier):
        '''
        For each sample, text will be split into \n, & labels will be added
        '''
        text = email['text'] #raw email content with newlines
        text_split = text.split("\n") #split email content w/o newlines

        label_info = email['label_info']

        final_label_list = len(text_split)*[None] #as we split our input text into \n, we need to create the associate labels for it

        for lbl in label_info: #algorithm to map (start,end,label) to list of labels
            start,end,label = lbl #e.g. (0,9,Rest)

            text_extracted = text[start:end] #raw text extracted from 0:9
            text_extracted_split = text_extracted.split("\n")

            N_split = len(text_extracted_split) #length of raw text extracted and split by \n

            l_split = N_split*[label] #label list, same length as text_extracted_split

            #find sublist: text_extracted_split in text_split
            N_found=0
            for i in range(len(text_split)-N_split): #sliding integer approach: find sublist in string
                if text_split[i:i+N_split] == text_extracted_split:
                    N_found+=1
                    if N_found >1:
                        warnings.warn(f"By searching excerpt we found multiple loacations. You may check ID: {identifier} if it was labeled correctly.")

                    final_label_list[i:i+N_split] = l_split

        final_label_list = [NEWLINELABEL if i is None else i for i in final_label_list] #for all \n which were not labeled in Labelstudio, casz NEWLINELABEL

        return text_split,final_label_list


    def flatten_dataset(self,emails):
        '''
        Flatten all samples, as described in flatten_dataset_instance method
        '''
        dataset={}
        for mailID in emails:
            dataset[mailID] = {}
            email = emails[mailID]
            
            text,label = self.flatten_dataset_instance(email,mailID)
            dataset[mailID] = {'text':text,
                               'label':label}
        return dataset


    def apply(self,test_split=0.2,val_split=0.2):
        assert test_split is not None and val_split is not None
        assert 0<test_split<1 and 0<val_split<1
        
        reduced = self.reduced
        N_reduced = self.N_reduced
        
        jsondir = self.jsondir
        
        # Reordered dataset converted from LabelStudio
        emails = self.reorder_dataset(jsondir)

        # Split text by \n and align labels
        dataset = self.flatten_dataset(emails)
        
        if reduced:
            keys = list(dataset.keys())
            ids = random.sample(keys,k=N_reduced)
            
            filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
            dataset = filter(dataset,tuple(ids))
            

        if test_split:
            dataset_temp,dataset_test = split_dataset(dataset,test_split)
        if val_split:
            dataset_train,dataset_val = split_dataset(dataset_temp,val_split)

        return dataset_train,dataset_val,dataset_test


    def tabularized_dataset(self,dataset):
        reduced = self.reduced
        N_reduced = self.N_reduced
        
        text_list=[]
        id_list=[]
        label_list=[]

        for key in dataset:
            text = dataset[key]['text']
            label = dataset[key]['label']
            key_list = len(text)*[key]

            text_list.extend(text)
            id_list.extend(key_list)
            label_list.extend(label)

        df_data = {'id':id_list,
                   'text':text_list,
                   'label':label_list}

        df = pd.DataFrame(df_data)

        if reduced:
            return df.head(N_reduced)
        return df


    def show_tabular(self):
        jsondir = self.jsondir
        
        # Reordered dataset converted from LabelStudio
        emails = self.reorder_dataset(jsondir)

        # Reordered flattened dataset
        dataset = self.flatten_dataset(emails)

        df_final = self.tabularized_dataset(dataset)
        return df_final