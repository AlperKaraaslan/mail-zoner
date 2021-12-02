from operator import itemgetter
from collections import Counter
import random

import pandas as pd

from ..util.util import load_jsonl, flatten_list
from ..parameters import MAXLENMAIL


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


def label_occurrences(df):
    return df.groupby('label').count().applymap(lambda s: s/df.count().to_list()[0] * 100)['id']


class Preprocess_webis:
    def __init__(self,jsonldir,reduced=False,N_reduced=10000,remove_long_mails=True,test_split=0.2):
        self.jsonldir = jsonldir
        self.reduced = reduced
        self.N_reduced = N_reduced

        self.remove_long_mails = remove_long_mails
        self.test_split = test_split

        # assert 0< test_split <=1 or test_split==None, 'test_split must be between [0,1]'


    def string_to_text(self,text_sample,start,end,tag):
        '''
        Convert to list of labels as we habe a string containing the text "text_sample" and the corresproning labels "tag"
        with "start" and "end" of the text_sample. We make a list "text_list" with corresponfing label_list "label_list"
        '''
        text = text_sample[start:end]
        text_list = text.split("\n")
        
        N = len(text_list)
        label_list = [tag] * N
        
        return (text_list,label_list)


    def align_labels(self,data_list):
        '''

        '''
        newdata = {}
        for sample in data_list:
            id = sample['id']
            text = sample['text']
            label = sample['labels']
            label_list = sorted(label, key=itemgetter(1))
            
            newdata[id] = {}
            newdata[id]['text']=[]
            newdata[id]['label']=[]
            
            for comp in label_list:
                start,end,tag = comp[0], comp[1], comp[2]
                
                txt_list,lbl_list = self.string_to_text(text,start,end,tag)
                newdata[id]['text'].append(txt_list)
                newdata[id]['label'].append(lbl_list)
                
        return newdata


    def flatten_instance(self,sample):
        '''
        Flatten list of list into list for single key
        '''
        text = sample['text']
        labels = sample['label']

        # Flatten 
        text_flat = flatten_list(text)
        labels_flat = flatten_list(labels)
        
        return text_flat,labels_flat


    def flatten(self,dataset):
        '''
        Flatten list of list into list for all keys
        '''
        flattened_dataset={}
        for key in dataset:
            flattened_dataset[key] = {}
            
            sample = dataset[key]
            
            text_flat,labels_flat = self.flatten_instance(sample)

            
            flattened_dataset[key]['label'] = labels_flat
            flattened_dataset[key]['text'] = text_flat
        
        return flattened_dataset


    def prepare_dataset(self,flattened_dataset):  #list of str #!!!
        ''' 
        Makes a single df out of all data
        '''
        reduced = self.reduced
        N_reduced = self.N_reduced
                
        labels=[]
        texts = []
        ids = []
        for key in flattened_dataset:
            txt = flattened_dataset[key]['text']
            lbl = flattened_dataset[key]['label']
            
            texts.append(txt)
            labels.append(lbl)
            ids.append(key)

        labels_flat = [item for sublist in labels for item in sublist]
        texts_flat = [item for sublist in texts for item in sublist]

        ids_new=[]
        for txt,key in zip(texts,ids):
            N = len(txt)
            ids_new.append(N*[key])

        ids_flat = [item for sublist in ids_new for item in sublist]


        df_data = {'id':ids_flat,
                   'text':texts_flat,
                   'label':labels_flat}

        df = pd.DataFrame(df_data)
        
        if reduced:
            return df.head(N_reduced)
        
        return df
        

    def del_long_mails(self,dataset_flattened):
        '''
        Delete key where length of email is longer than MAXLENMAIL
        '''
        toremove=[]
        for key,val in dataset_flattened.items():
            N = len(val['text'])

            if N>MAXLENMAIL:
                toremove.append(key)
                
        for key in toremove:
            dataset_flattened.pop(key)
        
        return dataset_flattened

    def reorder_data(self): #careful with return, when test_split is 0
        test_split = self.test_split
        remove_long_mails = self.remove_long_mails
        
        
        reduced = self.reduced
        N_reduced = self.N_reduced
        
        data_list = load_jsonl(self.jsonldir)
        dataset = self.align_labels(data_list)
        
        dataset_flattened =self.flatten(dataset)

        if remove_long_mails:
            dataset_flattened = self.del_long_mails(dataset_flattened)
        
        if reduced:
            keys = list(dataset_flattened.keys())
            ids = random.sample(keys,k=N_reduced)
            
            filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
            dataset_flattened = filter(dataset_flattened,tuple(ids))
                    
        if test_split:
            dataset_train,dataset_test = split_dataset(dataset_flattened,test_split)

            return dataset_train,dataset_test
        
        return dataset_flattened,None


    def make_tabular(self):
        data_list = load_jsonl(self.jsonldir)
        dataset = self.align_labels(data_list)

        dataset_flattened =self.flatten(dataset)

        df_final = self.prepare_dataset(dataset_flattened)
        return df_final