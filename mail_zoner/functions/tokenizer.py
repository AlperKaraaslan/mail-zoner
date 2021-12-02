import pandas as pd
from transformers import AutoTokenizer

from ml_classification_preprocessing_email_zone.util.util import encode_labels
from ml_classification_preprocessing_email_zone.parameters import TOKENIZER, PAD


def tokenize_sample(tokenizer,data_sample_text,data_sample_labels):
    '''
    Just for exploratory usage
    '''
    tokenized = tokenizer(data_sample_text)

    input_ids = tokenized['input_ids'] #list of list
    attention_mask = tokenized['attention_mask'] #list of list
    # tokenizer_decoded = [tokenizer.decode(i) for i in input_ids] #list
    #can also do: tokenizer.tokenize("Hi my name is Alper")
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in input_ids] #list of list

    # visualize tokens
    input_ids_flat = [', '.join(map(str, i)) for i in input_ids]
    attention_mask_flat = [', '.join(map(str, i)) for i in attention_mask]
    tokens_flat = [" ".join(i) for i in tokens]

    data_df = {'input_ids':input_ids_flat,
                'attention_mask':attention_mask_flat,
                'tokens':tokens_flat,
                'labels':data_sample_labels,
                'text':data_sample_text}

    df_tokens = pd.DataFrame(data_df)
    
    return df_tokens

class Tokenizer:
    def __init__(self,modelname=TOKENIZER):
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)

    def tokenize_dataset(self,dataset):
        '''
        https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase
        '''
        
        tokenizer = self.tokenizer

        data={}
        for key in dataset:
            data[key] = {}
            
            text = dataset[key]['text']

            tokenized = tokenizer(text,
                                  padding=PAD,
                                  truncation=True,
                                  return_tensors='np',
                                  #   return_length=True, #truncated to max length
                                  )
            
            data[key]['tokenized'] = tokenized
            data[key]['labels_encoded'] = encode_labels(dataset[key]['label'])
            
        return data