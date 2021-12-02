# *** WEBIS DATASET SPECIFIC PARAMETERS - used in: utils
NLABELS=15
LABELMAPPINGS={'log_data': 0,
                'inline_headers': 1,
                'patch': 2,
                'salutation': 3,
                'visual_separator': 4,
                'raw_code': 5,
                'paragraph': 6,
                'technical': 7,
                'personal_signature': 8,
                'quotation': 9,
                'section_heading': 10,
                'mua_signature': 11,
                'tabular': 12,
                'quotation_marker': 13,
                'closing': 14
                }

LABELMAPPINGS_rev = {val:key for key,val in LABELMAPPINGS.items()} #reverse mappings

# *** MODEL SPECIFIC PARAMETERS

# TOKENIZER
TOKENIZER='xlm-roberta-base' 
PAD = 'max_length'

# ARCHITECTURE
MODELNAME='jplu/tf-xlm-roberta-base'
MAXLENMAIL=360 #all emails containing more than 360 lines will be discarded
MAXTIMESTEP=360 #should bne same as MAXLENMEAIL
MASKVALUE=-10

# *** TRAIN PARAMETERS
EPOCHS=10
BATCHSIZE=32

CHECKPOINTNAME='mailzone.ckpt'
