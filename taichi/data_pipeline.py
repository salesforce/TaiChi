"""
expected input format: clinc150
likable features for sample class
1. able to see how many samples per class are there
2. able to sample them based on the number n

"""
import pandas as pd
import json
import re
import os

from collections import Counter

class DataPipeline(object):
    """
    expects path with csv or json files to process and sample them
    default language set is english with the onus on user to change 
    the language in case of working with a different language
    """
    def __init__(self, name, data_path, language='en_US'):
        self.name = name
        self.data_path = data_path
        self.language = language
        self.raw_data = None


    def sample_from_csv(self, n_shot=None):
        """
        expects a file in csv format as follows:
        utterance,language,label (no headers and no index)
        
        Example:
        book a ticket from San Francisco to New York,en_US,Book a Flight
        
        returns a subsampled pandas dataframe with utterance, language and label columns
        """        
        df = pd.read_csv(self.data_path, names=['utterance', 'language', 'label'])
        examples_per_class = dict(Counter((df.label)))
        minimum_examples_per_class = min(examples_per_class.values())
        if minimum_examples_per_class >= n_shot:
            subsampled_df = df.groupby('label').sample(n=n_shot, random_state=0)
            return subsampled_df
        else:
            error_message = "number of examples per class are not enough to sample based on n_shot={} value".format(n_shot)
            raise Exception(error_message)        


    def sample_from_json(self, n_shot=None, split='train'):
        """
        expects a file in json format as follows:
        {split: list(list containing utterance and label)}
        
        Example:
        {'train':[[utterance1, label1], [utterance2, label2], ... 'test':[[...]]}
        
        returns a subsampled pandas dataframe with utterance, language and label columns
        """
        with open(self.data_path, 'r') as input_json:
            self.raw_data = json.load(input_json)

        df = pd.DataFrame.from_records(self.raw_data[split], columns=['utterance', 'label'])
        df['language'] = self.language # for clinc150, how to do for other languages?
        cols = list(df.columns)
        cols = cols[:1] + [cols[-1]] + cols[1:-1]
        df = df[cols]
        examples_per_class = dict(Counter((df.label)))
        minimum_examples_per_class = min(examples_per_class.values())
        if minimum_examples_per_class >= n_shot:
            subsampled_df = df.groupby('label').sample(n=n_shot, random_state=0)
            return subsampled_df
        else:
            error_message = "number of examples per class are not enough to sample based on n_shot={} value".format(n_shot)
            raise Exception(error_message)


    def save_subsampled_data_to_csv(self, save_dir, n_shot=None, split='train', is_json=False):
        """
        saves the subsampled data from csv/json into csv with utterance,
        language and label columns with no headers and no index in the requisite
        save directory
        """
        if is_json:
            subsampled_df = self.sample_from_json(n_shot=n_shot, split=split)
        else:
            subsampled_df = self.sample_from_csv(n_shot=n_shot)

        save_path = os.path.join(save_dir, self.name + '_' + str(n_shot) + 
                                '_shot_' + split + '.csv')
        subsampled_df.to_csv(save_path, header=None, index=None)


    def save_subsampled_data_to_json(self, save_dir, n_shot=None, split='train', is_json=False):
        """
        saves the subsampled data from csv/json into json with individual utterance,
        language and label values as an independent array in the requisite
        save directory
        """
        if is_json:
            subsampled_df = self.sample_from_json(n_shot=n_shot, split=split)
        else:
            subsampled_df = self.sample_from_csv(n_shot=n_shot)

        save_path = os.path.join(save_dir, self.name + '_' + str(n_shot) + 
                                '_shot_' + split + '.json')
        subsampled_df.to_json(save_path, orient='values')        
