# -*- coding: utf-8 -*-
import os
import re
import gc
import json
import simdjson
import numpy as np
import pandas as pd
from zipfile import ZipFile
from datasets import Dataset, DatasetDict
from data.make_datasets import Custom_Dataset

class Load_data:
    def __init__(self):
        super(Load_data, self).__init__()
    
    def load_train():
        train_data = []
        file_path = "/root/Abstract_summarization/data/train"
        for filename in os.listdir(file_path):
            with open(os.path.join(file_path, filename),'rb') as f:
                data = simdjson.loads(f.read())
            train_data.extend(data['data'])
        train_from_datasetdict = pd.DataFrame(columns = ["doc_id", "title","date","reg_no","ipc","issued_by","summary_entire","summary_section"])
        train_from_datasetdict["doc_id"] = [train[list(train_data[1].keys())[1]] for train in train_data]
        train_from_datasetdict["title"] = [train[list(train_data[2].keys())[2]] for train in train_data]
        train_from_datasetdict["date"] = [train[list(train_data[3].keys())[3]] for train in train_data]
        train_from_datasetdict["reg_no"] = [train[list(train_data[4].keys())[4]] for train in train_data]
        train_from_datasetdict["ipc"] = [train[list(train_data[5].keys())[5]] for train in train_data]
        train_from_datasetdict["issued_by"] = [train[list(train_data[6].keys())[6]] for train in train_data]
        train_from_datasetdict["summary_entire_orignal_text"] = [train[list(train_data[8].keys())[8]][0]['orginal_text'] for train in train_data]
        train_from_datasetdict["summary_entire_summary_text"] = [train[list(train_data[8].keys())[8]][0]['summary_text'] for train in train_data]
        train_from_datasetdict["summary_section_orignal_text"] = [train[list(train_data[9].keys())[9]][0]['orginal_text'] for train in train_data]
        train_from_datasetdict["summary_section_summary_text"] = [train[list(train_data[9].keys())[9]][0]['summary_text'] for train in train_data]
        
        return Custom_Dataset.from_pandas(train_from_datasetdict)
    
    def load_test():
        valid_data = []
        file_path = "/root/Abstract_summarization/data/validation"
        for filename in os.listdir(file_path):
            with open(os.path.join(file_path, filename),'rb') as f:
                data = simdjson.loads(f.read())
            valid_data.extend(data['data'])
        test_from_datasetdict = pd.DataFrame(columns = ["doc_id", "title","date","reg_no","ipc","issued_by","summary_entire","summary_section"])
        test_from_datasetdict["doc_id"] = [valid[list(valid_data[1].keys())[1]] for valid in valid_data]
        test_from_datasetdict["title"] = [valid[list(valid_data[2].keys())[2]] for valid in valid_data]
        test_from_datasetdict["date"] = [valid[list(valid_data[3].keys())[3]] for valid in valid_data]
        test_from_datasetdict["reg_ no"] = [valid[list(valid_data[4].keys())[4]] for valid in valid_data]
        test_from_datasetdict["ipc"] = [valid[list(valid_data[5].keys())[5]] for valid in valid_data]
        test_from_datasetdict["issued_by"] = [valid[list(valid_data[6].keys())[6]] for valid in valid_data]
        test_from_datasetdict["summary_entire_orignal_text"] = [valid[list(valid_data[8].keys())[8]][0]['orginal_text'] for valid in valid_data]
        test_from_datasetdict["summary_entire_summary_text"] = [valid[list(valid_data[8].keys())[8]][0]['summary_text'] for valid in valid_data]
        test_from_datasetdict["summary_section_orignal_text"] = [valid[list(valid_data[9].keys())[9]][0]['orginal_text'] for valid in valid_data]
        test_from_datasetdict["summary_section_summary_text"] = [valid[list(valid_data[9].keys())[9]][0]['summary_text'] for valid in valid_data]
        
        return Dataset.from_pandas(test_from_datasetdict)
    
    
class Preprocessing():
    def __init__(self,):
        super(Preprocessing, self).__init__()
        
    def split_entire_section(dataset):
        
        doc_id = []
        title = []
        date = []
        reg_no = []
        ipc = []
        issued_by = []
        documents = []
        summary = []
        
        for info in dataset:
            
            for i in range(2):
                doc_id.append(info['doc_id']+"_"+str(i))
                title.append(info['title'])
                date.append(info['date'])
                reg_no.append(info['reg_no'])
                ipc.append(info['ipc'])
                issued_by.append(info['issued_by'])
            
            documents.append(info['summary_entire_orignal_text'])
            documents.append(info['summary_section_orignal_text'])
            
            summary.append(info['summary_entire_summary_text'])
            summary.append(info['summary_section_summary_text'])
        
        dataset = Dataset.from_dict({
            'doc_id':doc_id,
            'title':title,
            'date':date,
            'reg_no':reg_no,
            'ipc':ipc,
            'issued_by':issued_by,
            'documents':documents,
            'summary':summary
        })
        return dataset
        
            
    def remove_sequence_character(text):
        
         # 순서 문자 제거
        special_char = re.compile("["
        u"\u2460-\u2469"  # ①
        u"\u2474-\u247F"  # ⑿
        u"\u326E-\u3279"  # ㉮
        u"\u2160-\u2169"  # Ⅳ
                           "]+", flags=re.UNICODE)
        
        text = special_char.sub(r' ', text)
        
        return text
    
    def remove_ja_mo(text):
        pattern = '([ㄱ-ㅎ ㅏ-ㅣ])+'
        text = re.sub(pattern=pattern, repl=' ',string=text)
         
        pattern = re.compile(r'\s+')
        text = re.sub(pattern=pattern, repl=' ', string=text)
        
        return text
    
    def remove_background(text):
        pattern = "배경:"
        text = re.sub(pattern=pattern, repl=' ',string=text)
        
        pattern = "연구목적:"
        text = re.sub(pattern=pattern, repl=' ',string=text)
        
        pattern = "목적:"
        text = re.sub(pattern=pattern, repl=' ',string=text)
        
        return text.strip()
    
    def for_train(self, data):
        text = data['documents']
        text = Preprocessing.remove_sequence_character(text)
        text = Preprocessing.remove_background(text)
        text = Preprocessing.remove_ja_mo(text)
        data['documents'] = text
        return data
        


class Filter:
    def __init__(self, min_document_size, max_document_size, min_summary_size, max_summary_size) :
        self.min_document_size = min_document_size
        self.max_document_size = max_document_size
        self.min_summary_size = min_summary_size
        self.max_summary_size = max_summary_size
    
    def __call__(self, data) :
        if len(data['summary']) < self.min_summary_size or len(data['summary']) > self.max_summary_size:
            return False
        
        elif len(data['documents']) < self.min_document_size or len(data['documents']) > self.max_document_size:
            return False
        
        return True
    
def preprocess_function(examples, tokenizer, data_args):
    text = examples['documents']
    targets = examples['summary']
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    
    # padding = "max_length" if data_args.pad_to_max_length else False
    texts = [bos + x + eos for x in text]
    model_inputs = tokenizer(texts, max_length=data_args.max_input_len, truncation = True)
    
    
    with tokenizer.as_target_tokenizer():
        targetss = [bos + x + eos for x in targets]
        labels = tokenizer(targetss, max_length=data_args.max_target_len, truncation=True)
        
    # 나중에 padding token loss 계산 시, ignore 하고 싶을 경우 추가로 코드 작성
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs