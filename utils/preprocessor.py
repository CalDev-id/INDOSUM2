import json
import os
import sys
import torch

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from indobenchmark import IndoNLGTokenizer
from tqdm import tqdm

import lightning as L

import pandas as pd

class Preprocessor(L.LightningDataModule):
    def __init__(self, max_length, batch_size, lm_model = None):
        self.indosum_dir = "indosum"

        if lm_model:
            #selain indobart
            self.tokenizer = AutoTokenizer.from_pretrained(lm_model)
        else:
            #indobart
            self.tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")

        self.max_length = max_length
        self.batch_size = batch_size
    
    def join_paragraphs(self, paragraphs):
        # Join List of paragraphs to string paragraphs
        string_paragraph = ""
        for parag in paragraphs:
            for kal in parag:
                kalimat = " ".join(kal)
                string_paragraph += kalimat 
        
        return string_paragraph
            

    def join_summary(self, summaries):
        # Join List of paragraphs to string paragraphs
        string_summary = ""
        for sumr in summaries:
            kal_sum = " ".join(sumr)
            string_summary += kal_sum
            
        return string_summary
    def load_data(self, flag):
        list_files = os.listdir(self.indosum_dir)
        datasets = []
        for fl in list_files:
            if flag in fl:
                with open(f"{self.indosum_dir}/{fl}", "r", encoding = "utf-8") as json_reader:
                    # load file jsonl (jsonl = kumpulan file json format di gabung jadi satu file)
                    
                    data_raw = json_reader.readlines()
                    # print(data_raw[0])                   
                    # json_raw = [json.loads(jline) for jline in json_reader.readlines().rstrip().splitlines()]
                
                json_raw = []  
                for dd in data_raw:
                    
                    data_line = json.loads(dd)
                    paragraphs = self.join_paragraphs(data_line["paragraphs"])
                    summary = self.join_paragraphs(data_line["summary"])
                    
                    data = {
                        "id": data_line["id"],
                        "paragraphs": paragraphs,
                        "summary": summary,
                    }
                    
                    json_raw.append(data)
                
                datasets += json_raw
        
                # print(datasets)
        return datasets
    
    def list2tensor(self, data):
        # Convert list to tensor
        x_ids, x_att, y_ids, y_att = [],[],[],[]
        for i_d, d in tqdm(enumerate(data)):
            x_tok = self.tokenizer(
                d["paragraphs"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )
            y_tok = self.tokenizer(
                d["summary"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )
            x_ids.append(x_tok["input_ids"])
            x_att.append(x_tok["attention_mask"])
            y_ids.append(y_tok["input_ids"])
            y_att.append(y_tok["attention_mask"])

            if i_d > 100:
                break
            # print("x token keys")
            # print(x_tok.keys())
            # sys.exit()
            print(type(x_ids))
            # print(x_ids.shape)

            x_ids = torch.tensor(x_ids)
            x_att = torch.tensor(x_att)

            print(type(x_ids))
            print(x_ids)
            print(x_ids.shape)

            y_ids = torch.tensor(y_ids)
            y_att = torch.tensor(y_att)

            return TensorDataset(x_ids, x_att, y_ids, y_att)

    def preprocessor(self):
        #raw data masi format list
        raw_train_data = self.load_data(flag = "train")
        raw_test_data = self.load_data(flag = "test")
        raw_val_data = self.load_data(flag = "dev")
        
        train_data = self.list2tensor(data = raw_train_data)
        val_data = self.list2tensor(data = raw_val_data)
        test_data = self.list2tensor(data = raw_test_data)

        return train_data, val_data, test_data
    
    def setup(self, stage = None):
        train_data, val_data, test_data = self.preprocessor()
        if stage == "fit":
            self.train_data = train_data
            self.val_data = val_data
        elif stage == "test":
            self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=3
        )
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=3
        )
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=3
        )
        
if __name__ == "__main__":
    pre = Preprocessor(max_length=512, batch_size=5)
    pre.setup(stage="fit")
    for data in pre.train_dataloader():
        print(len(data))
        sys.exit()