import json
import os
import sys

import pandas as pd

class Preprocessor():
    def __init__(self):
        self.indosum_dir = "indosum"
    
    def join_paragraphs(self, paragraphs):
        # Join List of paragraphs to string paragraphs
        string_paragraph = ""
        for parag in paragraphs:
            for kal in parag:
                kalimat = " ".join(kal)
                string_paragraph += kalimat 
        
        return string_paragraph
            
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
                    # summary = self.join_paragraphs(data_line["summary"])
                    
                    data = {
                        "id": data_line["id"],
                        "paragraphs": paragraphs,
                        # "summary": summary,
                    }
                    
    
                    json_raw.append(data)
                
                datasets += json_raw
        
        
        return datasets
                    
        # print(datasets)
        
if __name__ == "__main__":
    pre = Preprocessor()
    pre.load_data(flag = "train")