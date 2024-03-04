import json
import os

class Preprocessor():
    def __init__(self):
        self.indosum = 'indosum'

    def join_paragraph(self, paragraphs):
        string_paragraph = ""
        for parag in paragraphs:
            for kal in parag:
                print(kal)
                kalimat = " ".join(kal)
                string_paragraph += kalimat
                
        print(string_paragraph)
        return string_paragraph

    def load_data(self, flag):
        list_files = os.listdir(self.indosum)
        datasets = []
        # print(list_files)
        for fl in list_files:
            if flag in fl:
                with open(f"{self.indosum}/{fl}", 'r', encoding="utf-8") as json_reader:

                    data_raw = [json.loads(jline) for jline in json_reader.readlines()[0].rstrip().splitlines()]
                    # lihat data
                    # print(len(data_raw[0]))
                    # print(data_raw[0])
                    
                    paragraph = self.join_paragraph(data_raw[0]['paragraphs'])
                    summary = self.join_paragraph(data_raw[0]['summary'])
                    

                    data = {
                        "id": data_raw[0]['id'],
                        "paragraphs": paragraph,
                        "summary": summary,
                    }

                    # datasets.append(data)
                    datasets += data

        print(len(datasets))
        return datasets

if __name__ == '__main__':
    pre = Preprocessor()
    pre.load_data(flag="train")
