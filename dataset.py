import csv
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path = "data/yelp_review_polarity_csv/train.csv", max_length=1024):
        self.data_path = data_path
        self.vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        texts, labels = [], []
        with open(data_path, encoding='utf8') as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for _, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx
                    text += " "
                label = int(float(line[0])) - 1
                texts.append(text.lower())
                labels.append(label)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = [self.vocabulary.index(i) + 1 for i in list(raw_text) if i in self.vocabulary]
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif len(data) < self.max_length:
            data += [0] * (self.max_length - len(data))
        label = self.labels[index]
        return np.array(data, dtype=np.float32), label
