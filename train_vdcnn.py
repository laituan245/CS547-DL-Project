import torch
from dataset import MyDataset
from vdcnn_model import VDCNN
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data_fpath = "data/yelp_review_polarity_csv/train.csv"

train_dataset = MyDataset(data_fpath)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = MyDataset(data_fpath)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataset = MyDataset(data_fpath)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = VDCNN()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(5):
    model.train()
    for idx, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(x.long())
        print(outputs.size())
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    model.eval()
    for idx, (x, y) in enumerate(val_dataloader):
        outputs = model(x)
for idx, (x, y) in enumerate(test_dataloader):
    outputs = model(x, y)
