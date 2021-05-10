import torch
from dataset import MyDataset
from vdcnn_model import VDCNN
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data_fpath = "data/yelp_review_polarity_csv/"

train_dataset = MyDataset(data_fpath+"train.csv")
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = MyDataset(data_fpath+"val.csv")
val_dataloader = DataLoader(val_dataset, batch_size=64)
test_dataset = MyDataset(data_fpath+"test.csv")
test_dataloader = DataLoader(test_dataset, batch_size=64)

num_classes = train_dataset.num_classes

model = VDCNN(num_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(5):
    """model.train()
    for idx, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        logits = model(x.long())
        loss = model.compute_loss(logits, y)
        loss.backward()
        optimizer.step()"""
    model.eval()
    err, tot_count = 0, 0
    for idx, (x, y) in enumerate(val_dataloader):
        logits = model(x.long())
        pred_y = torch.argmax(logits, dim=1)
        err += (pred_y != y).sum()
        tot_count += y.size(0)
    print("Val err: " + str(err/tot_count))
err, tot_count = 0, 0
for idx, (x, y) in enumerate(test_dataloader):
    logits = model(x.long())
    pred_y = torch.argmax(logits, dim=1)
    err += (pred_y != y).sum()
    tot_count += y.size(0)
print("Test err: " + str(err/tot_count))
