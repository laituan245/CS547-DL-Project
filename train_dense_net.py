import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
#from tqdm import tqdm
from dataset import MyDataset
from dense_net_model import DenseNet

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def train(training_set_path, test_set_path, output_class_num, 
    save_path = "./model_yelp_params.pkl"):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    training_params = {"batch_size": 128,
                       "shuffle": True,
                       "num_workers": 0}
    test_params = {"batch_size": 128,
                   "shuffle": False,
                   "num_workers": 0}

    training_set = MyDataset(training_set_path, 1024)
    test_set = MyDataset(test_set_path, 1024)
    training_generator = DataLoader(training_set, **training_params)
    test_generator = DataLoader(test_set, **test_params)

    model = DenseNet(input_length=1024, conv_dim=12, growth_rate=32, layer=5, output_class=output_class_num)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    num_epochs = 20
    num_iter_per_epoch = len(training_generator)

    #for epoch in tqdm(range(num_epochs)):
        #for iter, batch in tqdm(enumerate(training_generator), total=len(training_generator)):
    for epoch in range(num_epochs):
        for iter, batch in enumerate(training_generator):
            feature, label = batch
            feature = feature.reshape(-1, 1, 1024)
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            predictions = model(feature)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()

            
            if (iter%100 == 0):
                training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                    epoch + 1,
                    num_epochs,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss, training_metrics["accuracy"]))

    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train("data/yelp_review_polarity_csv/train.csv", "data/yelp_review_polarity_csv/test.csv", output_class_num=2, save_path = "./model_yelp_params.pkl")
    #train("data/yahoo_answers_csv/train.csv", "data/yahoo_answers_csv/test.csv", output_class_num=10, save_path = "./model_yahoo_params.pkl")