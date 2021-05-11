import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
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

def train(training_set_path, test_set_path, output_class_num, epochs=20, learning_rates=0.1,  
    save_path = "./model_yelp_params.pkl", load_model=False):
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
    training_generator = DataLoader(training_set, **training_params)

    model = DenseNet(input_length=1024, conv_dim=40, growth_rate=32, layer=6, output_class=output_class_num)
    if load_model == True:
        model.load_state_dict(torch.load(save_path))
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    num_epochs = epochs
    num_iter_per_epoch = len(training_generator)

    loss_tr_ep = []
    accu_tr_ep = []
    loss_te_ep = []
    accu_te_ep = []
    #for epoch in tqdm(range(num_epochs)):
        #for iter, batch in tqdm(enumerate(training_generator), total=len(training_generator)):
    for epoch in range(num_epochs):
        loss_tr = []
        cnt_tr = []
        label_tr = []
        pred_tr = []
        if epoch > 10: 
            scheduler.step()
        for _, batch in tqdm(enumerate(training_generator), total=num_iter_per_epoch):
            feature, label = batch
            feature = feature.reshape(-1, 1, 1024)
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            predictions = model(feature)
            loss = criterion(predictions, label)
            label_tr.extend(label.clone().cpu())
            pred_tr.append(predictions.detach().cpu())
            loss_tr.append(loss.detach() * label.shape[0])
            cnt_tr.append(label.shape[0])
            loss.backward()
            optimizer.step()
            '''print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                    epoch + 1,
                    num_epochs,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss, training_metrics["accuracy"]))'''
        loss_tr_ep.append(sum(loss_tr)/sum(cnt_tr))
        preds = torch.cat(pred_tr, 0)
        labels = np.array(label_tr)
        metrics = get_evaluation(labels, preds, list_metrics=["accuracy"])
        accu_tr_ep.append(metrics["accuracy"])
        accu_te, loss_te = eval_test(model, test_set_path)
        loss_te_ep.append(loss_te)
        accu_te_ep.append(accu_te)
        print("Epoch: {}/{}, Lr: {}, Loss_train: {}, Accuracy_train: {}, Loss_test: {}, Accuracy_test: {}".format(
            epoch + 1,
            num_epochs,
            optimizer.param_groups[0]['lr'],
            sum(loss_tr)/sum(cnt_tr),
            metrics["accuracy"],
            loss_te,
            accu_te
        ))
    plt.plot(accu_tr_ep)
    plt.plot(accu_te_ep)
    plt.legend(["Training Accuracy", "Test Accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title('Losses of the net')
    plt.savefig("./accuracy.png")
    plt.clf()
    plt.plot(loss_tr_ep)
    plt.plot(loss_te_ep)
    plt.legend(["Training Loss", "Test Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title('Accuracy of the net')
    plt.savefig("./losses.png")
    torch.save(model.state_dict(), save_path)

def eval_test(model, test_set_path):
    test_params = {"batch_size": 128,
        "shuffle": False,
        "num_workers": 0}
    test_set = MyDataset(test_set_path, 1024)
    test_generator = DataLoader(test_set, **test_params)
    criterion = nn.CrossEntropyLoss()
    te_label_ls = []
    te_pred_ls = []
    te_losses = []
    for _, batch in tqdm(enumerate(test_generator), total=len(test_generator)):
        te_feature, te_label = batch
        te_feature = te_feature.reshape(-1, 1, 1024)
        if torch.cuda.is_available():
            te_feature = te_feature.cuda()
            te_label = te_label.cuda()
            with torch.no_grad():
                te_predictions = model(te_feature)
            loss = criterion(te_predictions, te_label)
            te_losses.append(loss*len(te_label))
            te_label_ls.extend(te_label.clone().cpu())
            te_pred_ls.append(te_predictions.clone().cpu())
    te_pred = torch.cat(te_pred_ls, 0)
    te_label = np.array(te_label_ls)
    loss = sum(te_losses)/test_set.__len__()
    test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
    print("Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
        test_metrics["accuracy"],
        test_metrics["confusion_matrix"]))
    return test_metrics["accuracy"], loss

if __name__ == "__main__":
    #train("data/yelp_review_polarity_csv/train.csv", "data/yelp_review_polarity_csv/test.csv", output_class_num=2, epochs=20, learning_rates=0.1, save_path = "./model_yelp_params.pkl", load_model=False)
    train("data/yahoo_answers_csv/train.csv", "data/yahoo_answers_csv/test.csv", output_class_num=10, epochs=20, learning_rates=0.1, save_path = "./model_yahoo_params.pkl", load_model=False)
    