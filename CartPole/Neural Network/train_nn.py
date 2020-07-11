
import torch
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

from nn import NN

def train_epoch(data_loader, model, criterion, optimizer):
    print(data_loader)
    for X, labels in data_loader:
        optimizer.zero_grad()

        output = model(X)
        
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

def evaluate_episode(data, model, loss_f, epoch):
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []

    #print(training)
    for X, y in data:
        #y = torch.tensor([max(yi, 0.9) for yi in y])
        #print(y)
        output = model(X)
        #y = y.view(-1, 1)
        predicted = torch.argmax(output.data, axis=1)
        #predicted = output.data
        y_true.append(y)
        y_pred.append(predicted)
        total += y.size(0)
        #print(f'm.out: {predicted},\tlabel:{y}')
        #quit()
        correct += (predicted == y).sum().item()
        running_loss.append(loss_f(output, y).item())
            
    train_loss = np.mean(running_loss)
    train_acc = correct / total

    return train_acc, train_loss

def evaluate_epoch(training, validation, model, loss_f, epoch, stats):
    train_res = evaluate_episode(training, model, loss_f, epoch)
    train_acc, train_loss = train_res
        
    val_res = evaluate_episode(validation, model, loss_f, epoch)
    val_acc, val_loss = val_res

    #import pdb; pdb.set_trace()

    #new_stats = pd.DataFrame({'val_acc':val_acc, 'val_loss':val_loss,
     #                                                  'train_acc':train_acc, 'train_loss':train_acc},
     #                                                   index=[epoch])

    stats.loc[epoch] = [val_acc, val_loss, train_acc, train_loss]
    #stats.append([val_acc, val_loss, train_acc, train_loss])


def main():

    
    model = NN(fc_depth=1, hidden_units = 128)
    loss_f = torch.nn.CrossEntropyLoss()

    lr = 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    '''
    m, start_ep, stats = restore_checkpoint(model))
    '''
    start_epoch = 0
    stats= pd.DataFrame(columns=['val_acc', 'val_loss', 'train_acc', 'train_loss'])
    #tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
    #   num_classes=config('challenge.num_classes'))
    batch_size = 2
    x = torch.randn(3,  batch_size, 5)
    y =  torch.tensor([[0.9 for _yi in yi] for yi in torch.randn(3, batch_size, 2)]).long()
    training, validation = list(zip(x, y)), list(zip(x, y))

   #fig, axes = utils.make_cnn_training_plot(name='Challenge')
    
    evaluate_epoch(training, validation, model, loss_f, start_epoch, stats)

    num_epochs = 5
    
    # Evaluate 
    for epoch in range(start_epoch, num_epochs):
        
        train_epoch(training, model, loss_f, optimizer)
        
        evaluate_epoch(training, validation, model, loss_f, epoch+1, stats)
        
        #save_checkpoint(model,...

    print('Finished Training')

    
def load_data():
    # data loaders
    return  get_train_val_test_loaders(
                     num_classes=config('challenge.num_classes'))

def eval_model(model, training, validation, loss_f, num_epochs):
    start_time = time()    
    start_epoch, stats = 0, []


    for epoch in range(start_epoch, num_epochs):
        train_epoch(training, model, loss_f)
        evaluate_epoch(training, validation,
                                        model, loss_f, epoch+1, stats)

        # Export learned model parameters to file

        
    plot_name += f'_runtime_{round((time() - start_time)/60, 2)}_min'
    print(f'Finished Training {plot_name}')

    return 0

def graph_training(model, num_epochs,
                                   plot_name, training, validation,
                                   loss=None, optimizer=None):
    if loss == None:
        loss = torch.nn.CrossEntropyLoss()
    if optimizer == None:
        loss = optimizer = torch.optim.Adam(model.parameters(), 0.1)
    eval_model(model, training,
                         validation, optimizer,
                         num_epochs)
    
def learn_hyperparameters(num_epochs=15):
    # todo torch.optim.lr_scheduler.ReduceLROnPlateau
    #           torch.nn.BatchNorm1d
    
    tr_loader, va_loader, te_loader, _ = load_data()
    new_model = Challenge

    '''
    if True:
        kernel_1d_size = [2**n for n in range(4, 10)]#16 -->512
        for k in kernel_1d_size:
            model = new_model(hidden_units=36, kernel_1d = k)
            plot_name = f"ch_1dcnn_kernelSize_{k}_1fc_38hidden"
            graph_training(model, num_epochs,
                                    plot_name, tr_loader, va_loader)
  '''

if __name__ == '__main__':
    #learn_hyperparameters()
    main()
