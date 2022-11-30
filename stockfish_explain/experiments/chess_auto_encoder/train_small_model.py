import matplotlib.pyplot as plt
from small_model import DeepAutoencoder
import torch
import pandas as pd
import chess
from tqdm import tqdm
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from utils import   get_FenBatchProvider, transform
from stockfish_explain.gen_concepts import create_custom_concepts
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model():
    writer = SummaryWriter()   
    model_type = 'small'
    # create unique model name for tensorboard
    model_name = f"model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"                                                             

    batch_size = 200
    train_loader = get_FenBatchProvider(batch_size=batch_size)
    val_loader = get_FenBatchProvider(batch_size=batch_size)


    # Instantiating the model and hyperparameters
    model = DeepAutoencoder(input_size=768)
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()
    num_epochs = 200
    max_iterations = 500
    learning_rate = 0.0003
    save_interval = 10

    # print model parameters
    print(f'Learing rate: {learning_rate}')


    #optimizer = torch.optim.SGD(model.parameters(), lr=100)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500*10, eta_min=10)

    model = model.cuda()

    # List that will store the training loss
    train_loss = []
    
    
    # Training loop starts
    for epoch in tqdm(range(num_epochs)):
            
        # Initializing variable for storing 
        # loss
        running_loss = 0
        
        it = 0
        # Iterating over the training dataset
        for batch in train_loader:

            if it == max_iterations:
                break
            it += 1
                
            # Loading image(s) and
            # reshaping it into a 1-d vector
            img = batch
            img = transform(img) 
            img = img.reshape(-1, model.input_size).cuda()
            
            # Generating output
            out = model(img)
            
            # Calculating loss
            loss = criterion(out, img)
            
            # Updating weights according
            # to the calculated loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Incrementing loss
            running_loss += loss.item()
            #scheduler.step()
        
        # Averaging out loss over entire batch
        running_loss /= batch_size
        train_loss.append(running_loss)

        writer.add_scalar("Loss/train", running_loss, epoch)
        #writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("LearningRate", get_lr(optimizer), epoch)

        # save model every 100 epochs
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f'./models/{model_name}_{epoch}.pt')
        
    
    
    writer.flush()
    # Plotting the training loss
    #plt.plot(range(1,num_epochs+1),train_loss)
    #plt.xlabel("Number of epochs")
    #plt.ylabel("Training Loss")
    #plt.show()

    # save model to disk
    torch.save(model.state_dict(), f'./models/{model_name}.pt')

if __name__ == '__main__':
    train_model()
