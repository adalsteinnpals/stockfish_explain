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


def train_model():
    writer = SummaryWriter()   
    model_name = 'BCE_small_3'                                                                                             

    batch_size = 200
    train_loader = get_FenBatchProvider(batch_size=batch_size)
    val_loader = get_FenBatchProvider(batch_size=batch_size)


    # Instantiating the model and hyperparameters
    model = DeepAutoencoder(input_size=768)
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()
    num_epochs = 1000
    max_iterations = 500
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500*10, eta_min=0.0001)

    model = model.cuda()

    # List that will store the training loss
    train_loss = []
    
    # Dictionary that will store the
    # different images and outputs for 
    # various epochs
    outputs = {}
    
    
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
            scheduler.step()
        
        # Averaging out loss over entire batch
        running_loss /= batch_size
        train_loss.append(running_loss)

        writer.add_scalar("Loss/train", running_loss, epoch)

        # save model every 100 epochs
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'./models/model_{model_name}_{epoch}.pt')
        
    
    
    writer.flush()
    # Plotting the training loss
    #plt.plot(range(1,num_epochs+1),train_loss)
    #plt.xlabel("Number of epochs")
    #plt.ylabel("Training Loss")
    #plt.show()

    # save model to disk
    torch.save(model.state_dict(), f'./models/model_{model_name}.pt')

if __name__ == '__main__':
    train_model()
