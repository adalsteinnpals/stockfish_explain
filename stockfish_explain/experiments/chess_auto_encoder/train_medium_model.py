import matplotlib.pyplot as plt
from model_medium import DeepAutoencoder
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

def train_model():
    writer = SummaryWriter()   
    model_type = 'medium'                                                                                             

    batch_size = 200
    train_loader = get_FenBatchProvider(batch_size=batch_size)
    val_loader = get_FenBatchProvider(batch_size=batch_size)
    
    model_name = f"model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"                                                             


    # Instantiating the model and hyperparameters
    model = DeepAutoencoder(input_size=768)
    criterion = torch.nn.BCELoss()
    num_epochs = 200
    max_iterations = 500
    save_interval = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


    print('starting...')
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
        
        # Averaging out loss over entire batch
        running_loss /= batch_size
        train_loss.append(running_loss)

        writer.add_scalar("Loss/train", running_loss, epoch)

        # save model every 100 epochs
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f'./models/{model_name}_{epoch}.pt')
        
        # Storing useful images and
        # reconstructed outputs for the last batch
    
    
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
