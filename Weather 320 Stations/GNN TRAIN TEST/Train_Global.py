import numpy as np 
#import pandas as pd 
from tqdm import tqdm 
import os
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
#from torch_geometric_temporal.signal import temporal_signal_split
#import torch.nn as nn

import torch 
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.attention.stgcn import STConv

from torch.utils.data import Dataset, DataLoader

### DataLoader
class WeatherDatasetLoader(object):
    
    def __init__(self):    
        # self._snapshots = np.load('./snapshots_transpose.npy')
        self._snapshots = np.load('D:/ICIMOD/YASH_PROCESS/snapshots.npy')
    
    def _get_edge_index(self):
        # self._edges = torch.load("./edge_index.pt")
        self._edges = torch.load('D:/ICIMOD/YASH_PROCESS/edge_index.pt')

    def _get_edge_weights(self):
        # self._edge_weights = torch.load('./edge_weights.pt').to(torch.float32)
        loaded_data = torch.load('D:/ICIMOD/YASH_PROCESS/edge_weights.pt')
        self._edge_weights = loaded_data.to(torch.float32)
        
        #self._edge_weights = torch.load('E:/ICIMOD_Project/ICIMOD PROJECT/YASH_PROCESS/edge_weights.pt'.to(torch.float32))
        #self._edge_weights = torch.load('E:/ICIMOD_Project/ICIMOD PROJECT/YASH_PROCESS/edge_weights.pt')
    
    def _get_targets_and_features(self):
        stacked_target = self._snapshots
        
        self.features = [
            np.expand_dims(stacked_target[i : i + self.lags, :, :], axis=0)
            for i in range(stacked_target.shape[0] - self.lags - self._pred_seq)
        ]
        
        self.targets = [
            # np.expand_dims(stacked_target[i + self.lags:(i + self.lags+self._pred_seq), : ,[5, 7, 8]].T, axis=0)
            np.expand_dims(np.transpose(stacked_target[i + self.lags:(i + self.lags + self._pred_seq), : ,[5, 7, 8]].T, (2, 1, 0)), axis=0)
            for i in range(stacked_target.shape[0] - self.lags - self._pred_seq)
        ]
    
    def get_dataset(self, lags: int = 4, pred_seq: int = 5) -> StaticGraphTemporalSignal:
        self.lags = lags
        self._pred_seq = pred_seq 
        self._get_edge_index()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
        
#######
## Custom Datasetloader
#######

def custom_collate(batch):
    # return Batch.from_data_list(batch)
    return batch 

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_name)

        return data

#Best Network
class STGCN_Best(torch.nn.Module):
    """
    Processes a sequence of graph data to produce a spatio-temporal embedding
    to be used for regression, classification, clustering, etc.
    prediction time seq. = Lags - 2(Kernel size -1)*STConv_Layers = 41 -2(7-1)*3 = 5 for first STconv and reduded similarly for other
    
    """
    def __init__(self):
        super(STGCN_Best, self).__init__()
        self.stconv_block1 = STConv(210, 14, 64, 64, 9, 3)       # Last is Chebyshev filter size 
        self.stconv_block2 = STConv(210, 64, 128, 128, 7, 3)       # Second last is ouputs 
        self.stconv_block3 = STConv(210, 128, 256, 512, 3, 3)
        self.stconv_block4 = STConv(210, 512, 512, 512, 3, 3)
        self.fc = torch.nn.Linear(512, 3)
        
    def forward(self, x, edge_index, edge_attr):
        temp1 = self.stconv_block1(x, edge_index, edge_attr)
        temp1 = F.relu6(temp1)
        
        temp2 = self.stconv_block2(temp1, edge_index, edge_attr)
        temp2 = F.relu6(temp2)
        
        temp3 = self.stconv_block3(temp2, edge_index, edge_attr)
        temp3 = F.relu6(temp3)
        
        temp4 = self.stconv_block4(temp3, edge_index, edge_attr)
        temp4 = F.hardswish(temp4)
        
        temp = self.fc(temp4)
        temp = F.sigmoid(temp) #mish, hardswish, relu6, rrelu, tanhshrink 
        
        return temp

''  
class STGCN_Corrected_Best(torch.nn.Module):
    """
    Processes a sequence of graph data to produce a spatio-temporal embedding
    to be used for regression, classification, clustering, etc.
    prediction time seq. = Lags - 2(Kernel size -1)*STConv_Layers = 41 -2(7-1)*3 = 5 for first STconv and reduded similarly for other
    
    """
    def __init__(self):
        super(STGCN_Corrected_Best, self).__init__()
        self.stconv_block1 = STConv(210, 14, 64, 64, 9, 3)       # Last is Chebyshev filter size 
        self.stconv_block2 = STConv(210, 64, 128, 128, 7, 3)       # Second last is ouputs 
        self.stconv_block3 = STConv(210, 128, 256, 256, 3, 3)
        self.stconv_block4 = STConv(210, 256, 512, 512, 3, 3)
        self.fc = torch.nn.Linear(512, 3)
        
    def forward(self, x, edge_index, edge_attr):
        temp1 = self.stconv_block1(x, edge_index, edge_attr)
        temp1 = F.relu6(temp1)
        
        temp2 = self.stconv_block2(temp1, edge_index, edge_attr)
        temp2 = F.relu6(temp2)
        
        temp3 = self.stconv_block3(temp2, edge_index, edge_attr)
        temp3 = F.relu6(temp3)
        
        temp4 = self.stconv_block4(temp3, edge_index, edge_attr)
        temp4 = F.relu6(temp4)
        
        temp = self.fc(temp4)
        temp = F.sigmoid(temp) #mish, hardswish, relu6, rrelu, tanhshrink 
        
        return temp


class STGCN(torch.nn.Module):
    """
    Processes a sequence of graph data to produce a spatio-temporal embedding
    to be used for regression, classification, clustering, etc.
    prediction time seq. = Lags - 2(Kernel size -1)*STConv_Layers = 41 -2(7-1)*3 = 5 for first STconv and reduded similarly for other
    
    """
    def __init__(self):
        super(STGCN, self).__init__()
        self.stconv_block1 = STConv(210, 14, 64, 64, 9, 3)       # Last is Chebyshev filter size 
        self.stconv_block2 = STConv(210, 64, 64, 64, 7, 3)       # Second last is ouputs 
        self.stconv_block3 = STConv(210, 64, 64, 64, 3, 3)
        self.stconv_block4 = STConv(210, 128, 256, 256, 3, 3)
        self.fc = torch.nn.Linear(256, 3)
        
    def forward(self, x, edge_index, edge_attr):
        temp1 = self.stconv_block1(x, edge_index, edge_attr)
        #temp1 = F.relu6(temp1)
        
        temp2 = self.stconv_block2(temp1, edge_index, edge_attr)
        #temp2 = F.relu6(temp2)
        
        temp3 = self.stconv_block3(temp2, edge_index, edge_attr)
        #temp3 = F.relu6(temp3)
        
        temp4 = self.stconv_block4(temp3, edge_index, edge_attr)
        #temp4 = F.hardswish(temp4)
        
        temp = self.fc(temp4)
        #temp = F.sigmoid(temp) #mish, hardswish, relu6, rrelu, tanhshrink 
        
        return temp

class STGCN_Parallel(torch.nn.Module):
    """
    Processes a sequence of graph data to produce a spatio-temporal embedding
    to be used for regression, classification, clustering, etc.
    prediction time seq. = Lags - 2(Kernel size -1)*STConv_Layers = 41 -2(7-1)*3 = 5 for first STconv and reduded similarly for other
    
    """
    def __init__(self):
        super(STGCN_Parallel, self).__init__()
        
        self.stconv_block11 = STConv(320, 14, 64, 64, 9, 3)       # Last is Chebyshev filter size 
        self.stconv_block12 = STConv(320, 64, 128, 128, 7, 3)       # Second last is ouputs 
        self.stconv_block13 = STConv(320, 128, 256, 256, 3, 3)
        self.stconv_block14 = STConv(320, 256, 512, 512, 3, 3)
        #self.fc11 = torch.nn.Linear(512, 256)
        
        self.stconv_block21 = STConv(320, 14, 64, 64, 9, 3)       # Last is Chebyshev filter size 
        self.stconv_block22 = STConv(320, 64, 128, 128, 7, 3)       # Second last is ouputs 
        self.stconv_block23 = STConv(320, 128, 256, 256, 3, 3)
        self.stconv_block24 = STConv(320, 256, 512, 512, 3, 3)
        #self.fc21 = torch.nn.Linear(512, 256)
        
        self.fc = torch.nn.Linear(512, 3)
        
        
    def forward(self, x, edge_index, edge_attr):
        temp11 = self.stconv_block11(x, edge_index, edge_attr)
        temp11 = F.relu6(temp11)
        temp12 = self.stconv_block12(temp11, edge_index, edge_attr)
        temp12 = F.relu6(temp12)
        temp13 = self.stconv_block13(temp12, edge_index, edge_attr)
        temp13 = F.relu6(temp13)
        temp14 = self.stconv_block14(temp13, edge_index, edge_attr)
        #temp14 = F.relu6(temp14)
        #Fc_temp11 = self.fc11(temp14)
        
        temp21 = self.stconv_block11(x, edge_index, edge_attr)
        temp21 = F.relu6(temp21)
        temp22 = self.stconv_block22(temp21, edge_index, edge_attr)
        temp22 = F.relu6(temp22)
        temp23 = self.stconv_block23(temp22, edge_index, edge_attr)
        temp23 = F.relu6(temp23)
        temp24 = self.stconv_block24(temp23, edge_index, edge_attr)
        #temp24 = F.relu6(temp24)
        
        #Fc_temp21 = self.fc11(temp24)
        
        #temp = Fc_temp11 + Fc_temp21
        #print(Fc_temp11.shape, Fc_temp21.shape, temp.shape)
        
        temp = self.fc(temp14+temp24)
        temp = F.sigmoid(temp) #mish, hardswish, relu6, rrelu, tanhshrink 
        
        return temp


class STGCN_Best_BRC(torch.nn.Module):
    def __init__(self):
        super(STGCN_Best_BRC, self).__init__()
        self.stconv_block1 = STConv(320, 14, 64, 128, 9, 4)
        self.stconv_block2 = STConv(320, 128, 256, 64, 7, 4)
        self.stconv_block3 = STConv(320, 64, 32, 16, 5, 3)
        self.fc = torch.nn.Linear(16, 3)
        
    def forward(self, x, edge_index, edge_attr):
        temp = self.stconv_block1(x, edge_index, edge_attr)
        temp = self.stconv_block2(temp, edge_index, edge_attr)
        temp = self.stconv_block3(temp, edge_index, edge_attr)
        temp = self.fc(temp)
        
        return temp


class STGCN_Parallel_UP(torch.nn.Module):
    """
    Processes a sequence of graph data to produce a spatio-temporal embedding
    to be used for regression, classification, clustering, etc.
    prediction time seq. = Lags - 2(Kernel size -1)*STConv_Layers = 41 -2(7-1)*3 = 5 for first STconv and reduded similarly for other
    
    """
    def __init__(self):
        super(STGCN_Parallel_UP, self).__init__()
        
        self.stconv_block11 = STConv(317, 14, 32, 64, 9, 4)       # Last is Chebyshev filter size 
        self.stconv_block12 = STConv(317, 128, 256, 512, 7, 4)       # Second last is ouputs 
        self.stconv_block13 = STConv(317, 512, 256, 128, 5, 3)
        #self.stconv_block14 = STConv(210, 128, 64, 32, 3, 3)
        #self.fc11 = torch.nn.Linear(512, 256)
        '''
        self.stconv_block21 = STConv(210, 14, 64, 128, 9, 4)       # Last is Chebyshev filter size 
        self.stconv_block22 = STConv(210, 128, 256, 512, 7, 4)       # Second last is ouputs 
        self.stconv_block23 = STConv(210, 512, 256, 128, 3, 3)
        self.stconv_block24 = STConv(210, 128, 64, 32, 3, 3)
        #self.fc21 = torch.nn.Linear(512, 256)
        '''
        self.fc = torch.nn.Linear(32, 3)
        
        
    def forward(self, x, edge_index, edge_attr):
        temp11 = self.stconv_block11(x, edge_index, edge_attr)
        #temp11 = F.relu6(temp11)
        temp12 = self.stconv_block12(temp11, edge_index, edge_attr)
        #temp12 = F.relu6(temp12)
        temp13 = self.stconv_block13(temp12, edge_index, edge_attr)
        #temp13 = F.relu6(temp13)
        temp14 = self.stconv_block14(temp13, edge_index, edge_attr)
        #temp14 = F.relu6(temp14)
        #Fc_temp11 = self.fc11(temp14)
        '''
        temp21 = self.stconv_block11(x, edge_index, edge_attr)
        temp21 = F.relu6(temp21)
        temp22 = self.stconv_block22(temp21, edge_index, edge_attr)
        temp22 = F.relu6(temp22)
        temp23 = self.stconv_block23(temp22, edge_index, edge_attr)
        temp23 = F.relu6(temp23)
        temp24 = self.stconv_block24(temp23, edge_index, edge_attr)
        #temp24 = F.relu6(temp24)
        
        #Fc_temp21 = self.fc11(temp24)
        
        #temp = Fc_temp11 + Fc_temp21
        #print(Fc_temp11.shape, Fc_temp21.shape, temp.shape)
        temp = self.fc(temp14+temp24)
        '''
        temp = self.fc(temp14)
        #temp = F.sigmoid(temp) #mish, hardswish, relu6, rrelu, tanhshrink 
        
        return temp


   
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
   
if __name__ == '__main__':
    
    '''
    #Step-1: Data processing (For Batch)
    loader = WeatherDatasetLoader()
    #dataset = loader.get_dataset(lags=41)  #41 sample in a temporal line
    dataset = loader.get_dataset(lags=43, pred_seq=7) # 'pred_seq' is the number of output seq.  

    Save_data_dir = "D:/ICIMOD/YASH_PROCESS/Data_with_Time_Seq_Seven"
    create_folder(Save_data_dir)
    # 
    for i, data in tqdm(enumerate(dataset)):
        print(i)
        #print("The data shape is:", data)
        
        torch.save(data, os.path.join(Save_data_dir, f'data_{i}.pt'))
    '''
    
    #######
    # Step-2: Training starts here
    #######
    
    Output_Save_path = "D:/ICIMOD/YASH_PROCESS/Outcome_320Stations_43Lags_7Seq_STGCN_Best_Adam/"
    create_folder(Output_Save_path)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # Move the model to GPU
    #model = STGCN_Parallel().to(device)
    model = STGCN_Best_BRC().to(device)
    
    print(model)
    '''
    #Load pre-trained weight
    weights_path = "D:/ICIMOD/YASH_PROCESS_320_LAST/Outcome_320Stations_43Lags_7Seq_Adam/Model_PreTrained.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("The pre-trained weight loaded !")
    '''
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05) #ASGD
    #optimizer = torch.optim.ASGD(model.parameters(), lr=0.005)
    
    # Define training and validation loss lists
    train_losses = []
    val_losses = []
    
    # Create DataLoader with custom collate function
    batch_size = 32
    num_workers = 2
    #data_dir = 'D:/ICIMOD/YASH_PROCESS/Data_with_Time_Seq_MINI'
    data_dir ="D:/ICIMOD/YASH_PROCESS/Data_with_Time_Seq_Seven"
    dataset = CustomDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)
    
    ######                                  
    # Splitting the dataset into training, validation and test set
    ######
    from torch.utils.data import random_split
    
    # Calculate the sizes for train, val, and test sets
    total_size = len(dataset)
    train_size = int(0.85 * total_size)  # 80% for training
    #val_size = int(0.1 * total_size)    # 10% for validation
    #test_size = total_size - train_size - val_size  # Remaining for testing
    val_size = total_size - train_size
    
    # Use random_split to create train, val, and test datasets
    #train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoader instances for train, val, and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=num_workers)
    
    # start the timer
    from timeit import default_timer as timer
    start_time = timer()
    
    print("Training started ... ")
    
    # Training loop
    num_epochs = 300
    train_losses = []
    val_losses = []
    
    # Define variables for early stopping and best validation accuracy
    best_MSE = 10.0
    patience = 15  # Number of epochs to wait for improvement
    counter = 0  # Counter to track epochs without improvement
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0
        for batch, train_batch in enumerate(train_dataloader):
            # model.to(device)
            cost = 0
            for time, data in enumerate(train_batch):
                data.to(device)  # Move the data to GPU
                y_hat = model(data.x, data.edge_index, data.edge_attr)
                
                #print("The data shape is:", data.x.shape)
                #print("The label shape is:",y_hat.shape, data.y.shape) #Output: torch.Size([1, 92, 210, 3]) torch.Size([1, 210, 3]
                cost = cost + torch.mean((y_hat - data.y) ** 2)
    
                del data
    
            cost = cost / (time + 1)
            total_train_loss += cost.item() 
    
            # Backward pass and optimization
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            del train_batch 
    
        total_train_loss /= (batch + 1)
        train_losses.append(total_train_loss)
        # Validation loop for `val_set`
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch, val_batch in enumerate(val_dataloader):
                val_cost = 0
                for time, snapshot in enumerate(val_batch):
                    snapshot.to(device)
                    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                    val_cost = val_cost + torch.mean((y_hat - snapshot.y) ** 2)
                    
                    del snapshot
                
                
                val_cost = val_cost / (time + 1)
                del val_batch
                
                
                total_val_loss += val_cost.item() 
            
            total_val_loss = total_val_loss / (batch + 1)
            val_losses.append(total_val_loss)
            
            
            # Check if the current validation accuracy is better than the previous best accuracy
            if total_val_loss < best_MSE:
                best_MSE = total_val_loss
                counter = 0  # Reset the counter since there is improvement
                # Save the model checkpoint with the best validation accuracy
                torch.save(model.state_dict(), Output_Save_path+'Model_43Lags_STConv_Best.pt')
                print(f'Saved Best Model @ Epoch: {epoch:03d}')
        
            else:
                counter += 1
    
            # Check if early stopping condition is met
            if counter >= patience:
                print("Early stopping! No improvement for {} epochs.".format(patience))
                break
        
        # Adjust learning rate if necessary
        if counter == patience // 2:
            new_lr = optimizer.param_groups[0]['lr'] * 0.1  # Reduce learning rate by a factor of 10
            print("Reducing learning rate to:", new_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # Print the average loss for the epoch
        print(f'Epoch {(epoch + 1)}/{num_epochs}, Train_Loss: {total_train_loss}, Val_Loss: {total_val_loss}')
    
    
    end_time = timer()
    print(f"Total training time: {end_time-start_time:3f} seconds")
    
    # Save model weight 
    torch.save(model.state_dict(), Output_Save_path+"Model_100Lags_STConv_Last.pth")
    
    
    ## Plotting
    import matplotlib.pyplot as plt 
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    import time 
    timestamp = time.strftime("%Y%m%d%H%M%S")
    # Saving the learning curve
    plt.savefig(Output_Save_path+'learning curve_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
 