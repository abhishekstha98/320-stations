import torch
import networkx as nx
#import random
import numpy as np
import pandas as pd
#import os
from tqdm import tqdm


def createWeatherGraph(file_path, desired_ratio):
    """
    Creates the weighted undirected graph based on Pearson Correlation of functional weather signals.
    REPLACED: Physical Distance -> Statistical Correlation (Teleconnections).
    
    :param file_path: Path of csv file containing weather attributes.
    :param desired_ratio: Percentile control for sparsity (e.g. 0.33 means top 33% correlations kept).
    :return: edge_index, edge_weights, stations
    """
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
    stations = list(df['Location'].unique())
    
    # PIVOT: Create a matrix of Date x Station for the 'T2M' (Temperature) signal
    # This reveals which stations fluctuate together.
    # We use T2M as the primary proxy for weather pattern similarity.
    print("Calculating Pearson Correlation Matrix based on T2M signal...")
    pivot_df = df.pivot_table(index='Date', columns='Location', values='T2M')
    
    # Compute Correlation (Absolute value: -0.9 is as strong a link as +0.9)
    corr_matrix = pivot_df.corr(method='pearson').abs().fillna(0)
    
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes with metadata (preserved for reference)
    for i, station in enumerate(stations):
        # Taking the first occurrence for static metadata
        station_data = df[df['Location'] == station].iloc[0]
        alt = station_data['Altitude']
        lat = station_data['Latitude']
        long = station_data['Longitude']
        G.add_node(i, altitude=alt, latitude=lat, longitude=long, station_name=station)

    num_nodes = len(stations)
    edge_weights = []

    # Build Edges based on Correlation
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            station_i = stations[i]
            station_j = stations[j]
            
            # Lookup correlation
            if station_i in corr_matrix.index and station_j in corr_matrix.columns:
                sim = corr_matrix.loc[station_i, station_j]
            else:
                sim = 0.0
                
            edge_weights.append(sim)
            G.add_edge(i, j)

    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    # Thresholding to sparsify
    threshold = np.percentile(edge_weights.numpy(), (1 - desired_ratio) * 100)
    mask = edge_weights > threshold
    edge_weights = edge_weights[mask]

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_index = edge_index[:, mask]

    # Interchange the two rows for undirected graph
    reversed_edge_index = edge_index[[1, 0], :]

    edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)
    edge_weights = torch.tile(edge_weights, (2,))

    # Add self-loops for active nodes
    active_nodes = torch.unique(edge_index[0])
    self_loop = active_nodes.repeat(2, 1)
    
    edge_index = torch.cat([edge_index, self_loop], dim=1)
    
    # Self-loops have correlation 1.0
    edge_weights = torch.cat([edge_weights, torch.ones(self_loop.size(1))], dim=0)
    
    return edge_index, edge_weights, stations

'''
#Station wise mean and SD (OLD and head NAN in prediction due to zero value in Rainfall)
def features_dataframe(file_path, stations):
    MEAN_STD = dict()
    df_new = pd.DataFrame()
    df = pd.read_csv(file_path)
    #'QV2M'='RH2M',
    #'T2M_RANGE'='T2M_MAX'-'T2M_MIN',
    #'WS10M_RANGE' ='WS10M_MAX'-'WS10M_MIN',
    #'WS50M_RANGE' = 'WS50M_MAX'-'WS50M_MIN',
    #selected_features = ['QV2M', 'RH2M', 'PRECTOTCORR', 'T2M', 'T2MWET', 'TS', 'PS', 'WS10M', 'WS50M', 'Location']
    selected_features = ['T2M', 'T2MWET','TS','T2M_MAX','T2M_MIN','RH2M',
                       'PRECTOTCORR','PS','WS10M', 'WS10M_MAX','WS10M_MIN',
                       'WS50M','WS50M_MAX','WS50M_MIN']
    
    # target= [4, 5, 6]

    for station in stations:
        #print(station)
        df_ = df[df['Location'] == station][selected_features]
        mean = df_.select_dtypes(include=['float']).mean()
        std = df_.select_dtypes(include=['float']).std()
        #df_i = df_[selected_features].select_dtypes(include=['float']).apply(lambda x: (x - mean) / std, axis=1)

        # Assuming df_ is your DataFrame and selected_features contains numeric column names
        df_i = (df_[selected_features] - mean) / std
        df_i['Location'] = station

        MEAN_STD[f'{station}_mean'] = mean
        MEAN_STD[f'{station}_std'] = std

        df_new = pd.concat([df_new, df_i], axis=0, ignore_index=True)

    return df_new, MEAN_STD

'''
def features_dataframe(file_path, save_dir):
    df1 = pd.read_csv(file_path)
    #print(len(df1))
    df1['Date'] = pd.to_datetime(df1['Date'])
    
    # Perform boolean indexing to select data up to the specific date
    #df = df1[df1['Date'] <= '2023-12-25'] #Taking all the value until 2023-12-25
    df = df1                                #Taking all the value until Jan.29. 2024
    print(len(df))

    selected_feat = ['T2M', 'T2MWET', 'TS', 'T2M_MAX', 'T2M_MIN', 'RH2M',
            'PRECTOTCORR', 'PS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS50M', 'WS50M_MAX',
            'WS50M_MIN']
    data1 = df[selected_feat]
    print(data1.columns)
    # mean_data = data.mean()
    
    mean_values = data1.mean()
    std_values = data1.std()
    
    # Converting the Pandas Series Object into DataFrame Object.
    means = pd.DataFrame(mean_values)
    stds = pd.DataFrame(std_values)
    print(means)
    print(stds)
    # Save mean and standard deviation to data frame
    means.T.to_csv(save_dir+'mean_values.csv', index=False)
    stds.T.to_csv(save_dir+'std_values.csv', index=False)
    
    '''
    # Normalize the data
    normalized_data1 = (data1 - mean_values) / std_values
    df[selected_feat] = normalized_data1[selected_feat]
    df.to_csv(save_dir+'normalized_data.csv', index=False)
    '''


def get_features(df, stations):
    #target_features = ['QV2M', 'RH2M', 'PRECTOTCORR', 'T2M', 'T2MWET', 'TS', 'PS', 'WS10M', 'WS50M']
    target_features = ['T2M', 'T2MWET','TS','T2M_MAX','T2M_MIN','RH2M',
                       'PRECTOTCORR','PS','WS10M', 'WS10M_MAX','WS10M_MIN',
                       'WS50M','WS50M_MAX','WS50M_MIN']

    #                    0        1          2           3       4       5     6      7        8
    # Our target labels: [4, 5, 6] 

    STATIONS_SNAPSHOTS = []

    # the `pd.Categorical` function is used to convert the 'Location' column to a categorical type with a
    # custom order specified by the `custom_order` list. The `ordered=True` argument ensures that the custom
    # order is respected when performing operations like `groupby`.
    df['Location'] = pd.Categorical(df['Location'], categories=stations, ordered=True)

    grouped_df = df.groupby('Location')

    for _, group in tqdm(grouped_df):
        # Append the features for each station to the list
        snapshot = group[target_features].values.tolist()
        STATIONS_SNAPSHOTS.append(snapshot)

    return STATIONS_SNAPSHOTS



def get_stations(filename):
    with open(filename, 'r') as f:
        stations = [line.rstrip('\n') for line in f]
    return stations


if __name__ == '__main__':
    import os
    
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use the historical archive for correlation graph construction.
    # The single-date station snapshot is not sufficient for time-series correlation.
    file_path = os.path.join(base_dir, 'Weather_320_Stations_DB.csv')
    save_dir = os.path.join(base_dir, 'GNN TRAIN TEST', 'processed_data') 
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    desired_ratio = 0.33 # Keep top 33% correlations
        
    print(f"Processing file: {file_path}")
    print(f"Saving to: {save_dir}")

    # (Optional) Normalize data - uncomment if needed for training
    # features_dataframe(file_path, save_dir)
    
    # 1. Generate Correlation Graph
    print("Generating Correlation Graph...")
    edge_index, edge_weights, stations = createWeatherGraph(file_path, desired_ratio)
    
    print("Graph Stats:")
    print(" - Edge Index Shape:", edge_index.shape)
    print(" - Edge Weights Shape:", edge_weights.shape)
    print(" - Number of Stations:", len(stations))
    
    # 2. Save Graph structure
    torch.save(edge_index, os.path.join(save_dir, 'edge_index.pt'))
    torch.save(edge_weights, os.path.join(save_dir, 'edge_weights.pt'))
    
    # 3. Save Stations List
    with open(os.path.join(save_dir, 'stations.txt'), 'w') as f:
        for station in stations:
            f.write(str(station) + "\n")
    
    print("Success! Graph saved.")
