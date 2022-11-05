import pandas as pd 
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
def df_timestamps(df, time_p):
    """
    Take dataframe and return another data frame with time stamps, date, and true target
    """
    columns = ['Date', 'True_Target']
    for i in range(time_p):
        columns.append(f'Target{i+1}')
    new_data = pd.DataFrame(columns= columns)
    for i in tqdm(range(len(df['Close']))):
        if i == len(df['Close']) - time_p:
            break
        new_row = {}
        new_row['Date'] = df.axes[0].tolist()[i]
        new_row['True_Target'] = df.iloc[i+time_p]['Close']
        for j in range(time_p):
            new_row[f'Target{j+1}'] = df['Close'][i:time_p+i][j]
        new_data = new_data.append(new_row, ignore_index = True)

    columns = ["Date","True_Target"]
    for i in range(time_p):
        columns.insert(i+1, f"Target{i+1}")
    new_data = new_data[columns]
    new_name = ['Date','True_Target']
    j = time_p
    for i in range(time_p):
        new_name.insert(i+1,f'-t{j}')
        j = j - 1

    new_data = new_data.set_axis(new_name, axis = 1, inplace = False)
    return new_data



def dataset_c(X, y):
    """
    Take dataset and convert it to tuple inside array to use it in pytorch
    """
    dataset = []
    for i in range(len(X)):
        dataset.append((X[i], y[i]))
    
    return dataset


def pred_plot(model,time_p, stock, X, scaler, plot):
    S = []
    for i in X:
        A = model(torch.tensor(i.reshape(1,time_p,1)).float().cuda())
        S.append(float(scaler.inverse_transform(A.detach().cpu().reshape(-1,1))))
    if plot == True:
        plt.plot(stock.axes[0].tolist()[time_p:], stock['Close'].values[time_p:])
        plt.plot(stock.axes[0].tolist()[time_p:], S)
        plt.legend(['real closed value', 'predicted closed value'])
        plt.xlabel("Year")
        plt.ylabel("Closed Value")
    return S, stock['Close'].values[time_p:]
    
def pred_new(model, df_stock, time_p, scaler):
    """
    Predicts new data points
    """
    last_n = torch.tensor(df_stock['Close'][-time_p:].values.reshape(1,time_p,1)).float().cuda()
    return scaler.inverse_transform(model(last_n).detach().cpu().reshape(-1,1))