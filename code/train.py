from torch import nn
import torch
from hcts import Ts_Train, Ts_Dataloader
from models import Transformer, CNN_Transformer, LSTM, CNN_LSTM
from utlis import set_random_seed


def f(df):
    return (df-df.mean())/(df.std()+1e-8)


set_random_seed(3407)

train_loader = Ts_Dataloader(path='../train/usd_sentiment.csv', input_time_steps=9, output_time_steps=1,
                             output_idx=3, transform=f, batch_size=128, shuffle=False)


validation_loader = Ts_Dataloader(path='../validation/usd_sentiment.csv', input_time_steps=9, output_time_steps=1,
                             output_idx=3, transform=f, batch_size=128, shuffle=False)

trans_sentiment = CNN_Transformer(stride=(3,),  kernel_size=(3,), paddding=(0,), input_size=7, hidden_size=256,
                                  Transformer_layers=3, n_heads=4, time_step=9, dropout=0.3)
Ts_Train(train_dataloader=train_loader, validation_dataloader=validation_loader, log_process=True, is_show=True,
         learning_rate=0.001, model=trans_sentiment, num_epochs=1000)

train_loader = Ts_Dataloader(path='../train/USD.csv', input_time_steps=9, output_time_steps=1,
                             output_idx=0, transform=f, batch_size=128, shuffle=False)


validation_loader = Ts_Dataloader(path='../validation/USD.csv', input_time_steps=9, output_time_steps=1,
                             output_idx=0, transform=f, batch_size=128, shuffle=False)
trans = Transformer(input_size=4, hidden_size=256, num_layers=3, n_heads=4, time_step=9, dropout=0.3)
Ts_Train(train_dataloader=train_loader, validation_dataloader=validation_loader, log_process=True, is_show=True,
         learning_rate=0.001, model=trans, num_epochs=1000)