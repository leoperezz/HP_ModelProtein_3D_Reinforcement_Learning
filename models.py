import torch
from torch import nn
from torch.nn import functional as F


class LSTM_HP(nn.Module):

  def __init__(self,input_size,hidden_size,num_layers,fc_units,num_actions):
    super(LSTM_HP,self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
    self.fc1 = nn.Linear(hidden_size,fc_units)
    self.fc2 = nn.Linear(fc_units,num_actions)
    
  def forward(self,x):
    
    out,_=self.lstm(x)
    out_fc1 = self.fc1(out[:,-1,:])
    out_fc2 = self.fc2(out_fc1)

    return out_fc2

  
class LSTM_HP_ATT(nn.Module):

  def __init__(self,input_size,hidden_size,fc_units,num_actions):
    super(LSTM_HP_ATT,self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size

    self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True)
    self.att = nn.MultiheadAttention(hidden_size,8,batch_first=True)
    self.lstm2 = nn.LSTM(hidden_size*2,hidden_size,batch_first = True)
    self.fc1 = nn.Linear(hidden_size,fc_units)
    self.fc2 = nn.Linear(fc_units,num_actions)
    
  def forward(self,x):
    
    out_lstm,_ = self.lstm(x)
    att_out,_ = self.att(out_lstm,out_lstm,out_lstm)
    att_out = torch.cat((out_lstm,att_out),dim=2)
    out_lstm2,_= self.lstm2(att_out)

    out_fc1 = self.fc1(out_lstm2[:,-1,:])
    out_fc2 = self.fc2(out_fc1)


    return out_fc2  

