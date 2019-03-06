import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
    
class TCN(nn.Module):
    def __init__(self,num_channels,in_channels,kernel_size,dropout,out_features):
        super(TCN,self).__init__()
        self.tcn=TemporalConvNet(num_channels,in_channels,kernel_size,dropout)
        self.linear=nn.Linear(num_channels[-1],out_features)
    
    def forward(self, x):
        # inputs should have shape: (N,C_in,L),
        # where N is a batch size, C denotes a number of channels, L is a length of signal sequence.
        # and output's shape: (N,C_out,L_out)
        y=self.tcn(x)
        o=self.linear(y[:,:,-1])
        '''
        Before linear layer, why discard others?
        Here, the original author explained in issues section why just y[:,:,-1] is ok:
        You can certainly include the others (for instance, take the last 5 outputs and 
        flatten them to a vector). Since y1[:, :, -1] has a receptive field that covers
        all the input pixels in the sequence, we think it is sufficient to just use this
        last output (which is what RNNs do on this task).
        '''
        return F.log_softmax(o,dim=1)

class TemporalConvNet(nn.Module):
    def __init__(self,num_channels,in_channels,kernel_size,dropout):
        super(TemporalConvNet,self).__init__()
        layers=[]
        num_layers=len(num_channels)
        for i in range(num_layers):
            dilation_size=2**i
            channels_in=in_channels if i==0 else num_channels[i-1]
            channels_out=num_channels[i]
            layers+=[TemporalConvBlock(channels_in,channels_out,kernel_size,
                                stride=1,padding=(kernel_size-1)*dilation_size,
                                dilation=dilation_size,dropout=dropout)]

        self.net=nn.Sequential(*layers)
    
    def forward(self,x):
        return self.net(x)

class Chomp1d(nn.Module):
    # chomping padding
    # padding both sides, and then cut off the right padding("chomping")
    # or we can only pad the left side, then the "chomping" layer is actually not necessary.
    def __init__(self,chomp_size):
        super(Chomp1d,self).__init__()
        self.chomp_size=chomp_size
    
    def forward(self, x):
        return x[:,:,:-self.chomp_size]

class TemporalConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation,dropout):
        super(TemporalConvBlock,self).__init__()
        self.conv1=weight_norm(nn.Conv1d(in_channels,out_channels,
                    kernel_size,stride,padding,dilation))
        self.chomp1=Chomp1d(padding)
        self.relu1=nn.ReLU()
        self.dropout1=nn.Dropout(dropout)

        self.conv2=weight_norm(nn.Conv1d(out_channels,out_channels,
                    kernel_size,stride,padding,dilation))
        self.chomp2=Chomp1d(padding)
        self.relu2=nn.ReLU()
        self.dropout2=nn.Dropout(dropout)
    
        self.net=nn.Sequential(self.conv1,self.chomp1,self.relu1,self.dropout1,
                                self.conv2,self.chomp2,self.relu2,self.dropout2)
        self.size_matching=nn.Conv1d(in_channels,out_channels,1) if in_channels != out_channels else None
        self.relu=nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0,0.01)
        self.conv2.weight.data.normal_(0,0.01)
        if self.size_matching:
            self.size_matching.weight.data.normal_(0,0.001)
        
    def forward(self,x):
        outputs=self.net(x)
        residual=x if self.size_matching is None else self.size_matching(x)
        return self.relu(outputs+residual)
