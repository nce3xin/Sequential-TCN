import argparse
from model import TemporalConvNet
from dataloader import data_loader
import torch.optim as optim
import torch.nn as nn

def parse_args():
    # run with cuda: python setup.py --cuda
    # run without cuda: python setup.py
    # run with permuted MNIST: python setup.py --permuted
    # run without permuted MNIST: python setup.py
    parser=argparse.ArgumentParser(description='Sequential TCN')
    parser.add_argument('-b','--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('-c','--cuda',action='store_true',help='use cuda or not')
    parser.add_argument('-d','--dropout',type=float,default=0.3,help='dropout value')
    parser.add_argument('-g','--gradient_clip',type=float,default=0.5,help='gradient clipping')
    parser.add_argument('-e','--epoch',type=int,default=5,help='epochs')
    parser.add_argument('-k','--kernel_size',type=int,default=5,help='kernel size')
    parser.add_argument('-l','--n_layers',type=int,default=8,help='# of layers')
    parser.add_argument('-L','--log_interval',type=int,default=100,help='log interval')
    parser.add_argument('-i','--initial_lr',type=float,default=2e-3,help='initial learning rate')
    parser.add_argument('-o','--optimizer',type=str,default='Adam',help='optimizer')
    parser.add_argument('-n','--n_hidden',type=int,default=20,help='# of hidden units per layer')
    parser.add_argument('-r','--seed',type=int,default=42,help='random seed')
    parser.add_argument('-p','--permuted',action='store_true',help='permuted or not')
    args=parser.parse_args()
    return args

def create_model():
    model=TemporalConvNet()
    return model

def get_optimizer(opt_name,lr,model):
    opt=None
    if opt_name=='Adam':
        opt=optim.Adam(model.parameters(),lr=lr)
    return opt

def get_criterion():
    criterion=nn.NLLLoss()
    return criterion

def train(args,model,train_loader,optimizer,criterion):
    for batch_idx,data in enumerate(train_loader):
        inputs,targets=data
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('some info.')

def test(test_loader):
    pass

if __name__=='__main__':
    args=parse_args()
    lr=args.initial_lr
    model=create_model()
    train_loader,test_loader=data_loader(args.bs)
    optimizer=get_optimizer(args.opt_name,lr,model)
    criterion=get_criterion()
    for i in range(1,args.epoch+1):
        train(args,model,train_loader,optimizer,criterion)
        test(test_loader)
        # learning rate annealing
        if i%5==0:
            lr/=10
            for g in optimizer.param_groups:
                g['lr']=lr
