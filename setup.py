import argparse
from model import TCN
from dataloader import data_loader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def parse_args():
    # run with cuda: python setup.py --cuda
    # run without cuda: python setup.py
    # run with permuted MNIST: python setup.py --permuted
    # run without permuted MNIST: python setup.py
    parser=argparse.ArgumentParser(description='Sequential TCN')
    parser.add_argument('-b','--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('-c','--cuda',action='store_true',help='use cuda or not')
    parser.add_argument('-d','--dropout',type=float,default=0.05,help='dropout value')
    parser.add_argument('-g','--gradient_clip',type=float,default=-1,help='gradient clipping (default: -1), -1 means no clipping')
    parser.add_argument('-e','--epoch',type=int,default=20,help='epochs')
    parser.add_argument('-k','--kernel_size',type=int,default=7,help='kernel size')
    parser.add_argument('-l','--n_layers',type=int,default=8,help='# of layers')
    parser.add_argument('-L','--log_interval',type=int,default=5,help='log interval, the number of batches between two output statements')
    parser.add_argument('-i','--initial_lr',type=float,default=2e-3,help='initial learning rate')
    parser.add_argument('-o','--optimizer',type=str,default='Adam',help='optimizer')
    parser.add_argument('-n','--n_hidden',type=int,default=25,help='# of hidden units per layer')
    parser.add_argument('-r','--seed',type=int,default=1111,help='random seed')
    parser.add_argument('-p','--permute',action='store_true',help='permuted or not')
    args=parser.parse_args()
    return args

def create_model(num_channels,in_channels,kernel_size,dropout,out_features):
    model=TCN(num_channels,in_channels,kernel_size,dropout,out_features)
    return model

def get_optimizer(opt_name,lr,model):
    opt=None
    if opt_name=='Adam':
        opt=optim.Adam(model.parameters(),lr=lr)
    return opt

def get_criterion():
    '''
    Remind that we use F.log_softmax() as the last layer of our model,
    The NLLLoss is a perfect match for log_softmax(). 
    Check https://pytorch.org/docs/stable/nn.html?highlight=nllloss#torch.nn.NLLLoss
    NLLLoss() input shape: (N,C) where C=number of classes,
              target shape: (N) where each value is 0<=targets[i]<=C-1
              output shape: scala.
    '''
    criterion=nn.NLLLoss()
    return criterion

def train(args,model,train_loader,optimizer,criterion,in_channels,sequence_length,current_epoch):
    model.train()
    train_loss=0
    for batch_idx,data in enumerate(train_loader):
        inputs,targets=data
        if args.cuda:
            inputs,targets=inputs.cuda(),targets.cuda()
        inputs=inputs.view(-1,in_channels,sequence_length)
        if args.permute:
            inputs=inputs[:,:,permute]
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,targets)
        loss.backward()
        if args.gradient_clip>0:
            torch.nn.utils.clip_grad_norm_(m.parameters(),args.gradient_clip)
        optimizer.step()
        train_loss+=loss
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}\t({:.0f}%)]\tLoss: {:.6f}'
                .format(current_epoch,batch_idx*args.batch_size,len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),train_loss/args.log_interval))
            '''
            in the beginning, I forgot this statement,
            and then some weird things happened:
            the value of the loss function steadily increased!
            Oh Jesus...After a long time looking for bugs,
            I finally found that I forgot to clear the loss value...
            Stupid like me...
            '''
            train_loss=0

def test(model,test_loader):
    model.eval()
    correct=0
    test_loss=0
    with torch.no_grad():
        for inputs,targets in test_loader:
            if args.cuda:
                inputs,targets=inputs.cuda(),targets.cuda()
            inputs=inputs.view(-1,in_channels,sequence_length)
            if args.permute:
                inputs=inputs[:,:,permute]
            outputs=model(inputs)
            loss=F.nll_loss(outputs,targets,reduction='none').data[0]
            test_loss+=loss
            pred=outputs.max(dim=1)[1]
            correct+=pred.eq(targets.view_as(pred)).cpu().sum().item()
    
    test_loss/=len(test_loader.dataset)
    accuracy=correct/len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'
                    .format(test_loss,correct,len(test_loader.dataset),
                    100*accuracy))
    return test_loss

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def warning_if_cuda_exists_but_not_used():
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')

def move_model_to_cuda(args,model):
    if args.cuda:
        return model.cuda()
    else:
        return model

def move_tensor_to_cuda(tensor):
    if args.cuda:
        return tensor.cuda()
    else:
        return tensor

if __name__=='__main__':
    args=parse_args()
    print(args)

    set_random_seed(args.seed)
    warning_if_cuda_exists_but_not_used()
    lr=args.initial_lr

    num_channels=[args.n_hidden]*args.n_layers
    in_channels=1
    kernel_size=args.kernel_size
    dropout=args.dropout
    out_features=10
    sequence_length=28*28 # 28*28 is the size of pictures in MNIST dataset
    # model
    m=create_model(num_channels,in_channels,kernel_size,dropout,out_features)
    m=move_model_to_cuda(args,m)

    if args.permute:
        permute=torch.Tensor(np.random.permutation(sequence_length)).long()
        permute=move_tensor_to_cuda(permute)

    train_loader,test_loader=data_loader(args.batch_size)
    optimizer=get_optimizer(args.optimizer,lr,m)
    criterion=get_criterion()

    for ep in range(1,args.epoch+1):
        train(args,m,train_loader,optimizer,criterion,in_channels,sequence_length,ep)
        test(m,test_loader)
        # learning rate annealing
        if ep%10==0:
            lr/=10
            for g in optimizer.param_groups:
                g['lr']=lr
