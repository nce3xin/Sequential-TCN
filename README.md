# Sequential TCN
[![](https://img.shields.io/badge/language-python3-blue.svg)](https://www.python.org/ 'built with python3')
[![](https://img.shields.io/badge/weibo-@nce3xin-orange.svg)](https://weibo.com/u/1646027981 'my weibo')
[![](https://img.shields.io/github/license/nce3xin/Sequential-TCN.svg)](https://github.com/nce3xin/Sequential-TCN/blob/master/LICENSE 'LICENSE')

## Table of contents
* [Overview](#Overview)
* [Usage](#Usage)
* [Visualization](#Visualization)
* [Results](#Results)
* [Requirements](#Requirements)
* [References](#References)

## Overview
This repository is a rewrite of the original author's code for two tasks in this paper: [An empirical evaluation of generic convolutional and recurrent networks for sequence modeling](https://arxiv.org/pdf/1803.01271.pdf) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun:
* Sequential MNIST. Sequential MNIST is frequently used to test a recurrent network’s ability to retain information from the distant past. In this task, MNIST images (LeCun et al., 1998) are presented to the model as a 784×1 sequence for digit classification. 
* P-MNIST. In the more challenging P-MNIST setting, the order of the sequence is permuted at random.
  
The model uses TCN as proposed in the above paper instead of RNN.

## Usage
```
$ git clone git@github.com:nce3xin/Sequential-TCN.git
$ cd Sequential-TCN/
```
For more command line arguments, please check:
```
$ python setup.py -h
```
### Sequential MNIST Task
* GPU version:
```
$ python setup.py runs/exp-1/ --cuda 
```
* CPU version:
```
$ python setup.py runs/exp-1/
```
### P-MNIST Task
P-MNIST task means permuted MNIST task.
* GPU version
```
$ python setup.py runs/exp-1/ -p --cuda
```
* CPU version
```
$ python setup.py runs/exp-1/ -p
```

## Visualization
We use [tensorboardX](https://github.com/lanpa/tensorboardX) for training visualization purpose.

Once the `setup.py` is done, run the following command to visualize the training process:
```
$ tensorboard --logdir=runs/exp-1/
```
Then open your browser (chrome or firefox, the others may not display), enter http://localhost:6006. If you use remote server, you need to use the tunnel option in Xshell to show the panel on the remote server locally. Follow the steps below:

* Open your Xshell, select the remote server you want to connect in the left panel. Right click and select 'Properties'. Then a dialog pops up.
* Select 'Tunneling' in the left panel, click 'Add' button in the right panel, fill the form as below:

Key | Value | 
:-: | :-: | 
Type(Direction) | Local(Outgoing) | 
Source Host | localhost| 
Listening Port | 6006
Destination Host | localhost |
Destination Port | 6006 |

Then ssh your server as normal, start tensorboard, then enter http://localhost:6006 in your local browser. Then you can watch remote tensorboard on your own pc.

## Results
### Training loss
![](images/train_loss.svg 'training loss')

### Test Accuracy
![](images/test_accuracy.svg 'test accuracy')
After 20 epochs of training, the classification accuracy on the final test set reached 98.8%.
## Requirements
* python3.5+
* [pytorch](https://pytorch.org/)
* [tensorboardX](https://github.com/lanpa/tensorboardX). For the purpose of training visualization. 

## References
* [temporal convolutional networks.](https://github.com/locuslab/TCN)