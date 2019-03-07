# Sequential TCN
[![](https://img.shields.io/badge/language-python3-blue.svg)](https://www.python.org/ 'built with python3')
[![](https://img.shields.io/badge/weibo-@nce3xin-orange.svg)](https://weibo.com/u/1646027981 'my weibo')
[![](https://img.shields.io/github/license/nce3xin/Sequential-TCN.svg)](https://github.com/nce3xin/Sequential-TCN/blob/master/LICENSE 'LICENSE')

## Table of contents
* [Overview](#Overview)
* [Usage](#Usage)
* [Requirements](#Requirements)
* [Thanks](#Thanks)

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
$ python setup.py --cuda
```
* CPU version:
```
$ python setup.py
```
### P-MNIST Task
P-MNIST task means permuted MNIST task.
* GPU version
```
$ python setup.py -p --cuda
```
* CPU version
```
$ python setup.py -p
```
## Requirements
* python3.5+
* [pytorch](https://pytorch.org/)

## Thanks
* [temporal convolutional networks.](https://github.com/locuslab/TCN)