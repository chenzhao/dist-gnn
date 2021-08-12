import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import sys
import re
import os


def read_lines_since_last_occur(fname, trigger):
    fsize = os.stat(fname).st_size
    bufsize = min(1024*16, fsize-1)
    with open(fname, 'r') as f:
        for i in range(1, fsize//bufsize+1):
            f.seek(fsize-bufsize*i)
            lastlines = f.readlines()
            for line_no, line in enumerate(reversed(lastlines)):
                if trigger in line:
                    return lastlines[len(lastlines)-line_no:]
        raise Exception('no occurrance of %s in %s'%(trigger, fname))


def plot_acc_to_file(filename, epochs, acc_train, acc_val, acc_test):
    fig, ax = plt.subplots()
    limited_size = 20
    limited_plot = lambda acc, label:ax.plot(epochs[limited_size:], acc[limited_size:], label=label)
    limited_plot(acc_train, 'train')
    limited_plot(acc_val, 'val')
    limited_plot(acc_test, 'test')
    ax.legend() 
    ax.set(xlabel='epoch', ylabel='acc', title='')
    ax.grid()
    fig.savefig(filename+".png")


import datetime as dt

def plot_to_file(filename, epochs, values_dict):
    fig, ax = plt.subplots()
    limited_size = 0
    limited_plot = lambda val, label:ax.plot(epochs[limited_size:], val[limited_size:], label=label)
    for title, values in values_dict.items():
        limited_plot(values, title)
    ax.legend() 
    ax.set(xlabel='epoch', ylabel='acc', title='')
    ax.grid()
    fig.savefig(filename+".png")



def read_log_and_plot(log_fname):
    lines = read_lines_since_last_occur(log_fname, 'dist data inited')
    accs = []
    nll_losses = []
    nll_epoches = []
    l2_diffs = []
    l2_epoches = []
    backl2_diffs = []
    backl2_epoches = []
    timestamp = lines[0].split(' ')[0].replace(':', '-')
    for line in lines:
        #                     -8     -7       -6     -5      -4   -3      -2    -1    
        # 18:44:46.285616 [0] Epoch: 495/500, Train: 0.9293, Val: 0.9313, Test: 0.9314
        if 'Epoch' in line and 'Train:' in line:
            parts = line.split(' ')
            epoch = int(parts[-7].split('/')[0])
            train_acc = float(parts[-5][:-1])
            val_acc = float(parts[-3][:-1])
            test_acc = float(parts[-1][:-1])
            accs.append((epoch, train_acc, val_acc, test_acc))

        # 15:50:23.239814 [1] nll loss: 199 0.712547779083252
        if 'nll loss' in line:
            parts = line.split(' ')
            epoch = int(parts[-2])
            loss = float(parts[-1])
            nll_losses.append(loss)
            nll_epoches.append(epoch)
        # 15:50:23.239512 [2] layer2 cache: 199 2 True 0.08231066912412643
        if 'layer2 cache' in line:
            parts = line.split(' ')
            epoch = int(parts[-4])
            loss = float(parts[-1])
            l2_diffs.append(loss)
            l2_epoches.append(epoch)

        if 'backward_layer2:' in line:
            parts = line.split(' ')
            epoch = int(parts[-4])
            loss = float(parts[-1])
            backl2_diffs.append(loss)
            backl2_epoches.append(epoch)

    if accs:
        plot_acc_to_file('acc_%s_'%timestamp, *list(map(list, zip(*accs))))

    plot_to_file('l2_%s_'%timestamp, l2_epoches, {'l2':l2_diffs})
    plot_to_file('nll_%s_'%timestamp, nll_epoches, {'nll':nll_losses})

    plot_to_file('backl2_%s_'%timestamp, backl2_epoches, {'backl2':backl2_diffs})


if __name__=='__main__':
    # read_log_and_plot('testlog.txt')
    read_log_and_plot('all_log_0.txt')
    # plot_to_file('test', [1,2,5,10], {'v1':[2,4,5,6], 'v2':[4,5,6,7]})

