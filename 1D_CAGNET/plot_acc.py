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


def plot_acc_to_file(epochs, acc_train, acc_val, acc_test):
    fig, ax = plt.subplots()
    limited_size = 20
    limited_plot = lambda acc, label:ax.plot(epochs[limited_size:], acc[limited_size:], label=label)
    limited_plot(acc_train, 'train')
    limited_plot(acc_val, 'val')
    limited_plot(acc_test, 'test')
    ax.legend() 
    ax.set(xlabel='epoch', ylabel='acc', title='')
    ax.grid()
    fig.savefig("test.png")


def read_log_and_plot(log_fname):
    lines = read_lines_since_last_occur(log_fname, 'dist data inited')
    data = []
    for line in lines:
        #                     -8     -7       -6     -5      -4   -3      -2    -1    
        # 18:44:46.285616 [0] Epoch: 495/500, Train: 0.9293, Val: 0.9313, Test: 0.9314
        if 'Epoch' in line and 'Train:' in line:
            parts = line.split(' ')
            epoch = int(parts[-7].split('/')[0])
            train_acc = float(parts[-5][:-1])
            val_acc = float(parts[-3][:-1])
            test_acc = float(parts[-1][:-1])
            data.append((epoch, train_acc, val_acc, test_acc))
    plot_acc_to_file(*list(map(list, zip(*data))))


if __name__=='__main__':
    read_log_and_plot('testlog.txt')
    #read_log_and_plot('all_log_0.txt')

