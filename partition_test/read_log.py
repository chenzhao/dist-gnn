# -*- coding:utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import numpy as np


class NodeData:
    def __init__(self, rank):
        self.rank = rank
        self.data = {}

    def add(self, key, item):
        if key not in self.data:
            self.data[key] = [item]
        else:
            self.data[key].append(item)

    def has_key(self, key):
        return key in self.data

    def get(self, key):
        return self.data[key]


def plot(order, data, y_label):
    # columns should be nodes, rows should be keys
    table = []
    columns = [node_data.rank for node_data in data]
    rows = order
    for key in order:
        temp = []
        for node_data in data:
            temp.append(sum(node_data.get(key)))
        table.append(temp)
    # data = [[66386, 174296, 75131, 577908, 32015],
    #         [58230, 381139, 78045, 99308, 160454],
    #         [89135, 80552, 152558, 497981, 603535],
    #         [78415, 81858, 150656, 193263, 69638],
    #         [139361, 331509, 343164, 781380, 52269]]

    # columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
    # rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

    # values = np.arange(0, 2500, 500)
    # value_increment = 1000

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(table)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, table[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + table[row]
        cell_text.append(['%1.1f' % x for x in table[row]])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel(y_label)
    # plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    # plt.title('Loss by Disaster')
    plt.tight_layout()
    plt.show()


def test():
    data = [[66386, 174296, 75131, 577908, 32015],
            [58230, 381139, 78045, 99308, 160454],
            [89135, 80552, 152558, 497981, 603535],
            [78415, 81858, 150656, 193263, 69638],
            [139361, 331509, 343164, 781380, 52269]]

    columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
    rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

    values = np.arange(0, 2500, 500)
    value_increment = 1000

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Loss in ${0}'s".format(value_increment))
    plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title('Loss by Disaster')

    plt.show()


if __name__ == '__main__':
    # test()
    file_prefix = 'ml'
    with open(f'logs/{file_prefix}_timer.pkl', 'rb') as f:
        # times: [{'key', 'duration'}]
        times = pickle.load(f)
    with open(f'logs/{file_prefix}_mem.pkl', 'rb') as f:
        # mems: ['{time} {rank} [gcn_broadcast_layer1_ep10] size: 1.02 MBytes']
        mems = pickle.load(f)
    run_times = []
    times_key_order = []
    broadcast_sizes = []
    sizes_key_order = []
    for rank in range(len(times)):
        node_data = NodeData(rank)
        duration_array = times[rank]
        for item in duration_array:
            if item['key'] == 'train':
                continue
            key, epoch = item['key'].strip().rsplit('_', 1)
            if key not in times_key_order:
                times_key_order.append(key)
            epoch = int(epoch[2:])
            d = item['duration']
            node_data.add(key, d)
        run_times.append(node_data)
    for rank in range(len(mems)):
        node_data = NodeData(rank)
        for line in mems[rank]:
            if 'MBytes' not in line:
                continue
            # line: {time} {rank} [gcn_broadcast_layer1_ep10] size: 1.02 MBytes
            parts = line.split()
            key, epoch = parts[2].strip('[]').rsplit('_', 1)
            if key in times_key_order and key not in sizes_key_order:
                sizes_key_order.append(key)
            epoch = int(epoch[2:])
            size = float(parts[-2])
            # print(size)
            node_data.add(key, size)
        broadcast_sizes.append(node_data)

    print('times keys:')
    print(times_key_order)
    print('broadcast keys:')
    print(broadcast_sizes)
    plot(times_key_order, run_times, 'run time')
    plot(sizes_key_order, broadcast_sizes, 'broadcast size')
