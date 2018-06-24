import csv

import numpy as np
import matplotlib.pyplot as plt
import os

class DataLogger:
    def __init__(self):
        self.data_dict = {}
        self.ids_list = []

    def reset(self):
        self.data_dict = {}
        self.ids_list = []

    def insert_data(self, id_num, key, data_point):
        if key not in self.data_dict:
            self.data_dict[key] = {}
        if id_num not in self.data_dict[key]:
            self.data_dict[key][id_num] = []
            self.ids_list.append(id_num)
        self.data_dict[key][id_num].append(data_point)

    def print_data(self):
        for key in self.data_dict.keys():
            for id_num in self.data_dict[key].keys():
                max = np.max(self.data_dict[key][id_num])
                min = np.min(self.data_dict[key][id_num])
                var = np.std(self.data_dict[key][id_num])
                mean = np.mean(self.data_dict[key][id_num])
                print "drone_id: {} key: {} mean: {} std: {} min: {} max: {}".format(id_num, key, mean, var, min, max)


    def draw_graph(self, id_num, key_list, colors):
        data = []
        for keys in key_list:
            data.append(self.data_dict[keys][id_num])

        for index, d in enumerate(data):
            plt.plot(d, color=colors[index])

        plt.show()

    def save_to_folder(self, folder, file_name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i in self.ids_list:
            output_f = file_name + "_" + str(i) + ".csv"
            output_f = os.path.join(folder, output_f)

            with open(output_f, 'w') as csvfile:
                fieldnames = self.data_dict.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                for idx in range(len(self.data_dict[fieldnames[0]][i])):
                    d = {}
                    for k in self.data_dict.keys():
                        d[k] = self.data_dict[k][i][idx]
                    writer.writerow(d)


