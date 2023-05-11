# -*- coding: UTF-8 -*-

'''
Write the result(every epoch) into file
'''

import os
import csv

class ResultWriter():

    def __init__(self, save_folder, file_name):
        super(ResultWriter, self).__init__()
        self.save_path = os.path.join(save_folder, file_name)
        self.csv_writer = None

    def create_csv(self, csv_head):
        with open(self.save_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_head)

    def write_csv(self, data_row):
        with open(self.save_path, 'a') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(data_row)