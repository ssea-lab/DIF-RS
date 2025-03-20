# coding=utf-8
import random

import pymysql
import yaml

import numpy as np

from objective import timestamp_idx


class DataGenerator:
    def __init__(self, dataset):
        self.db = DataGenerator.connect()
        self.cursor = self.db.cursor()
        self.dataset = dataset

    @staticmethod
    def connect():
        with open('database.yaml', 'r') as file:
            database_info = yaml.safe_load(file)

        mysql_info = database_info['mysql']
        host = mysql_info['host']
        user = mysql_info['user']
        passwd = mysql_info['passwd']

        return pymysql.connect(host=host, user=user, passwd=passwd, db='edge_task_scheduling', charset='utf8')

    def get_instance(self, task_n=500, server_m=5):
        task_n_per_server = task_n // server_m

        # query = 'SELECT MAX(id) FROM {}'.format(self.dataset)
        # self.cursor.execute(query)
        # max_id = self.cursor.fetchone()[0] - task_n_per_server
        # start_id = random.randint(0, max_id)

        # query = 'SELECT * FROM {} WHERE id > {} LIMIT {}'.format(self.dataset, start_id, task_n_per_server)

        query = 'SELECT * FROM {} LIMIT {}'.format(self.dataset, task_n_per_server)
        self.cursor.execute(query)
        # [id, cpu,io,bandwidth,ram,timestamp,duration,latitude,longitude]
        task_list = self.cursor.fetchall()
        task_list = np.array(task_list)[:, 1:-2]
        task_list[:, timestamp_idx] = abs(task_list[:, timestamp_idx] - task_list[-1][timestamp_idx])

        return task_list

if __name__ == '__main__':
    data = DataGenerator("alibaba_cluster_trace")
    instance = data.get_instance(task_n=50,server_m=1)
    print(len(instance))