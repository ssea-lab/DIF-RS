# coding=gbk
import random

import pymysql
import yaml

import numpy as np

from algorithms.objective import timestamp_idx


class DataGenerator:
    def __init__(self, dataset):
        self.db = DataGenerator.connect()
        self.cursor = self.db.cursor()
        self.dataset = dataset

    @staticmethod
    def connect():
        """
        当前只支持mysql的连接，如果数据保存在其他数据库中，则需要扩写这块代码
        """
        with open('database.yaml', 'r') as file:
            database_info = yaml.safe_load(file)

        mysql_info = database_info['mysql']
        host = mysql_info['host']
        user = mysql_info['user']
        passwd = mysql_info['passwd']

        return pymysql.connect(host=host, user=user, passwd=passwd, db='edge_task_scheduling', charset='utf8')

    def get_instance(self, task_n=500, server_m=5):
        """
        这里的处理是求得每台服务器平均承载的任务后直接从数据库中抽取相应数量的任务
        因为当前调度只是在服务器内部的调度，并不考虑任务应该下发到哪个具体的服务器
        论文中给出的场景依旧是多服务器场景，随机选择覆盖范围内的服务器进行任务下发
        如果要将任务卸载结合进来，就不能这样简单处理，要考虑哪些服务器承载哪些任务

        这里最严谨的处理其实是先从edge_server表中取m条服务器出来，然后从任务数据集中取n条任务出来
        根据服务器的坐标、任务的坐标，计算出距离，根据距离选择就近的服务器下发任务
        但其实没区别啦，因为任务的坐标数据也是随机生成的 :)
        万物随机 :)
        """
        task_n_per_server = task_n // server_m

        query = 'SELECT MAX(id) FROM {}'.format(self.dataset)
        self.cursor.execute(query)
        max_id = self.cursor.fetchone()[0] - task_n_per_server
        start_id = random.randint(0, max_id)

        # 从一个随机id开始抽取数据
        query = 'SELECT * FROM {} WHERE id > {} LIMIT {}'.format(self.dataset, start_id, task_n_per_server)
        self.cursor.execute(query)
        # [id, cpu,io,bandwidth,ram,timestamp,duration,latitude,longitude]
        task_list = self.cursor.fetchall()
        task_list = np.array(task_list)[:, 1:-2]
        task_list[:, timestamp_idx] = abs(task_list[:, timestamp_idx] - task_list[-1][timestamp_idx])

        return task_list

if __name__ == '__main__':
    data = DataGenerator("alibaba_cluster_trace")
    print(data.get_instance())