# -*- coding: utf-8 -*-
import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from tqdm import tqdm
import time
import math

def myfind(x,y):
    return [ a for a in range(len(y)) if y[a] == x]

# MiniBatchKmeans聚类
def cluster(filename):

    F=150
    batch_size = 50000
    # 加载itemAttribute.txt中的数据
    itemid=[]
    attr1=[]
    attr2=[]
    f = open(filename,encoding='UTF-8')
    line = f.readline()
    # 读取文件每行内容并进行切片，添加到相应的list
    while line:
        line=line.replace('None','0')
        item_id,item_x,item_y=line.split('|')
        itemid.append(int(item_id))
        attr1.append(int(item_x))
        attr2.append(int(item_y.split('\n')[0]))
        line = f.readline()
    f.close()
    X=[attr1,attr2]
    X=np.transpose(X)
    # 使用MiniBatchKmeans算法进行聚类
    KM=MiniBatchKMeans(init='k-means++', n_clusters=F, batch_size=batch_size, n_init=10, max_no_improvement=10, verbose=0)
    # y_pred和centers分别是聚类得到的每个点的分类label和每个类的中心点坐标
    y_pred = KM.fit_predict(X)

    return  itemid, y_pred

# 加载训练集
def LoadTrainset(filename):
    # 聚类
    print('开始加载itemAttribute文件!')
    item_ID, y_pred=cluster('./Data/itemAttribute.txt')
    print('itemAttribute文件加载完毕!')
    print('开始加载train文件!')
    with open(filename, 'r') as f:
        lines = f.readlines()
        f.close()
    rating_list = []
    index = 0
    n_users = 0
    n_items = 0
    while index < len(lines):
        # print(index)
        # 获取userid
        line = lines[index].strip()
        index += 1
        if line == "":
            continue
        if '|' not in line:
            continue
        userid, n_ratings = line.split('|')
        n_users=max(n_users, int(userid))
        n_ratings = int(n_ratings)
        user_item_num=n_ratings
        # 获取用户对item的打分
        for i in range(n_ratings):
            line = lines[index+i].strip()
            if line == "":
                index += 1
                i -= 1
                continue
            itemid, grade = line.split('  ')
            n_items=max(n_items, int(itemid))
            rating_list.append((userid, itemid, float(grade)/10))
            # 填充打分数量小于15的
            if user_item_num<15:
                if int(line.split('  ')[0]) in item_ID:
                    temp_list=myfind(y_pred[item_ID.index(int(line.split('  ')[0]))],y_pred)
                    temp_index=random.randint(0,len(temp_list)-1)
                    temp=temp_list[temp_index]
                    rating_list.append((userid, item_ID[temp], int(line.split('  ')[1].split('  ')[0])))
                    user_item_num=user_item_num+1
        index += n_ratings
    print('train文件加载完毕!')
    return rating_list,n_users+1,n_items+1

# 加载测试集
def LoadTestset(filename):

    print('开始加载test文件!')
    with open(filename, 'r') as f:
        lines = f.readlines()
        f.close()
    test_list = []
    index = 0
    while index < len(lines):
        # 切分userid 和打分的个数
        line = lines[index].strip()
        index += 1
        if line == "":
            continue
        if '|' not in line:
            continue
        userid, n_ratings = line.split('|')
        n_ratings = int(n_ratings)

        # 获取itemid
        for i in range(n_ratings):
            line = lines[index + i].strip()
            if line == "":
                index +=1
                i -= 1
                continue
            itemid = line
            test_list.append((userid, itemid))
    print('test文件加载完毕!')
    return test_list

class Trainset(object):

    """
        preprocess the trainset and calculate some useful data

        args & attributes:
            ratingList: userid itemid score
            ur: the users ratings
            ir: the items ratings
            n_users: total number of users
            n_items: total number of items
            n_ratings: total number of ratings
            global_mean: the mean of all ratings
            raw_users_id: raw user id -> inner user id
            raw_items_id: raw item id -> inner item id
            inner_users_id: inner user id -> raw user id
            inner_items_id: inner item id -> raw item id
    """

    def __init__(self, ur, ir, n_users, n_items, n_ratings, raw_users_id, raw_items_id, rating_scale):
        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self._global_mean = None
        self.raw_users_id = raw_users_id
        self.raw_items_id = raw_items_id
        self.inner_users_id = None
        self.inner_items_id = None
        self.rating_scale = rating_scale

    def get_all_ratings(self):
        for uid, u_rating in self.ur.items():
            for iid, rate in u_rating:
                yield uid, iid, rate

    def get_user_ratings(self, uid):
        for iid, rate in self.ur[uid]:
            yield iid, rate

    @property
    def global_mean(self):
        if self._global_mean is None:
            self._global_mean = np.mean([r for (_, _, r) in self.get_all_ratings()])
        return self._global_mean



# 从文件中加载数据并创建训练集
def construct_trainset(dataset, n_users, n_items):
    raw_users_id = dict()
    raw_items_id = dict()
    ur = defaultdict(list)
    ir = defaultdict(list)
    u_index = 0
    i_index = 0
    for ruid, riid, rating in dataset:
        # 创建外部userid到内部userid的映射
        try:
            uid = raw_users_id[ruid]
        except KeyError:
            uid = u_index
            raw_users_id[ruid] = uid
            u_index += 1

        # 创建外部itemid到内部itemid的映射
        try:
            iid = raw_items_id[riid]
        except KeyError:
            iid = i_index
            raw_items_id[riid] = iid
            i_index += 1

        ur[uid].append([iid, rating])
        ir[iid].append([uid, rating])
    n_ratings = len(dataset)
    rating_scale = (0, 100)
    trainset = Trainset(ur, ir, n_users, n_items, n_ratings, raw_users_id, raw_items_id, rating_scale)
    return trainset

class SVD(object):
    # 初始化
    def __init__(self, n_factors=8, n_epochs=3, learningRate=0.005, regularization=0.018, init_mean=0.0, init_std_dev=0.05):
        self.trainset = None
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learningRate = learningRate
        self.regularization = regularization
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.bu = None
        self.bi = None
        self.pu = None
        self.qi = None

    # 求内积
    def InerProduct(self, v1, v2, uid, iid, factor):

        result = 0
        for i in range(factor):
            result += v1[iid, i] * v2[uid, i]
        return result

    #进行训练
    def train(self, trainset):
        # 随机初始化user和item的factors
        self.trainset = trainset

        global_mean = self.trainset.global_mean
        bu = np.zeros(self.trainset.n_users, np.double)
        bi = np.zeros(self.trainset.n_items, np.double)
        randstate = np.random
        pu = randstate.normal(self.init_mean, self.init_std_dev, (self.trainset.n_users, self.n_factors))
        qi = randstate.normal(self.init_mean, self.init_std_dev, (self.trainset.n_items, self.n_factors))


        # 迭代
        for cur_epoch in range(self.n_epochs):
            # 更新学习率
            learningRate = self.learningRate * 0.95
            print('第 %d 次训练!' % (cur_epoch + 1))
            for uid, iid, rate in self.trainset.get_all_ratings():
                # 求内积
                multi = self.InerProduct(qi, pu, uid, iid, self.n_factors)
                predict = 0

                predict = global_mean + bu[uid] + bi[iid] + multi
                # 不能超出分数的范围0-100
                # predict = max(predict, 0)
                # predict = min(predict, 10)
                err = rate - predict

                # 更新参数bu,bi
                bu[uid] += learningRate * (err - self.regularization * bu[uid])
                bi[iid] += learningRate * (err - self.regularization * bi[iid])

                # 更新参数pu,qi
                for f in range(self.n_factors):
                    temp = pu[uid, f]
                    pu[uid, f] += learningRate * (err * qi[iid, f] - self.regularization * temp)
                    qi[iid, f] += learningRate * (err * temp - self.regularization * qi[iid, f])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        return

    def predict(self, ruid, riid):
        ruid=int(ruid)
        riid=int(riid)
        grade = self.trainset.global_mean + self.bu[ruid] + self.bi[riid] + np.dot(self.qi[riid], self.pu[ruid])
        # 不能超出分数的范围0-100
        grade = max(grade, 0)
        grade = min(grade, 10)
        return round(grade)*10

    def predict_all(self, testset, ur, ir):

        print('开始预测!')
        grades = []
        for ruid, riid in tqdm(testset):
            re_uid = ur[ruid]
            try:
                re_iid = ir[riid]
            except KeyError:
                ir[riid] = len(ir)
                re_iid = ir[riid]

            grade = self.predict(re_uid, re_iid)
            grades.append(grade)
        return grades

def split_train_and_test(dataset, n_split, seed, index):
    random.seed(seed)
    trainset = []
    testset = []
    # 分割测试集与训练集
    for i in range(len(dataset)):
        if random.randint(0, n_split) == index:
            testset.append(dataset[i])
        else:
            trainset.append(dataset[i])
    return trainset, testset

# 计算均方根误差
def RMSE(predictions, targets):
    assert len(predictions) == len(targets)
    length = len(predictions)
    sum = 0
    for i in range(length):
        sum += (predictions[i] - targets[i])**2
    sum /= length
    return math.sqrt(sum)

# 整体评估
def evaluation():
    # 加载数据集并进行分割，切分为训练集和测试集
    dataset,n_users,n_items = LoadTrainset('./Data/train.txt')
    trainset, testset = split_train_and_test(dataset, 10, 1, 1)
    dataset = construct_trainset(dataset,n_users,n_items)
    # 创建训练集并进行训练
    trainset = construct_trainset(trainset,n_users,n_items)
    svd = SVD()

    svd.train(trainset)
    targets = [grade*10 for (_, _, grade) in testset]
    items = [[userid, itemid] for (userid, itemid, grade) in testset]
    ur = dataset.raw_users_id
    ir = dataset.raw_items_id
    predictions = svd.predict_all(items, ur, ir)
    # 计算RMSE
    grade = RMSE(predictions, targets)
    print('RMSE:',grade)



if __name__ == '__main__':
    trainset,n_users,n_items = LoadTrainset('./Data/train.txt')
    trainset = construct_trainset(trainset,n_users,n_items)

    print('用户数量:', trainset.n_users)
    print('物品数量:', trainset.n_items)
    print('打分数量:', trainset.n_ratings)
    print('打分平均值:', trainset.global_mean*10)

    # 加载测试集
    start_time = time.time()
    testset = LoadTestset('./Data/test.txt')
    load_time = time.time() - start_time
    print("加载测试集的时间为 {} seconds .".format(load_time))

    svd = SVD()

    start_time = time.time()
    svd.train(trainset)
    train_time = time.time() - start_time
    print("训练的时间为 {} seconds .".format(train_time))

    ur = trainset.raw_users_id
    ir = trainset.raw_items_id
    results = svd.predict_all(testset, ur, ir)

    # 计算数量
    user_counts = dict()
    for i in range(len(testset)):
        userid, _ = testset[i]
        if userid in user_counts:
            user_counts[userid] += 1
        else:
            user_counts[userid] = 1

    # 写结果到文件
    f = open('./Data/result.txt', 'w+')
    cur_user = ""
    for i in range(len(testset)):
        userid, itemid = testset[i]
        grade = results[i]
        if userid == cur_user:
            f.write(itemid + '  ' + str(grade) + '\n')
        else:
            cur_user = userid
            f.write(userid + '|' + str(user_counts[userid]) + '\n')
            f.write(itemid + '  ' + str(grade) + '\n')
    f.close()
    print('已得到结果文件result.txt! 开始计算RMSE:')
    evaluation()