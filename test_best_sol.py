import copy
import pickle
import pandas as pd
from operator import itemgetter
from itertools import groupby

MAX_SAME_WIDTH_LEN = 200000
MAX_BATCH_LEN = 800000
m = 4

rename_dict = {
    'No.': 'id',
    '宽度(mm)': 'width',
    '厚度(mm)': 'thick',
    '硬度': 'hard',
    '长度(mm)': 'length',
    '加工时间pi (s)': 'time'}

with open('test_c_mat.pkl', 'rb') as g:
    c_mat = pickle.load(g)


def read_in(path):
    df = pd.read_csv(path)
    df.rename(columns=rename_dict, inplace=True)
    df = df.sort_values(by=['width', 'time'], ascending=[False, True]).reset_index(drop=True)
    return df


data = read_in('data.csv')
width = list(data['width'])
length = list(data['length'])
time = list(data['time'])


def is_feasible_batch(batches: list) -> bool:
    """
    某长序列是否合约束

    :param batches: 长序列，不含有分隔符，且已按照宽度降序，格式为(df_id, width, length)
    :return: True 表示该序列符合约束，否则为 False
    """
    batches = copy.deepcopy(batches)
    l_sum_list = []
    for batch in batches:
        l_sum, c_width, width_sum = 0, None, 0
        for i in range(len(batch)):
            l_sum += batch[i][2]
            if batch[i][1] != c_width:
                c_width, width_sum = batch[i][1], batch[i][2]
            else:
                width_sum += batch[i][2]
                if width_sum > MAX_SAME_WIDTH_LEN:
                    return False
        l_sum_list.append(l_sum)
        if l_sum > MAX_BATCH_LEN:
            return False
    return True

def is_feasible_sol(sol_list: list):
    """
    判断序列是否可行

    :param sol_list: 序号序列，长度为板数量+m-1个分隔符
    :return: True 表示可行，False 表示不可行
    """
    if sol_list[0] == -1 or sol_list[-1] == -1:
        return False, None
    sol = list(sol_list)
    raw_batches = [list(g) for k, g in groupby(sol, lambda x: x == -1) if not k]
    batches = []
    # 对 batch 内部作宽度排序(稳定排序)
    for batch in raw_batches:
        tmp = sorted([(ind, width[ind], length[ind], time[ind]) for ind in batch], key=itemgetter(1), reverse=True)
        batches.append(tmp)
    return is_feasible_batch(batches), batches

def get_fitness(sol_list: list) -> int:
    """
    从长序列计算适应度的函数

    :param sol_list: 序号序列，长度为板数量+m-1个分隔符
    :return: 适应度（越小越好），-1表示不可行解
    """
    # num_of_slab个元素 + m-1个分隔符
    # assert len(sol_list) == self.num_of_slab + self.m - 1, f'{len(sol_list)}, {self.num_of_slab + self.m - 1}'
    sol = list(sol_list)
    flag, sorted_batches = is_feasible_sol(sol)
    if not flag:
        return -1
    else:
        batches = copy.deepcopy(sorted_batches)
        # batch:(df_id, width, length)
        # T 和 P 在一次遍历中获得
        # cost_T
        peri_sum = 0  # batch前累计时间
        c_t = 0
        c_p = 0
        for batch in batches:
            item_t = 0  # batch内时间
            for i in range(len(batch)):
                c_t += item_t + peri_sum
                c_p += (c_mat[batch[i - 1][0], batch[i][0]] if i > 0 else 0)
                item_t += time[int(batch[i][0])]
            peri_sum += (10 + item_t)
        # cost_M
        c_m = 1000 * m
        print(f'{c_t, c_m, c_p}')
        return c_t + c_m + c_p

