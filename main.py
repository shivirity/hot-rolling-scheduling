import math
import random
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
import logging
import colorlog

from penalty_func import get_p_hard, get_p_thick, get_p_width
from test_best_sol import is_feasible_sol

# 理论下界（只考虑时间）: 557910.0

random.seed(42)  # 42
# set logger
log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename='run_file.log', mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
stream_fmt = colorlog.ColoredFormatter(
    fmt="%(log_color)s[%(asctime)s] - %(filename)-8s - %(levelname)-7s - line %(lineno)s - %(message)s",
    log_colors=log_colors_config)
file_fmt = logging.Formatter(
    fmt="[%(asctime)s] - %(name)s - %(levelname)-5s - %(filename)-8s : line %(lineno)s - %(funcName)s - %(message)s"
    , datefmt="%Y/%m/%d %H:%M:%S")
sh.setFormatter(stream_fmt)
fh.setFormatter(file_fmt)
logger.addHandler(sh)
logger.addHandler(fh)
sh.close()
fh.close()

rename_dict = {
    'No.': 'id',
    '宽度(mm)': 'width',
    '厚度(mm)': 'thick',
    '硬度': 'hard',
    '长度(mm)': 'length',
    '加工时间pi (s)': 'time'}


def read_in(path):
    df = pd.read_csv(path)
    df.rename(columns=rename_dict, inplace=True)
    df = df.sort_values(by=['width', 'time'], ascending=[False, True]).reset_index(drop=True)
    return df


class Problem:

    def __init__(self, df, num_of_batch: int, iteration: int, population: int):
        # const of system
        self.local_iter = 1000  # 局部搜索的间隔代数
        # self.local_items = population
        self.local_items = int(population/2)
        self.max_batch_len = 800000
        self.max_same_width_len = 200000
        self.max_cross_time = 5      # 产生子代时的最大交叉次数
        self.temp = 800              # 起始温度 800
        self.temp_dec_rate = 0.99    # 降温速度 0.99
        self.p_in_batch = 0.02       # 板内交换概率 0.02
        self.p_be_batch = 0.45       # 板间交换概率 0.45
        self.p_multi = 0.28          # 多交换概率 0.28
        self.p_cross_batch = 0.2     # 跨板交换概率 0.2
        self.forward_insert = 0.05   # 前插概率 0.05
        # assert self.p_multi+self.p_cross_batch+self.p_be_batch+self.p_in_batch+self.forward_insert == 1
        logger.info(f'{self.p_in_batch,self.p_be_batch,self.p_multi,self.p_cross_batch,self.forward_insert}')

        # system setting
        # 这里放各种概率

        # const by users
        self.iteration = iteration  # 迭代次数
        self.m = num_of_batch  # 轧制单元数量
        self.pop = population  # 种群数量

        # from Dataframe
        self.num_of_slab = len(df['id'])
        self.id = list(df['id'])
        self.width = list(df['width'])
        self.thick = list(df['thick'])
        self.hard = list(df['hard'])
        self.length = list(df['length'])
        self.time = list(df['time'])

        # compute from Dataframe
        self.c_mat = self.get_c_mat()

        # show solutions
        self.best_record = None  # 最优成绩节点
        self.iter_x = None       # 迭代次数
        self.iter_y = None       # 历代最优成绩
        self.best_sol = None     # 种群最优解
        self.best_score = np.inf # 种群最优score

        # 均分思想
        init_slabs, init_sol_tmp = [_ for _ in range(self.num_of_slab)], []
        for i in range(self.m):
            init_sol_tmp += init_slabs[i:len(init_slabs):self.m]
            if i < self.m-1:
                init_sol_tmp.append(-1)
        self.init_sol = init_sol_tmp
        self.init_sol_score = self.get_fitness(self.init_sol)

        '''
        # 贪婪（按照总长度）
        ind_list = []
        l_sum = 0
        for i in range(self.num_of_slab):
            l_sum += self.length[i]
            if l_sum <= self.max_batch_len:
                pass
            else:
                ind_list.append(i-1)
                l_sum = 0
        ind_list.append(99)
        init_sol_tmp = [_ for _ in range(ind_list[0]+1)] + [-1]
        for i in range(1, len(ind_list)):
            init_sol_tmp += [_ for _ in range(ind_list[i-1]+1, ind_list[i]+1)]
            init_sol_tmp.append(-1)
        init_sol_tmp.pop(-1)
        self.init_sol = init_sol_tmp
        self.init_sol_score = self.get_fitness(self.init_sol)
        '''
        '''精细的贪婪
        init_batches = [[] for _ in range(self.m)]
        length_list, same_width_list = [0 for _ in range(self.m)], [[0, 0] for _ in range(self.m)]
        for i in range(self.num_of_slab):
            for k in range(self.m):
                if length_list[k] + self.length[i] <= self.max_batch_len:
                    if self.width[i] == same_width_list[k][0] and \
                            same_width_list[k][1] + self.length[i] > self.max_same_width_len:  # 不可行，尝试下一块板
                        continue
                    else:
                        init_batches[k].append(i)  # 放入板
                        length_list[k] += self.length[i]  # 更新长度
                        if same_width_list[k][0] == self.width[i]:
                            same_width_list[k][1] += self.length[i]
                        else:
                            same_width_list[k][0], same_width_list[k][1] = self.width[i], self.length[i]
                        break
                else:
                    continue
        assert sum([len(batch) for batch in init_batches]) == self.num_of_slab, f'{sum([len(batch) for batch in init_batches])}'
        init_sol_tmp = []
        for batch in init_batches:
            init_sol_tmp += batch
            init_sol_tmp.append(-1)
        init_sol_tmp.pop(-1)
        self.init_sol = init_sol_tmp
        self.init_sol_score = self.get_fitness(self.init_sol)
        '''
        assert self.init_sol.count(-1) == self.m - 1
        assert len(self.init_sol) == self.num_of_slab + self.m - 1
        assert len(set(self.init_sol)) == len(self.init_sol) - (self.m - 2)

        self.sols = None  # 用以储存自身的解种群
        self.sols_scores = None  # 用以存储解种群的scores
        # test
        self.sols = [copy.deepcopy(self.init_sol) for _ in range(self.pop)]
        self.sols_scores = [self.init_sol_score for _ in range(self.pop)]

        '''
        # new_init_solutions
        self.sols = None  # 用以储存自身的解种群
        self.sols_scores = None  # 用以存储解种群的scores

        init_slabs, init_sol_tmp = [_ for _ in range(self.num_of_slab)], []
        for i in range(self.m):
            init_sol_tmp += init_slabs[i:len(init_slabs):self.m]
            if i < self.m-1:
                init_sol_tmp.append(-1)
        self.init_sol = init_sol_tmp
        self.init_sol_score = self.get_fitness(self.init_sol)
        self.sols, self.sols_scores = [list(self.init_sol)], [self.init_sol_score]
        while True:
            random.shuffle(init_slabs)
            init_sol_tmp = []
            for i in range(self.m):
                init_sol_tmp += init_slabs[i:len(init_slabs):self.m]
                if i < self.m - 1:
                    init_sol_tmp.append(-1)
            init_tmp_score = self.get_fitness(init_sol_tmp)
            if init_tmp_score > 0:
                self.sols.append(list(init_sol_tmp))
                self.sols_scores.append(init_tmp_score)
            if len(self.sols) == self.pop:
                break
        '''

    def get_c_mat(self) -> np.ndarray:
        """ 获得板材之间的成本矩阵 """
        c_mat = np.zeros((self.num_of_slab, self.num_of_slab), order='C')
        for i in range(self.num_of_slab):
            for j in range(self.num_of_slab):
                c_mat[i, j] = get_p_width(self.width[i] - self.width[j]) + \
                              get_p_hard(self.hard[i] - self.hard[j]) + \
                              get_p_thick(self.thick[i] - self.thick[j])
        return c_mat

    def is_feasible_batch(self, batches: list) -> bool:
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
                    if width_sum > self.max_same_width_len:
                        return False
            l_sum_list.append(l_sum)
            if l_sum > self.max_batch_len:
                return False
        # logger.debug(f'each_batch:{[len(batch) for batch in batches], l_sum_list}')
        return True

    def is_feasible_sol(self, sol_list: list):
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
            tmp = sorted([(ind, self.width[ind], self.length[ind]) for ind in batch], key=itemgetter(1), reverse=True)
            batches.append(tmp)
        return self.is_feasible_batch(batches), batches

    def get_fitness(self, sol_list: list) -> int:
        """
        从长序列计算适应度的函数

        :param sol_list: 序号序列，长度为板数量+m-1个分隔符
        :return: 适应度（越小越好），-1表示不可行解
        """
        # num_of_slab个元素 + m-1个分隔符
        # assert len(sol_list) == self.num_of_slab + self.m - 1, f'{len(sol_list)}, {self.num_of_slab + self.m - 1}'
        sol = list(sol_list)
        flag, sorted_batches = self.is_feasible_sol(sol)
        if not flag:
            return -1
        else:
            batches = copy.deepcopy(sorted_batches)
            # batch:(df_id, width, length)
            # T 和 P 在一次遍历中获得
            # cost_T
            peri_sum = 0
            c_t = 0
            c_p = 0
            for batch in batches:
                item_t = 0
                for i in range(len(batch)):
                    c_t += item_t + peri_sum
                    c_p += (self.c_mat[batch[i - 1][0], batch[i][0]] if i > 0 else 0)
                    item_t += self.time[int(batch[i][0])]
                peri_sum += (10 + item_t)
            # cost_M
            c_m = 1000 * self.m
            # logger.info(f'{c_t, c_m, c_p}')
            return c_t + c_m + c_p

    def compute_fitness(self, sols):
        """返回各样本的适应度"""
        sols = list(sols)
        return [self.get_fitness(sol) for sol in sols]

    def get_single_rand_sample(self, rand_batch: int, separator_idx: list, sol: list):
        """self.get_new_sol中的辅助函数"""
        if rand_batch == 0:
            rand_ind = random.sample(sol[0:separator_idx[0]], 1)
        elif rand_batch == self.m - 1:
            rand_ind = random.sample(sol[separator_idx[rand_batch-1] + 1:len(sol)], 1)
        else:
            rand_ind = random.sample(sol[separator_idx[rand_batch-1] + 1:separator_idx[rand_batch]], 1)
        return int(rand_ind[0])

    def get_new_sol(self, input_sol: list):
        """
        个体的交叉操作

        :param sol: 长序列（包含分隔符 -1）
        :return:
        """
        sol = list(input_sol)
        separator_idx = [ind for ind, item in enumerate(sol) if item == -1]  # 分隔符的索引位置
        rand_num = random.uniform(0, 1)

        # 板内交换
        if rand_num < self.p_in_batch:
            rand_ind = random.randint(0, len(separator_idx)-1)
            # decide which batch
            if rand_ind == 0:
                min_ind, max_ind = 0, separator_idx[rand_ind]-1
            elif rand_ind == len(separator_idx)-1:
                min_ind, max_ind = separator_idx[rand_ind]+1, len(sol)-1
            else:
                min_ind, max_ind = separator_idx[rand_ind]+1, separator_idx[rand_ind+1]-1
            width_list = [self.width[i] for i in sol[min_ind:max_ind+1] if i > 0]
            width_set = list(set(width_list))
            random.shuffle(width_set)
            for width in width_set:
                if width_list.count(width) > 1:
                    same_width_list = [ind for ind in range(min_ind, max_ind+1) if self.width[sol[ind]] == width]
                    rand_choice = random.sample(same_width_list, 2)
                    sol[rand_choice[0]], sol[rand_choice[1]] = sol[rand_choice[1]], sol[rand_choice[0]]
                    fit = self.get_fitness(sol_list=sol)
                    logger.debug(f"in_batch, fitness:{fit}")
                    if fit < self.best_score and fit != -1:
                        logger.debug(f"success in_batch, fitness:{fit}")
                    assert len(set(sol)) == len(sol) - (self.m - 2)
                    assert len(sol) == self.num_of_slab + self.m - 1
                    return sol, fit
            # 没有重复的元素
            logger.debug(f"in_batch, fitness:{-1}")
            return None, -1

        # 板间交换
        elif rand_num < self.p_in_batch+self.p_be_batch:
            bat_x, bat_y = random.sample([_ for _ in range(self.m)], 2)  # 决定哪两块板之间交换
            rand_ind_x = self.get_single_rand_sample(bat_x, separator_idx, sol)
            rand_ind_y = self.get_single_rand_sample(bat_y, separator_idx, sol)
            sol[rand_ind_x], sol[rand_ind_y] = sol[rand_ind_y], sol[rand_ind_x]
            fit = self.get_fitness(sol_list=sol)
            logger.debug(f"be_batch, fitness:{fit}")
            if fit < self.best_score and fit != -1:
                logger.debug(f"success be_batch, fitness:{fit}")
            assert len(set(sol)) == len(sol) - (self.m - 2)
            assert len(sol) == self.num_of_slab + self.m - 1
            return sol, fit

        # 多交换(默认 default_num 块)
        elif rand_num < self.p_in_batch+self.p_be_batch+self.p_multi:
            default_num = 3
            multi_flag = False
            while multi_flag is False:
                multi_flag = True
                rand_ind_x, rand_ind_y = random.sample(sol, 2)
                if abs(rand_ind_x-rand_ind_y) < 3:
                    multi_flag = False
                    continue
                for i in range(default_num):
                    if sol[rand_ind_x+i] in separator_idx:
                        multi_flag = False
                        break
                    elif sol[rand_ind_y+i] in separator_idx:
                        multi_flag = False
                        break
            sol[rand_ind_x:rand_ind_x+default_num], sol[rand_ind_y:rand_ind_y+default_num] = \
                sol[rand_ind_y:rand_ind_y+default_num], sol[rand_ind_x:rand_ind_x+default_num]
            fit = self.get_fitness(sol_list=sol)
            logger.debug(f"multi_batch, fitness:{fit}")
            if fit < self.best_score and fit != -1:
                logger.debug(f"success multi_batch, fitness:{fit}")
            assert len(set(sol)) == len(sol) - (self.m - 2)
            assert len(sol) == self.num_of_slab + self.m - 1
            return sol, fit

        # 跨板交换
        elif rand_num < self.p_in_batch+self.p_be_batch+self.p_multi+self.p_cross_batch:
            default_num = 3  # 交换长度
            batch_len_list = [separator_idx[i]-separator_idx[i-1]-1 for i in range(1, len(separator_idx))]
            batch_len_list.insert(0, separator_idx[0])
            batch_len_list.append(len(sol)-separator_idx[-1]-1)
            safe_gap = min(min(batch_len_list), default_num)  # 最短的轧制单元长度
            assert safe_gap > 1
            rand_sep = int(random.sample(separator_idx, 1)[0])
            front_num, behind_num = random.sample([_+1 for _ in range(safe_gap)], 2)
            new_sw_sol = sol[rand_sep+1:rand_sep+1+behind_num] + [sol[rand_sep]] + sol[rand_sep-front_num:rand_sep]
            sol[rand_sep-front_num:rand_sep+1+behind_num] = new_sw_sol
            fit = self.get_fitness(sol_list=sol)
            logger.debug(f"cross_batch, fitness:{fit}")
            if fit < self.best_score and fit != -1:
                logger.debug(f"success cross_batch, fitness:{fit}")
            if len(set(sol)) != len(sol) - (self.m - 2):
                logger.error('here')
            assert len(set(sol)) == len(sol) - (self.m - 2)
            assert len(sol) == self.num_of_slab + self.m - 1
            return sol, fit

        # 前插操作
        else:
            sep_ind = int(random.sample([_ for _ in range(len(separator_idx))], 1)[0])
            if sep_ind == len(separator_idx) - 1:
                rand_sep = separator_idx[sep_ind]
                sel_item = int(random.sample(sol[rand_sep+1:len(sol)], 1)[0])
            else:
                sel_item = int(
                    random.sample(sol[separator_idx[sep_ind]+1:separator_idx[sep_ind+1]], 1)[0])
            if sep_ind == 0:
                insert_ind = random.sample([i for i in range(0, separator_idx[sep_ind])], 1)[0]
            else:
                insert_ind = random.sample([i for i in range(separator_idx[sep_ind-1]+1, separator_idx[sep_ind])], 1)[0]
            assert sel_item > -1
            sol.remove(sel_item)
            sol.insert(insert_ind, sel_item)
            fit = self.get_fitness(sol_list=sol)
            logger.debug(f"forward_insert, fitness:{fit}")
            if fit < self.best_score and fit != -1:
                logger.debug(f"success forward_insert, fitness:{fit}")
            assert len(set(sol)) == len(sol) - (self.m - 2)
            assert len(sol) == self.num_of_slab + self.m - 1
            return sol, fit

    def ga(self):
        """
        遗传算法主程序

        :return: 最好个体，最好个体值
        """
        # 获得所有个体适应度
        c_sols, c_scores = copy.deepcopy(self.sols), copy.deepcopy(self.sols_scores)
        next_sols, next_scores = [], []
        min_ind = c_scores.index(min(c_scores))
        # 最好的直接进入下一代
        next_sols.append(c_sols[min_ind])
        next_scores.append(c_scores[min_ind])
        # 其余需要变异产生子代
        for i in range(len(c_sols)):
            if i == min_ind:  # 选到最优个体
                for _ in range(self.max_cross_time):
                    new_sol, new_fitness = self.get_new_sol(c_sols[i])
                    if new_fitness < 0:
                        continue
                    allowance = True if math.log(self.temp/800, self.temp_dec_rate) > 0 else False
                    if new_fitness < c_scores[i] or allowance:
                        next_sols.append(new_sol)
                        next_scores.append(new_fitness)
                        break
            else:
                flag = False
                for _ in range(self.max_cross_time):
                    new_sol, new_fitness = self.get_new_sol(c_sols[i])
                    if new_fitness < 0:
                        continue
                    if c_scores[i] < new_fitness:
                        allowance = random.uniform(0, 1) < math.exp((c_scores[i]-new_fitness)/self.temp)
                    else:
                        allowance = True
                    '''
                    try:
                    except Exception as err:
                        logger.error(f"score:{c_scores[i]}, fitness:{new_fitness}, temp:{self.temp}")
                        logger.error(err)
                    '''
                    if new_fitness <= c_scores[i] or allowance:
                        next_sols.append(new_sol)
                        next_scores.append(new_fitness)
                        flag = True
                        break
                if not flag:
                    next_sols.append(c_sols[i])
                    next_scores.append(c_scores[i])

        '''最优个体也加入变异过程，需要去除一个种群的最差值'''
        assert len(next_sols) == self.pop or len(next_sols) == self.pop + 1
        if len(next_sols) == self.pop + 1:  # 最优解成功插入，去除一个最差值
            max_ind = next_scores.index(max(next_scores))
            next_sols.pop(max_ind)
            next_scores.pop(max_ind)

        assert len(next_sols) == len(next_scores) and len(next_scores) == self.pop
        self.sols, self.sols_scores = next_sols, next_scores
        next_min_ind = next_scores.index(min(next_scores))
        return next_sols[next_min_ind], next_scores[next_min_ind]

    def get_local_search(self, sol, score, swap_on):
        """
        获得局部搜索后的个体

        :param sol: 需要局部搜索的个体
        :param score: 需要局部搜索的个体得分
        :param swap_on: 是否搜交换
        :return: 局部搜索后的个体，局部搜索后的个体值
        """
        c_sol, c_score, flag = list(sol), score, True
        # sep_ind_f, sep_ind_l = c_sol.index(-1), 102-c_sol[::-1].index(-1)
        # 插板
        while flag:
            sep_inds = [ind for ind, item in enumerate(c_sol) if item == -1]
            for i in range(1, sep_inds[-1]):
                if i == sep_inds[1] or i == sep_inds[0]:
                    pass
                else:
                    new_sol = list(c_sol)
                    t_item = new_sol.pop(i)
                    for j in range(i):
                        test_sol = list(new_sol)
                        test_sol.insert(j, t_item)
                        test_score = self.get_fitness(test_sol)
                        if c_score > test_score > 0:
                            break
                    else:
                        continue
                    assert len(set(test_sol)) == len(test_sol) - (self.m - 2)
                    assert len(test_sol) == self.num_of_slab + self.m - 1
                    c_sol = test_sol
                    c_score = test_score
                    break
            else:
                flag = False

        if swap_on is True:
            # 交换
            flag = True
            while flag:
                sep_inds = [ind for ind, item in enumerate(c_sol) if item == -1]
                for i in range(1, sep_inds[-1]):
                    if i == sep_inds[1] or i == sep_inds[0]:
                        pass
                    else:
                        new_sol = list(c_sol)
                        for j in range(i):
                            test_sol = list(new_sol)
                            test_sol[i], test_sol[j] = test_sol[j], test_sol[i]
                            test_score = self.get_fitness(test_sol)
                            if c_score > test_score > 0:
                                break
                        else:
                            continue
                        assert len(set(test_sol)) == len(test_sol) - (self.m - 2)
                        assert len(test_sol) == self.num_of_slab + self.m - 1
                        c_sol = test_sol
                        c_score = test_score
                        break
                else:
                    flag = False

        return c_sol, c_score

    def local_search(self, swap_on):
        """
        局部搜索的主函数

        :param swap_on: 是否搜交换
        :return: 最好个体，最好个体的值
        """
        logger.info('Doing LocalSearch...')
        # 获得所有个体适应度
        c_sols, c_scores = copy.deepcopy(self.sols), copy.deepcopy(self.sols_scores)
        min_c_score, avg_c_score = min(c_scores), sum(c_scores)/len(c_scores)
        search_inds = random.sample([_ for _ in range(self.pop)], self.local_items)
        next_sols, next_scores = [c_sols[i] for i in range(self.pop) if i not in search_inds], \
                                 [c_scores[i] for i in range(self.pop) if i not in search_inds]
        local_proc = 1
        for search_ind in search_inds:
            search_item, item_score = c_sols[search_ind], c_scores[search_ind]
            search_sol, search_score = self.get_local_search(search_item, item_score, swap_on)
            next_sols.append(search_sol)
            next_scores.append(search_score)
            logger.info(f'LocalSearching: {local_proc}/{self.local_items}')
            local_proc += 1
        assert len(next_sols) == len(next_scores) and len(next_scores) == self.pop
        self.sols, self.sols_scores = next_sols, next_scores
        min_next_score, avg_next_score = min(next_scores), sum(next_scores) / len(next_scores)
        logger.info(f'c_scores: {c_scores}')
        logger.info(f'next_scores: {self.sols_scores}')
        logger.info(
            'LocalSearch Done with min: {:.01f} -> {:.01f}, mean: {:.01f} -> {:.01f})'.format(
                min_c_score, min_next_score, avg_c_score, avg_next_score))
        next_min_ind = next_scores.index(min(next_scores))
        return next_sols[next_min_ind], next_scores[next_min_ind]

    def run(self):
        best_score = np.inf
        self.best_record, self.iter_x, self.iter_y = [], [], []
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            if i % self.local_iter == 0 and i >= self.local_iter:
                if i >= 8 * self.local_iter:
                    tmp_best_one, tmp_best_score = self.local_search(swap_on=True)
                else:
                    tmp_best_one, tmp_best_score = self.local_search(swap_on=False)
            self.temp *= self.temp_dec_rate  # 降温
            '''
            if i > 5000:
                self.p_in_batch = 0.1
                self.p_be_batch = 0
                self.p_multi = 0.4
                self.p_cross_batch = 0
                self.forward_insert = 0.5
                self.temp = 500
            '''
            self.iter_x.append(i)
            self.iter_y.append(tmp_best_score)
            if tmp_best_score < best_score:
                best_score = tmp_best_score
                best_sol = tmp_best_one
                self.best_record.append(best_score)
                self.best_sol = best_sol
                self.best_score = best_score
            # 打印每一次结果
            logger.info(f"epoch: {i}, score:{best_score}")
        logger.warning(f'finished with best score: {best_score}')
        # return BEST_LIST, self.location[BEST_LIST], 1. / best_score

    def plot(self):

        plt.style.use('ggplot')
        plt.figure(dpi=300, figsize=(8, 5))

        plt.plot(self.iter_x, self.iter_y)
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.show()


if __name__ == '__main__':
    time_start = time.time()
    data = read_in('data.csv')
    problem = Problem(df=data, num_of_batch=4, iteration=5000, population=20)
    problem.run()
    problem.plot()
    time_end = time.time()
    time_sum = time_end - time_start
    sol_flag, batches = is_feasible_sol(problem.best_sol)
    logger.warning("time span: {:.2f}s".format(time_sum))
    logger.warning(f'batches size: {[len(batch) for batch in batches]}')
    logger.warning(f'batches length: {[sum([i[2] for i in batch]) for batch in batches]}')

    ordered_batches = []
    for batch in batches:
        ordered_batch = [problem.id[i[0]] for i in batch]
        ordered_batches.append(ordered_batch)

