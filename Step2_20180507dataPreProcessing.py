import pandas as pd #ipython notebook
import numpy as np
import multiprocessing as mp
import datetime
from multiprocessing import cpu_count
import time

# 功能：从全体样本中筛选出部分样本
# str_feature_type：用于筛选的特征值
def select_sample_has_label(df_all_sample, df_part_sample, str_feature_type):
    # 获取标签训练集的 ID list  即 vid 的 list
    list_vid_train_label = df_part_sample[str_feature_type].tolist()
    # 根据ID号，筛选出38199个样本，再组成一个新的 dataframe
    df_selected_via_vid = df_all_sample.loc[df_all_sample[str_feature_type].isin(list_vid_train_label)]
    # 之前的 index 还是断断续续的，扔掉之前的，重新从0开始排序
    df_selected_via_vid_index_resetted = df_selected_via_vid.reset_index(drop=True)
    return df_selected_via_vid_index_resetted


def select_feature_via_threshold(df_all_feature, num_threshold_feature):
    num_rows = df_all_feature.shape[0]
    num_columns = df_all_feature.shape[1]
    # 输入的数据集的特征名称列表
    list_feature_names = df_all_feature.columns.tolist()
    # 特征对应数据值超过阈值的特征名称 组成的列表
    list_feature_names_after_threshold = []
    for i in range(num_columns):
        if df_all_feature.iloc[:, i].count() >= num_threshold_feature:
            list_feature_names_after_threshold.append(list_feature_names[i])

    df_selected_via_feature_threshold = df_all_feature[list_feature_names_after_threshold]

    # 以下部分用于验证该方法是否正确，是否筛选出了单个特征值数据值超过阈值的 这些特征
    '''
    print("\n", "list_feature_names_after_threshold: ","\n", list_feature_names_after_threshold)
    list_temp = []
    for i in range(df_selected_via_feature_threshold.shape[1]):
        list_temp.append(df_selected_via_feature_threshold.iloc[:,i].count())
    int_flag = 0
    for i in range(len(list_temp)):
        if list_temp[i] < num_threshold_feature:
            int_temp += 1

    print("\n", "list_temp:","\n", list_temp)
    print("\n", "int_flag:","\n", int_flag)
    '''
    return df_selected_via_feature_threshold

def change_not_float_to_nan(check_str):
    try:
        float(check_str)
        str_c = float(check_str)
        return str_c
    except ValueError:
        check_str = np.nan
        return check_str


def dataset_change_not_float_to_nan(df_under):
    df_changed = df_under
    for i in range(df_under.shape[0]):
        # 这里'vid' 特征是 str类型，但是要保存'vid'特征，'vid'特征是第0列，所以要从1开始循环
        for j in range(1,df_under.shape[1]):
            df_changed.iat[i, j] = change_not_float_to_nan(df_under.loc[i].values[j])

     # 测试只处理10份数据效果，只用一次循环
    # 测试成功
    '''
    for j in range(df_under.shape[1]):
        print("\n", "j: ", j)
        df_changed.iat[0, j] = change_not_float_to_nan(df_under.loc[0].values[j])
    '''
    return df_changed

def dataset_change_not_float_to_nan_multicore(df_under_change):
    print("\n", "df_under_change.shape: ", "\n",df_under_change.shape)
    print("\n", "type(df_under_change): ", "\n", type(df_under_change))
    num_cores = cpu_count()
    num_partitions = 4
    pool = mp.Pool(num_cores)
    df_under_change_split = np.array_split(df_under_change,num_partitions)
    for i in range(num_partitions):
        df_under_change_split[i] = df_under_change_split[i].reset_index(drop=True)
    df_changed_multicore = pd.concat(pool.map(dataset_change_not_float_to_nan,df_under_change_split))

    pool.close()
    pool.join()
    return  df_changed_multicore


if __name__ == '__main__':
    dataset_train_feature = pd.read_csv("tmp.csv")
    dataset_train_label = pd.read_csv("meinian_round1_train_20180408.csv", encoding="gb2312")
    dataset_train_feature_selected_by_label = select_sample_has_label(dataset_train_feature, dataset_train_label, 'vid')
    dataset_train_feature_selected_by_threshold_first = select_feature_via_threshold(dataset_train_feature_selected_by_label, 5000)
    print("\n", "dataset_train_feature_selected_by_threshold_first.shape: ", "\n",dataset_train_feature_selected_by_threshold_first.shape)
    #print("\n", "dataset_train_feature_selected_by_threshold_first.head(5): ", "\n",dataset_train_feature_selected_by_threshold_first.head(5))


    # 处理全部数据
    #  耗时13mins 速度是单核运行的 4.7倍

    starttime = datetime.datetime.now()
    dataset_train_feature_all_float = dataset_change_not_float_to_nan_multicore(dataset_train_feature_selected_by_threshold_first)
    endtime = datetime.datetime.now()
    print("\n", "(endtime - starttime).seconds: ", "\n", (endtime - starttime).seconds)
    print("\n", "dataset_train_feature_all_float_1000.shape: ", "\n", dataset_train_feature_all_float.shape)
    print("\n", "dataset_train_feature_all_float_1000.head(5): ", "\n", dataset_train_feature_all_float.head(5))
    
    # 写入和读取数据，这里将全部转为 float 的数据 写入csv文件，然后再读取，显示特征值增加了一列，和原来的index 内容相同，如果是再读取 csv文件来处理，要删掉第一列
    
    dataset_train_feature_all_float.to_csv('dataset_train_feature_all_float.csv')
    dataset_train_feature_all_float_read = pd.read_csv('dataset_train_feature_all_float.csv')
    print("\n", "dataset_train_feature_all_float_read .shape: ", "\n",
          dataset_train_feature_all_float_read.shape)
    print("\n", "dataset_train_feature_all_float_read .head(5): ", "\n",
          dataset_train_feature_all_float_read.head(5))










    # 先处理前1000条，统计时间
    # 1000条计算耗时 19s
    '''
    starttime = datetime.datetime.now()
    dataset_train_feature_selected_by_threshold_first_100 = dataset_train_feature_selected_by_threshold_first.loc[0:99,:]
    dataset_train_feature_all_float_100 = dataset_change_not_float_to_nan_multicore( dataset_train_feature_selected_by_threshold_first_100)
    endtime = datetime.datetime.now()
    print("\n", "(endtime - starttime).seconds: ", "\n", (endtime - starttime).seconds)
    print("\n", "dataset_train_feature_all_float_100.head(5): ", "\n", dataset_train_feature_all_float_100.head(5))
    '''
    #dataset_train_feature_all_float = dataset_change_not_float_to_nan_multicore(dataset_train_feature_selected_by_threshold_first)
    # print("\n", "dataset_train_feature_all_float.head(5): ", "\n",dataset_train_feature_all_float.head(5))


    # 测试只处理10条数据效果，分成10份
    # 测试成功
    '''
    dataset_train_feature_selected_by_threshold_first_10 = dataset_train_feature_selected_by_threshold_first.loc[0:9,:]
    print("\n", "dataset_train_feature_selected_by_threshold_first_10 .loc[0].values[0]: ", "\n",
          dataset_train_feature_selected_by_threshold_first_10.loc[0].values[0])
    dataset_train_feature_all_float_10 = dataset_change_not_float_to_nan_multicore( dataset_train_feature_selected_by_threshold_first_10)
    print("\n", "dataset_train_feature_all_float_10: ", "\n", dataset_train_feature_all_float_10)
    '''
    # 测试 change_not_float_to_nan 的功能，转换一个数据
    # 测试显示 功能正常
    '''
    print("\n", "dataset_train_feature_selected_by_threshold_first.loc[1].values[0]: ", "\n",
          dataset_train_feature_selected_by_threshold_first.loc[1].values[0])
    data_changed = change_not_float_to_nan(dataset_train_feature_selected_by_threshold_first.loc[1].values[0])
    print("\n", "data_changed: ", "\n",
          data_changed)

    # 测试 dataset_change_not_float_to_nan 不用多核的情况测试
    # 测试10条数据
    # 测试显示功能正常
    dataset_train_feature_selected_by_threshold_first_10 = dataset_train_feature_selected_by_threshold_first.loc[0:10,:]
    df_10_changed= dataset_change_not_float_to_nan(dataset_train_feature_selected_by_threshold_first_10)
    print("\n", "df_10_changed: ", "\n",df_10_changed)
    '''