import numpy as np
import pandas as pd
import torch


def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path, header=None)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path,  header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def generate_dataset(
    data, numlane,speedlimit,poi,temp,precip,wind,humidity,day0,day1,day2,day3,day4,day5,day6,hour_sin,hour_cos, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param weather: weather data
    :param calender: calender data, weekday/weekend
    :param poi: poi data, weekday/weekend
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    # all_data = np.stack((data,weather,calender,poi),axis = 2)
    # weather_nor = weather/np.max(np.max(weather))
    temp_nor = temp/np.max(np.max(temp))
    precip_nor = precip/np.max(np.max(precip))
    wind_nor = wind/np.max(np.max(wind))
    humidity_nor = humidity/np.max(np.max(humidity))
    all_data = np.stack((data,temp_nor,precip_nor,wind_nor,humidity_nor, day0, day1, day2, day3,day4,day5,day6,hour_sin, hour_cos, poi,numlane, speedlimit),axis = 2)
    # all_data = np.stack((data,temp_nor,numlane, speedlimit, poi),axis = 2)
    train_data = all_data[:train_size] # TODO, change to match higher dimension data
    test_data = all_data[train_size:time_len] # change
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        a1 = train_data[i: i + seq_len + pre_len]
        # a2 = weather_nor[i: i + seq_len + pre_len]
        # a2 = temp_nor[i: i + seq_len + pre_len]
        # a3 = precip_nor[i: i + seq_len + pre_len]
        # a4 = wind_nor[i: i + seq_len + pre_len]
        # a5 = humidity_nor[i: i + seq_len + pre_len]
        # a6 = day[i: i + seq_len + pre_len,:3169]
        # a7 = hour[i: i + seq_len + pre_len]
        # a = np.stack((a1[0:seq_len],a2[0: seq_len]),axis = 2)
        # a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len],a3[0: seq_len + pre_len],a4[0: seq_len + pre_len],
        #     a5[0: seq_len + pre_len],a6[0: seq_len + pre_len],a7[0: seq_len + pre_len],numlane[:1],speedlimit[:1],poi[:1]))
        # a = a1[0:seq_len]
        a = a1[0:seq_len]
        train_X.append(a)
        train_Y.append(a1[seq_len : seq_len + pre_len])
        # train_X.append(np.array(train_data[i : i + seq_len]))
        # train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len,:,0]))
    for i in range(len(test_data) - seq_len - pre_len):
        b1 = test_data[i: i + seq_len + pre_len]
        # b2 = weather_nor[i: i + seq_len + pre_len]
        # b2 = temp_nor[i: i + seq_len + pre_len]
        # b3 = precip_nor[i: i + seq_len + pre_len]
        # b4 = wind_nor[i: i + seq_len + pre_len]
        # b5 = humidity_nor[i: i + seq_len + pre_len]
        # b6 = day[i: i + seq_len + pre_len,:3169]
        # b7 = hour[i: i + seq_len + pre_len]
        # b = np.stack((b1[0:seq_len],b2[0: seq_len]),axis = 2)
        # b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len],b3[0: seq_len + pre_len],b4[0: seq_len + pre_len],
        #     b5[0: seq_len + pre_len],b6[0: seq_len + pre_len],b7[0: seq_len + pre_len],numlane[:1],speedlimit[:1],poi[:1]))
        # b = b1[0:seq_len]
        b = b1[0:seq_len]
        test_X.append(b)
        test_Y.append(b1[seq_len : seq_len + pre_len])
        # test_X.append(np.array(test_data[i : i + seq_len]))
        # test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len,:,0]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, numlane,speedlimit,poi,temp,precip,wind,humidity,day0,day1,day2,day3,day4,day5,day6,hour_sin,hour_cos,seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data, numlane,speedlimit,poi,temp,precip,wind,humidity,day0,day1,day2,day3,day4,day5,day6,hour_sin,hour_cos,
        # weather,
        # calender,
        # poi, change
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
