import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import utils.callbacks
import utils.data
# import utils.email
import utils.logging
import torch
import models
import pandas as pd
import itertools
import matplotlib.pyplot as plt


DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    "ventura": {"feat": "TGCN/T-GCN-PyTorch/data/speed_few_T.csv", "adj": "TGCN/T-GCN-PyTorch/data/adjacent_matrix_G_dual_few.csv"},
}
dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS["ventura"]["feat"], adj_path=DATA_PATHS["ventura"]["adj"], 
        numlane_path = "TGCN/T-GCN-PyTorch/data/lane_few_feature.csv",
        speedlimit_path = "TGCN/T-GCN-PyTorch/data/speedlimit_few_feature.csv",
        poi_path="TGCN/T-GCN-PyTorch/data/selectPOI_feature_few.csv",
        temp_path = "TGCN/T-GCN-PyTorch/data/temp_feature.csv",
        precip_path = "TGCN/T-GCN-PyTorch/data/precip_feature.csv",
        wind_path = "TGCN/T-GCN-PyTorch/data/wind_feature.csv",
        humidity_path = "TGCN/T-GCN-PyTorch/data/humidity_feature.csv",        
        day0_path="TGCN/T-GCN-PyTorch/data/dayofweek_feature0.csv",
        day1_path="TGCN/T-GCN-PyTorch/data/dayofweek_feature1.csv",
        day2_path="TGCN/T-GCN-PyTorch/data/dayofweek_feature2.csv",
        day3_path="TGCN/T-GCN-PyTorch/data/dayofweek_feature3.csv",
        day4_path="TGCN/T-GCN-PyTorch/data/dayofweek_feature4.csv",
        day5_path="TGCN/T-GCN-PyTorch/data/dayofweek_feature5.csv",
        day6_path="TGCN/T-GCN-PyTorch/data/dayofweek_feature6.csv",
        hour_sin_path="TGCN/T-GCN-PyTorch/data/hourofday_feature_sin.csv", 
        hour_cos_path="TGCN/T-GCN-PyTorch/data/hourofday_feature_cos.csv",
        
    )

# Load trained model
model = models.TGCN(adj=dm.adj, hidden_dim=64)
checkpoint = torch.load('model/lightning_logs/version_16069916/checkpoints/epoch=75-step=4104.ckpt', map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
fixed_state_dict = {}
for key in state_dict:
    new_key = key.replace("model.", "")
    fixed_state_dict[new_key] = state_dict[key]
del fixed_state_dict['regressor.weight'] 
del fixed_state_dict['regressor.bias']
model.load_state_dict(fixed_state_dict) #del fixed_state_dict['regressor.weight'] del fixed_state_dict['regressor.bias']
# model.load_state_dict(checkpoint['state_dict'], strict=False)
train_dataset, val_dataset = utils.data.functions.generate_torch_datasets(dm._feat, dm._numlane, dm._speedlimit, dm._poi, dm._temp, dm._precip, dm._wind, dm._humidity, dm._day0, dm._day1, dm._day2, dm._day3, dm._day4, dm._day5,dm._day6,dm._hour_sin,dm._hour_cos,dm.seq_len, dm.pre_len)
# input = train_dataset[0][0]

# train_dataset[:][0][:,1] # traning data for node 2

# Input data point
data_point = train_dataset[2784][0]# size(1, 12,3169,17) # 96 #2784, thursday afternoon
data_point = data_point.reshape(1,12,3169,17)

node_index = 1800 #882/ 881 / 1034
# Set target outcome
target_speed_increase = 0.06


# add constraint list: index of the specific road
road_index_df= pd.read_csv("Data/ventura_road/ETOB_link_index.csv", header=None)
road_index = road_index_df[0].to_numpy()

# lanes_value = [1,2,3,4,5,6,7,8]
# parameter_combinations = list(itertools.product(lanes_value))

def objective_function(poi_data, lanes_data,speedlimit_data, orig_data, target_speed_increase):
    modified_data = orig_data.clone()
    modified_data[:,:,:,14][:,:,road_index]= poi_data
    modified_data[:,:,:,15][:,:,road_index]= lanes_data
    modified_data[:,:,:,16][:,:,road_index]= speedlimit_data

    # modified_data[:,:,:,14]  = poi_data # 15th poi
    # modified_data[:,:,:,15] = lanes_data # the 15th feature is number of lanes
    # modified_data[:,:,:,16] = speedlimit_data # the 15th feature is number of lanes
    # modified_data[:,:,:,1] = temp_data.unsqueeze(-1) # the 2nd feature is temperature#temp_data, day_of_week_data, 

    # apply softmax to ensure one-hot encoding rule for day_of_week_data
    # day_of_week_prob = F.softmax(day_of_week_data, dim = -1)
    # modified_data[:,:,:,3:10] = day_of_week_prob
    # modified_data[:,:,:,12:14] = hour_of_day

    # calculate the difference between the predicted speed and target speed
    pred_speed = model(modified_data)[node_index]#[:,0,:,0]
    orig_speed = model(orig_data)[node_index]#[:,0,:,0]
    # target_speed = orig_speed+target_speed_increase #*(1+target_speed_increase)
    target_speed= 0.3436 # 56/163
    outcome_diff = torch.abs(pred_speed - target_speed)

    # Calculate the similarity between the original input and the counterfactual
    poi_diff = torch.norm(poi_data[0,0] - orig_data[:,:,:,14][:,:,road_index][0,0])
    lanes_diff = torch.norm(lanes_data[0,0] - orig_data[:,:,:,15][:,:,road_index][0,0])
    speedlimit_diff = torch.norm(speedlimit_data[0,0] - orig_data[:,:,:,16][:,:,road_index][0,0])
    # lanes_diff = torch.norm(lanes_data[0,0] - orig_data[0,0,:,15])
    # speedlimit_diff = torch.norm(speedlimit_data[0,0] - orig_data[0,0,:,16])
    # temp_diff = torch.norm(temp_data - orig_data[:,:,0,1], dim=-1)
    # day_of_week_diff = torch.norm(day_of_week_prob - orig_data[:,:,:,3:10], p = 1, dim=-1)
    # hour_of_day_diff = torch.norm(hour_of_day - orig_data[:,:,:,12:14], dim=-1)

    # Define weights for the objective function
    w_outcome = 163.0
    w_poi = 0.001
    w_lanes = 0.001
    w_speedlimit = 0.001
    # w_hour_of_day = 0.001
    # w_temp = 1.0
    # w_day_of_week = 1.0
    # print(outcome_diff.mean())

    return w_outcome*outcome_diff.mean(),w_speedlimit*speedlimit_diff.mean(),w_poi*poi_diff.mean(),w_lanes*lanes_diff.mean() # w_hour_of_day*hour_of_day_diff.mean() + w_temp*temp_diff.mean() + w_day_of_week*day_of_week_diff

# Initialize the counterfactual input
original_poi = data_point[:,0,:,14]
original_lanes = data_point[:,0,:,15]
original_speedlimit = data_point[:,0,:,16]
# counterfactual_poi = original_poi[:,road_index]
counterfactual_data = data_point.clone().detach().requires_grad_(True)
counterfactual_poi = original_poi[:,road_index].clone().detach().requires_grad_(True)
counterfactual_lanes = original_lanes[:,road_index].clone().detach().requires_grad_(True)
counterfactual_speedlimit = original_speedlimit[:,road_index].clone().detach().requires_grad_(True)
# counterfactual_hour_of_day = data_point[:,:,0,12:14].clone().detach().requires_grad_(True)
# counterfactual_temp = data_point[:,:,0,1].clone().detach().requires_grad_(True)
# counterfactual_day_of_week = data_point[:,:,:,3:10].clone().detach().requires_grad_(True)

# Set the optimizer, counterfactual_lanes, counterfactual_speedlimit
optimizer = optim.Adam([counterfactual_speedlimit, counterfactual_poi, counterfactual_lanes], lr=0.01)#, counterfactual_lanes, counterfactual_temp, counterfactual_day_of_week

# Optimize the objective function ,counterfactual_hour_of_day.unsqueeze(2).repeat(1,1,3169,1)
n_iterations = 120000
losses= []
losses_speed = []
losses_spdlimit = []
losses_poi = []
losses_lanes = []
for i in range(n_iterations):
    optimizer.zero_grad()
    loss_speed, loss_spdlimit, loss_poi, loss_lanes = objective_function(counterfactual_poi.unsqueeze(1).repeat(1,12,1), counterfactual_lanes.unsqueeze(1).repeat(1,12,1),counterfactual_speedlimit.unsqueeze(1).repeat(1,12,1), data_point, target_speed_increase)
    loss = loss_speed + loss_spdlimit + loss_poi + loss_lanes
    # loss = objective_function(counterfactual_poi, counterfactual_lanes,counterfactual_speedlimit, data_point, target_speed_increase)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    losses_speed.append(loss_speed.item())
    losses_spdlimit.append(loss_spdlimit.item())
    losses_poi.append(loss_poi.item())
    losses_lanes.append(loss_lanes.item())

    # project counterfactual inputs onto the feasible set
    counterfactual_poi.data.clamp_(0, 30)
    counterfactual_poi.data = counterfactual_poi.data.round()   
    counterfactual_lanes.data.clamp_(1, 8)
    counterfactual_lanes.data = counterfactual_lanes.data.round()   
    counterfactual_speedlimit.data.clamp_(0, 120)
    # counterfactual_hour_of_day.data.clamp_(-1,1)
    # print(sum(sum(data_point[:,0,:,16]-counterfactual_speedlimit)))

    # print progress every 100 iterations
    if i%100 == 0:
        print(f"Iteration {i}/{n_iterations}: Loss = {loss.item()}")
# sbatch -n 1 --cpus-per-task=4 --gpus=rtx_3090:4 --gres=gpumem:24576m --time=4:00:00 --mem-per-cpu=10752 --wrap="python TGCN/T-GCN-PyTorch/main.py"
# sbatch -n 1 --cpus-per-task=16 --gpus=a100_80gb:4 --gres=gpumem:28672m --time=4:00:00 --mem-per-cpu=10752 --wrap="python TGCN/T-GCN-PyTorch/main.py"
# Analyze the counterfactual
plt.plot(losses, label = "all loss")
plt.plot(losses_speed, label = "speed loss")
plt.plot(losses_spdlimit, label = "spdlimit loss")
plt.plot(losses_poi, label = "poi loss")
plt.plot(losses_lanes, label = "lanes loss")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig("loss.png")
orig_prediction = model(data_point)#[:,0,:,0]
counterfactual_data = data_point.clone()
counterfactual_data[:,:,:,14][:,:,road_index]= counterfactual_poi.unsqueeze(1).repeat(1,12,1)
counterfactual_data[:,:,:,15][:,:,road_index]= counterfactual_lanes.unsqueeze(1).repeat(1,12,1)
counterfactual_data[:,:,:,16][:,:,road_index]= counterfactual_speedlimit.unsqueeze(1).repeat(1,12,1)

# counterfactual_data[:,:,:,12:14] = counterfactual_hour_of_day.unsqueeze(2).repeat(1,1,3169,1)
# counterfactual_data[:,:,:,1] = counterfactual_temp
cf_prediction = model(counterfactual_data)[node_index]#[:,0,:,0]
# pd.DataFrame(counterfactual_poi).to_csv('Data/counterfactual_poi.csv', index=False, header=False)
# pd.DataFrame(counterfactual_lanes).to_csv('Data/counterfactual_lanes.csv', index=False, header=False)
pd.DataFrame(counterfactual_speedlimit.cpu().detach().numpy()).to_csv('Data/t4_counterfactual_speedlimit.csv', index=False, header=False)
pd.DataFrame(counterfactual_poi.cpu().detach().numpy()).to_csv('Data/t4_counterfactual_poi.csv', index=False, header=False)
pd.DataFrame(counterfactual_lanes.cpu().detach().numpy()).to_csv('Data/t4_counterfactual_lanes.csv', index=False, header=False)
# pd.DataFrame(counterfactual_hour_of_day.cpu().detach().numpy()).to_csv('Data/t4_counterfactual_hour_of_day.csv', index=False, header=False)
print('Original prediction:', orig_prediction[node_index]*163)
print('Counterfactual prediction:', cf_prediction*163)

pd.DataFrame(cf_prediction).to_csv('Data/t1_cf_prediction.csv', index=False, header=False)


print('Original prediction:', orig_prediction[node_index])
print('Counterfactual prediction:', cf_prediction)
print('Counterfactual lanes:', counterfactual_lanes)
print('Difference in input features:', counterfactual_data - data_point)
print('Difference in number of poi:', counterfactual_poi - data_point[:,0,:,14])
# print('Difference in temperature:', counterfactual_temp.unsqueeze(-1) - data_point[:,:,:,1])
