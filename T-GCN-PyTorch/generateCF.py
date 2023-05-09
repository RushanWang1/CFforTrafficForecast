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
data_point = train_dataset[96][0]# size(1, 12,3169,17)
data_point = data_point.reshape(1,12,3169,17)

node_index = 882
# Set target outcome
target_speed_increase = 0.2

def objective_function(poi_data, lanes_data,speedlimit_data, orig_data, target_speed_increase):
    modified_data = orig_data.clone()
    modified_data[:,:,:,14]  = poi_data # 15th poi
    modified_data[:,:,:,15] = lanes_data # the 15th feature is number of lanes
    modified_data[:,:,:,16] = speedlimit_data # the 15th feature is number of lanes
    # modified_data[:,:,:,1] = temp_data.unsqueeze(-1) # the 2nd feature is temperature#temp_data, day_of_week_data, 

    # apply softmax to ensure one-hot encoding rule for day_of_week_data
    # day_of_week_prob = F.softmax(day_of_week_data, dim = -1)
    # modified_data[:,:,:,3:10] = day_of_week_prob

    # calculate the difference between the predicted speed and target speed
    pred_speed = model(modified_data)[node_index]#[:,0,:,0]
    orig_speed = model(orig_data)[node_index]#[:,0,:,0]
    target_speed = orig_speed*(1+target_speed_increase)
    outcome_diff = torch.abs(pred_speed - target_speed)

    # Calculate the similarity between the original input and the counterfactual
    poi_diff = torch.norm(poi_data[0,0] - orig_data[0,0,:,14])
    lanes_diff = torch.norm(lanes_data[0,0] - orig_data[0,0,:,15])
    speedlimit_diff = torch.norm(speedlimit_data[0,0] - orig_data[0,0,:,16])
    # temp_diff = torch.norm(temp_data - orig_data[:,:,0,1], dim=-1)
    # day_of_week_diff = torch.norm(day_of_week_prob - orig_data[:,:,:,3:10], p = 1, dim=-1)

    # Define weights for the objective function
    w_outcome = 100.0
    w_poi = 0.001
    w_lanes = 0.001
    w_speedlimit = 0.001
    # w_temp = 1.0
    # w_day_of_week = 1.0

    return w_outcome*outcome_diff.mean() + w_poi*poi_diff.mean() #+ w_lanes*lanes_diff.mean() + w_speedlimit*speedlimit_diff.mean() # + w_temp*temp_diff.mean() + w_day_of_week*day_of_week_diff

# Initialize the counterfactual input
counterfactual_data = data_point.clone().detach().requires_grad_(True)
counterfactual_poi = data_point[:,0,:,14].clone().detach().requires_grad_(True)
counterfactual_lanes = data_point[:,0,:,15].clone().detach().requires_grad_(True)
counterfactual_speedlimit = data_point[:,0,:,16].clone().detach().requires_grad_(True)
# counterfactual_temp = data_point[:,:,0,1].clone().detach().requires_grad_(True)
# counterfactual_day_of_week = data_point[:,:,:,3:10].clone().detach().requires_grad_(True)

# Set the optimizer, counterfactual_lanes, counterfactual_speedlimit
optimizer = optim.Adam([counterfactual_poi], lr=0.0001)#, counterfactual_lanes, counterfactual_temp, counterfactual_day_of_week

# Optimize the objective function
n_iterations = 100
for i in range(n_iterations):
    optimizer.zero_grad()
    loss = objective_function(counterfactual_poi.unsqueeze(1).repeat(1,12,1), counterfactual_lanes.unsqueeze(1).repeat(1,12,1),counterfactual_speedlimit.unsqueeze(1).repeat(1,12,1), data_point, target_speed_increase)
    loss.backward()
    optimizer.step()

    # project counterfactual inputs onto the feasible set
    counterfactual_poi.data.clamp_(0, 30)
    counterfactual_poi.data = counterfactual_poi.data.round()   
    # counterfactual_lanes.data.clamp_(1, 8)
    # counterfactual_lanes.data = counterfactual_lanes.data.round()   
    # counterfactual_speedlimit.data.clamp_(0, 120)

    # print progress every 100 iterations
    if i%1 == 0:
        print(f"Iteration {i}/{n_iterations}: Loss = {loss.item()}")

# Analyze the counterfactual
orig_prediction = model(data_point)#[:,0,:,0]
counterfactual_data = data_point.clone()
counterfactual_data[:,:,:,14] = counterfactual_poi.unsqueeze(1).repeat(1,12,1)
counterfactual_data[:,:,:,15] = counterfactual_lanes.unsqueeze(1).repeat(1,12,1)
counterfactual_data[:,:,:,16] = counterfactual_speedlimit.unsqueeze(1).repeat(1,12,1)
# counterfactual_data[:,:,:,1] = counterfactual_temp
cf_prediction = model(counterfactual_data)[node_index]#[:,0,:,0]
pd.DataFrame(counterfactual_poi).to_csv('Data/counterfactual_poi.csv', index=False, header=False)
pd.DataFrame(counterfactual_lanes).to_csv('Data/counterfactual_lanes.csv', index=False, header=False)
pd.DataFrame(counterfactual_speedlimit).to_csv('Data/counterfactual_speedlimit.csv', index=False, header=False)
pd.DataFrame(cf_prediction).to_csv('Data/cf_prediction.csv', index=False, header=False)

print('Original prediction:', orig_prediction[node_index])
print('Counterfactual prediction:', cf_prediction)
print('Counterfactual lanes:', counterfactual_lanes)
print('Difference in input features:', counterfactual_data - data_point)
print('Difference in number of lanes:', counterfactual_lanes - data_point[:,0,:,14])
# print('Difference in temperature:', counterfactual_temp.unsqueeze(-1) - data_point[:,:,:,1])

