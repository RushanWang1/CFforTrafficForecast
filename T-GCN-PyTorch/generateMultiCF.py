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
from deap import base,creator, tools, algorithms
from sklearn.neighbors import NearestNeighbors


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
data_point = train_dataset[2772][0]# size(1, 12,3169,17) # 96 #2784-12, thursday afternoon
data_point = data_point.reshape(1,12,3169,17)

node_index = 1800 #882/ 881 / 1034
# Set target outcome
target_speed_increase = 0.06


# add constraint list: index of the specific road
road_index_df= pd.read_csv("Data/ventura_road/ETOB_link_index.csv", header=None)
road_index = road_index_df[0].to_numpy()

n_nodes = road_index.shape[0]
# lanes_value = [1,2,3,4,5,6,7,8]
# parameter_combinations = list(itertools.product(lanes_value))

training_samples = data_point[0,0,:,14:17]

def l2_distance(v1,v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))

def get_k_nearest(individual, k=5):
    training_array = np.array(training_samples)
    individual_array = np.array(individual).reshape(1,-1)

    nbrs = NearestNeighbors(n_neighbors = k, algorithm='ball_tree').fit(training_array)
    _, indices = nbrs.kneighbors(individual_array)

    return [training_samples[i] for i in indices[0]]

def calculate_plausibility(cf_point, original_samples, k):
    distances = []
    # for point in counterfactual
    for point in cf_point:
        point_distances=[l2_distance(point, sample) for sample in original_samples]
        closest_samples = np.partition(point_distances, k)[:k]
        avg_distance = np.mean(closest_samples)
        distances.append(avg_distance)
    overall_avg_distance = np.mean(distances)
    return overall_avg_distance

def objective_function(poi_data, lanes_data,speedlimit_data, orig_data, target_speed_increase, individual):
    modified_data = orig_data.clone()
    # modified_data[:,:,:,14]= torch.from_numpy(poi_data).float()
    # modified_data[:,:,:,15]= torch.from_numpy(lanes_data).float()
    # modified_data[:,:,:,16]= torch.from_numpy(speedlimit_data).float()
    modified_data[:,:,:,14][:,:,road_index]= torch.from_numpy(poi_data).float()
    modified_data[:,:,:,15][:,:,road_index]= torch.from_numpy(lanes_data).float()
    modified_data[:,:,:,16][:,:,road_index]= torch.from_numpy(speedlimit_data).float()

    # calculate the difference between the predicted speed and target speed
    pred_speed = model(modified_data)[node_index]#[:,0,:,0]
    orig_speed = model(orig_data)[node_index]#[:,0,:,0]
    # target_speed = orig_speed+target_speed_increase #*(1+target_speed_increase)
    target_speed= 0.3436 # 56/163
    outcome_diff = torch.abs(pred_speed - target_speed)

    # Calculate the similarity between the original input and the counterfactual
    poi_diff = torch.linalg.norm(torch.from_numpy(poi_data[0,0]).float()- orig_data[:,:,:,14][:,:,road_index][0,0])
    lanes_diff = torch.linalg.norm(torch.from_numpy(lanes_data[0,0]).float() - orig_data[:,:,:,15][:,:,road_index][0,0])
    speedlimit_diff = torch.linalg.norm(torch.from_numpy(speedlimit_data[0,0]).float() - orig_data[:,:,:,16][:,:,road_index][0,0])
    # poi_diff = torch.norm(torch.from_numpy(poi_data[0,0]).float()- orig_data[:,:,:,14][0,0])
    # lanes_diff = torch.norm(torch.from_numpy(lanes_data[0,0]).float() - orig_data[:,:,:,15][0,0])
    # speedlimit_diff = torch.norm(torch.from_numpy(speedlimit_data[0,0]).float() - orig_data[:,:,:,16][0,0])
    # lanes_diff = torch.norm(lanes_data[0,0] - orig_data[0,0,:,15])
    # speedlimit_diff = torch.norm(speedlimit_data[0,0] - orig_data[0,0,:,16])
    # temp_diff = torch.norm(temp_data - orig_data[:,:,0,1], dim=-1)
    # day_of_week_diff = torch.norm(day_of_week_prob - orig_data[:,:,:,3:10], p = 1, dim=-1)
    # hour_of_day_diff = torch.norm(hour_of_day - orig_data[:,:,:,12:14], dim=-1)

    # sparsity
    # sparsity = sum([1 for i in range(n_nodes) if (poi_data[0,0,i]!= original_poi[0,road_index[i]] or lanes_data[0,0,i]!=original_lanes[0,road_index[i]] or speedlimit_data[0,0,i]!=original_speedlimit[0,road_index[i]])])
    sparsity_poi = sum([1 for i in range(n_nodes) if (poi_data[0,0,i]!= original_poi[0,road_index[i]])])
    sparsity_lanes = sum([1 for i in range(n_nodes) if (lanes_data[0,0,i]!=original_lanes[0,road_index[i]])])
    sparsity_speedlimit = sum([1 for i in range(n_nodes) if (speedlimit_data[0,0,i]!=original_speedlimit[0,road_index[i]])])
    sparsity = sparsity_poi + sparsity_lanes + sparsity_speedlimit
    # plausibility
    cf_points = np.array(individual).reshape(3,197).T
    plausibility = calculate_plausibility(cf_points, training_samples, 3)
    # nearest_samples = get_k_nearest(individual)
    # plausibility = sum(l2_distance(individual, sample) for sample in nearest_samples)/len(nearest_samples)

    # Define weights for the objective function
    w_outcome = 163.0
    w_poi = 0.01
    w_lanes = 0.01
    w_speedlimit = 0.01
    w_sparsity = 1.0
    w_plausibility = 1.0

    return (w_outcome*outcome_diff.mean().item(), 
            w_speedlimit*speedlimit_diff.mean().item()+w_poi*poi_diff.mean().item()+w_lanes*lanes_diff.mean().item(),
            w_sparsity*sparsity,
            w_plausibility*plausibility
            #TODO: plausibility ,
            )
    # return w_outcome*outcome_diff.mean().detach().numpy(),w_speedlimit*speedlimit_diff.mean().detach().numpy(),w_poi*poi_diff.mean().detach().numpy(),w_lanes*lanes_diff.mean().detach().numpy() # w_hour_of_day*hour_of_day_diff.mean() + w_temp*temp_diff.mean() + w_day_of_week*day_of_week_diff

# Initialize the counterfactual input
original_poi = data_point[:,0,:,14]
original_lanes = data_point[:,0,:,15]
original_speedlimit = data_point[:,0,:,16]
# counterfactual_poi = original_poi[:,road_index]
counterfactual_data = data_point#.clone().detach().requires_grad_(True)
counterfactual_poi = original_poi[:,road_index].clone().detach().requires_grad_(True)
counterfactual_lanes = original_lanes[:,road_index].clone().detach().requires_grad_(True)
counterfactual_speedlimit = original_speedlimit[:,road_index].clone().detach().requires_grad_(True)

####################### Implement NSGA-II optimization #########################
# creator.create("FitnessMulti", base.Fitness, weight = (-1,-1,-1,-1)) # 4 objectives in total TODO
creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0, -1.0,-1.0))
creator.create("Individual", list, fitness = creator.FitnessMulti)
toolbox = base.Toolbox()

toolbox.register("poi_init", np.random.randint, 0, 36)
toolbox.register("lanes_init", np.random.randint, 1, 6)
toolbox.register("speedlimit_init", np.random.uniform, 50, 110)

def init_individual(poi_init, lanes_init, speedlimit_init, n_nodes):
    # Return a single individual, which is a list starting with the lane and speedlimit values, followed by POI values
    return creator.Individual([lanes_init() for _ in range(n_nodes)] + [speedlimit_init() for _ in range(n_nodes)]+ [poi_init() for _ in range(n_nodes)])

n_nodes = 197  
toolbox.register("init_individual", init_individual, toolbox.poi_init, toolbox.lanes_init, toolbox.speedlimit_init, n_nodes)
toolbox.register("individual", init_individual, toolbox.poi_init, toolbox.lanes_init, toolbox.speedlimit_init, n_nodes)
toolbox.register("population", tools.initRepeat,list, toolbox.individual)
# def evaluate(individual):
#     counterfactual_poi_1, counterfactual_lanes_1, counterfactual_speedlimit_1 = individual[::3], individual[1::3], individual[2::3]
#     input_poi = np.repeat(np.array(counterfactual_poi_1).reshape(1,197)[:, np.newaxis,:], 12, axis=1)
#     input_lanes = np.repeat(np.array(counterfactual_lanes_1).reshape(1,197)[:, np.newaxis,:], 12, axis=1)
#     input_speedlimit = np.repeat(np.array(counterfactual_speedlimit_1).reshape(1,197)[:, np.newaxis,:], 12, axis=1)
#     return objective_function(input_poi,input_lanes,input_speedlimit, data_point, target_speed_increase)
def evaluate(individual):
    counterfactual_lanes = individual[:n_nodes]
    counterfactual_speedlimit = individual[n_nodes:2*n_nodes]
    counterfactual_poi = individual[2*n_nodes:]    
    input_poi = np.repeat(np.array(counterfactual_poi).reshape(1,197)[:, np.newaxis,:], 12, axis=1)
    input_lanes = np.repeat(np.array(counterfactual_lanes).reshape(1,197)[:, np.newaxis,:], 12, axis=1)
    input_speedlimit = np.repeat(np.array(counterfactual_speedlimit).reshape(1,197)[:, np.newaxis,:], 12, axis=1)    
    return objective_function(input_poi,input_lanes,input_speedlimit, data_point, target_speed_increase, individual)

toolbox.register("evaluate", evaluate)
toolbox.register("gaussian_mutation", np.random.normal)

# def mutate(individual):
#     sigma_lanes = 3 # Define your own standard deviation
#     sigma_speedlimit = 30 # Define your own standard deviation
#     sigma_poi = 5 # Define your own standard deviation
    
#     # Mutate the common lanes and speed limit attributes
#     if np.random.random() < 0.1:  # 10% chance to mutate
#         individual[0] += toolbox.gaussian_mutation(0, sigma_lanes)
#         individual[0] = int(round(min(max(individual[0], 1), 6))) # Ensures the lanes value is within the expected range

#         individual[1] += toolbox.gaussian_mutation(0, sigma_speedlimit)
#         individual[1] = min(max(individual[1], 50), 110) # Ensures the speedlimit value is within the expected range

#     # Mutate the poi of each node
#     for i in range(2, len(individual)):  # skip the first two elements (lanes and speedlimit)
#         if np.random.random() < 0.1:  # 10% chance to mutate each poi
#             individual[i] += toolbox.gaussian_mutation(0, sigma_poi)
#             individual[i] = int(round(min(max(individual[i], 0), 30))) # Ensures the poi value is within the expected range

#     return individual,
def mutate(individual):
    sigma_lanes = 3 # Define your own standard deviation
    sigma_speedlimit = 30 # Define your own standard deviation
    sigma_poi = 5 # Define your own standard deviation
    # Mutate the lanes, speedlimit, and poi of each node
    for i in range(len(individual)):
        if np.random.random() < 0.1:  # 10% chance to mutate each attribute
            individual[i] += toolbox.gaussian_mutation(0, sigma_lanes if i<n_nodes else sigma_speedlimit if i<2*n_nodes else sigma_poi)
            if i<n_nodes:
                individual[i] = int(round(min(max(individual[i], 1),10)))
            elif i<2*n_nodes: 
                individual[i] = min(max(individual[i], 1),120) 
            else:
                individual[i] = int(round(min(max(individual[i], 1),36)))
    return individual,

# def mate(ind1, ind2):
#     if np.random.random() < 0.5:  # 50% chance to exchange common lanes and speed limit
#         ind1[0], ind2[0] = ind2[0], ind1[0]  # exchange lanes
#         ind1[1], ind2[1] = ind2[1], ind1[1]  # exchange speed limit

#     for i in range(2, len(ind1)):  # iterate over each poi
#         if np.random.random() < 0.5:  # 50% chance to exchange each poi
#             ind1[i], ind2[i] = ind2[i], ind1[i]

#     return ind1, ind2

def mate(ind1, ind2):
    for i in range(len(ind1)):
        if np.random.random() < 0.5:  # 50% chance to cross each attribute
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)

POP_SIZE=100
NGEN=100

pop=toolbox.population(n=POP_SIZE)

result_pop = algorithms.eaMuPlusLambda(pop, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, cxpb=0.5, mutpb=0.2, ngen=NGEN, verbose=True)

# pareto_front = tools.sortNondominated(result_pop, len(result_pop))[0]
pop = result_pop[0]

# Making sure all individuals in population have valid fitness
for ind in pop:
    if not ind.fitness.valid:
        ind.fitness.values = toolbox.evaluate(ind)

# get the Pareto front
pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)

import matplotlib.pyplot as plt
import pandas as pd

# Convert the Pareto front to a DataFrame for convenience
pareto_front_df = pd.DataFrame([ind.fitness.values for ind in pareto_front[0]])

# Save the Pareto front to a CSV file
pareto_front_df.to_csv('Data/moc/pareto_front_t6.csv', index=False)
pd.DataFrame(pareto_front[0]).to_csv('Data/moc/individual_t6.csv', index=False)
# pd.DataFrame(np.asarray(individual)).to_csv("Data/MOC_1_result.csv", header = False, index = False)
# Pairwise plot
objective_names = ['obj1', 'obj2', 'obj3', 'obj4']  # Replace with your objective function names
for i in range(4):
    for j in range(i+1, 4):
        plt.figure(figsize=(10, 7))
        plt.scatter(pareto_front_df[i], pareto_front_df[j])
        plt.xlabel(objective_names[i])
        plt.ylabel(objective_names[j])
        plt.title(f'Objective {objective_names[i]} vs {objective_names[j]}')
        plt.grid(True)
        plt.show()

print("\nCounterfactual Explanations:")
print("--------------------------------")

for ind in pareto_front:
    poi, lanes, speedlimit = ind[0],ind[1],ind[2]
    
    print(f"POI: {poi}, Lanes: {lanes}, Speed Limit: {speedlimit}")

# poi_values = [ind[0] for ind in pareto_front]
# lanes_values = [ind[1] for ind in pareto_front]
# speedlimit_values = [ind[2] for ind in pareto_front]

# plt.figure(figsize = (10,10))
# plt.plot(poi_values, 'b-', marker='o', label='POI')
# plt.plot(lanes_values, 'r-', marker='o', label='Lanes')
# plt.plot(speedlimit_values, 'g-', marker='o', label='Speed Limit')

# plt.title('Counterfactual Explanations')
# plt.xlabel('Individual')
# plt.ylabel('Values')
# plt.legend()
# plt.show()


'''
# Set the optimizer, counterfactual_lanes, counterfactual_speedlimit
optimizer = optim.Adam([counterfactual_speedlimit, counterfactual_poi, counterfactual_lanes], lr=0.01)#, counterfactual_lanes, counterfactual_temp, counterfactual_day_of_week

# Optimize the objective function ,counterfactual_hour_of_day.unsqueeze(2).repeat(1,1,3169,1)
n_iterations = 20000
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
plt.show()
orig_prediction = model(data_point)#[:,0,:,0]
counterfactual_data = data_point.clone()
counterfactual_data[:,:,:,14] = counterfactual_poi.unsqueeze(1).repeat(1,12,1)
counterfactual_data[:,:,:,15] = counterfactual_lanes.unsqueeze(1).repeat(1,12,1)
counterfactual_data[:,:,:,16] = counterfactual_speedlimit.unsqueeze(1).repeat(1,12,1)
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
'''
