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
        parking_path="TGCN/T-GCN-PyTorch/data/parking_feature_few.csv",
        gas_path="TGCN/T-GCN-PyTorch/data/gas_feature_few.csv",
        restaurant_path="TGCN/T-GCN-PyTorch/data/restaurant_feature_few.csv",
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
# model\lightning_logs\version_21139365\events.out.tfevents.1689154149.eu-g4-011.98892.0
model = models.TGCN(adj=dm.adj, hidden_dim=64)
checkpoint = torch.load('model/lightning_logs/version_19609064/checkpoints/epoch=1203-step=65016.ckpt', map_location=torch.device('cpu'))
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

# Input data point 8,12,15,18
hour = 8
data_point = train_dataset[2592+12*hour][0]# size(1, 12,3169,17) # 96 #2784-12, thursday afternoon //2664//2676(7am)/2748
data_point = data_point.reshape(1,12,3169,17)

node_index = 2185#882/ 881 / 1034 /1800/2185 
# Set target outcome
target_speed_increase = 0.06


# add constraint list: index of the specific road
road_index_df= pd.read_csv("Data/ventura_road/moorparkrd_west_56.csv", header = None)
# road_index_df= pd.read_csv("Data/ventura_road/spd_72_north_index.csv", header = None)
road_index = road_index_df.to_numpy()

n_nodes = road_index.shape[0]
road_index = road_index.reshape(n_nodes)
# lanes_value = [1,2,3,4,5,6,7,8]
# parameter_combinations = list(itertools.product(lanes_value))

training_samples = data_point[0,0,:,14:17]

# individual_df = pd.read_csv('Data/moc/individual_t15.csv')
# individual_value = individual_df.iloc[53].to_numpy()

##########################################
# import shap

# def model_wrapper(x):
#     # x = torch.Tensor(x).to(device)
#     model.eval()
#     with torch.no_grad():
#         output=model(x)[node_index]*163
#     return output.cpu().numpy()

# explainer = shap.DeepExplainer(model, data_point)
# shap_values = explainer.shap_values(data_point)

# shap.summary_plot(shap_values, data_point[:,:,node_index,:])
# ##########################################

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

def objective_function(cf_poi, cf_lanes, poi_data, lanes_data,speedlimit_data, orig_data, target_speed_increase, individual):
    modified_data = orig_data.clone()
    modified_data[:,:,:,14][:,:,road_index]= torch.from_numpy(poi_data).float()
    modified_data[:,:,:,15][:,:,road_index]= torch.from_numpy(lanes_data).float()
    modified_data[:,:,:,16][:,:,road_index]= torch.from_numpy(speedlimit_data).float()

    # calculate the difference between the predicted speed and target speed
    # pred_speed = model(modified_data)[node_index]#[:,0,:,0]
    orig_speed = model(orig_data)[node_index]#[:,0,:,0]
    pred_speed = model(modified_data)[node_index]#[:,0,:,0]
    # target_speed = orig_speed+target_speed_increase #*(1+target_speed_increase)
    target_speed= 56/163 # 72/163 => 56/163
    outcome_diff = torch.abs(pred_speed - target_speed)
    outcome_diff = outcome_diff.mean()

    # Calculate the similarity between the original input and the counterfactual
    poi_diff = torch.linalg.norm(torch.from_numpy(poi_data[0,0]).float()- orig_data[:,:,:,14][:,:,road_index][0,0])
    lanes_diff = torch.linalg.norm(torch.from_numpy(lanes_data[0,0]).float() - orig_data[:,:,:,15][:,:,road_index][0,0])
    speedlimit_diff = torch.linalg.norm(torch.from_numpy(speedlimit_data[0,0]).float() - orig_data[:,:,:,16][:,:,road_index][0,0])
    sparsity_poi = sum([1 for i in range(n_nodes) if (poi_data[0,0,i]!= original_poi[0,road_index[i]])])
    sparsity_lanes = sum([1 for i in range(n_nodes) if (lanes_data[0,0,i]!=original_lanes[0,road_index[i]])])
    sparsity_speedlimit = sum([1 for i in range(n_nodes) if (speedlimit_data[0,0,i]!=original_speedlimit[0,road_index[i]])])
    sparsity = sparsity_poi + sparsity_lanes + sparsity_speedlimit
    # plausibility
    # need poi, lanes, spdlimit, now spdlimit, lanes, poi
    new_individual= np.concatenate((cf_poi,cf_lanes,np.repeat(individual[0],n_nodes)),axis = 0)
    cf_points = np.array(new_individual).reshape(3,n_nodes).T
    plausibility = calculate_plausibility(cf_points, training_samples, 3)
    # nearest_samples = get_k_nearest(individual)
    # plausibility = sum(l2_distance(individual, sample) for sample in nearest_samples)/len(nearest_samples)

    # Define weights for the objective function
    w_outcome = 163.0
    w_poi = 0.01
    w_lanes = 0.01
    w_speedlimit = 0.01
    w_sparsity = 0.01
    w_plausibility = 0.01

    return (w_outcome*outcome_diff.item(), 
            w_speedlimit*speedlimit_diff.mean().item()+w_poi*poi_diff.mean().item()+w_lanes*lanes_diff.mean().item(),
            w_sparsity*sparsity,
            w_plausibility*plausibility
            )
    # return w_outcome*outcome_diff.mean().detach().numpy(),w_speedlimit*speedlimit_diff.mean().detach().numpy(),w_poi*poi_diff.mean().detach().numpy(),w_lanes*lanes_diff.mean().detach().numpy() # w_hour_of_day*hour_of_day_diff.mean() + w_temp*temp_diff.mean() + w_day_of_week*day_of_week_diff

# Initialize the counterfactual input
original_poi = data_point[:,0,:,14]
original_lanes = data_point[:,0,:,15]
original_speedlimit = data_point[:,0,:,16]
# original_temperature = data_point[:,:,0,1]
# counterfactual_poi = original_poi[:,road_index]
counterfactual_data = data_point#.clone().detach().requires_grad_(True)
counterfactual_poi = original_poi[:,road_index].clone().detach().requires_grad_(True)
counterfactual_lanes = original_lanes[:,road_index].clone().detach().requires_grad_(True)
counterfactual_speedlimit = original_speedlimit[:,road_index].clone().detach().requires_grad_(True)
# counterfactual_temperature = original_temperature[:,road_index].clone().detach().requires_grad_(True)

####################### Implement NSGA-II optimization #########################
# creator.create("FitnessMulti", base.Fitness, weight = (-1,-1,-1,-1)) # 4 objectives in total TODO
creator.create("FitnessMulti", base.Fitness, weights=(-10.0,-1.0, -1.0,-1.0))
creator.create("Individual", list, fitness = creator.FitnessMulti)
toolbox = base.Toolbox()

toolbox.register("poi_init", np.random.randint, 0, 40)
toolbox.register("lanes_init", np.random.randint, 1, 6)
toolbox.register("speedlimit_init", np.random.uniform, 50, 110)

def init_individual(poi_init, lanes_init, speedlimit_init, n_nodes):
    # Return a single individual, which is a list starting with the lane and speedlimit values, followed by POI values
    speedlimit = speedlimit_init()
    lanes = [lanes_init() for _ in range(n_nodes)]
    pois = [poi_init() for _ in range(n_nodes)]
    return creator.Individual([speedlimit]+[value for pair in zip(pois,lanes) for value in pair])
    # return creator.Individual([lanes_init() for _ in range(n_nodes)] + [speedlimit_init() for _ in range(n_nodes)] + [poi_init() for _ in range(n_nodes)])

 
toolbox.register("init_individual", init_individual, toolbox.poi_init, toolbox.lanes_init, toolbox.speedlimit_init, n_nodes)
toolbox.register("individual", init_individual, toolbox.poi_init, toolbox.lanes_init, toolbox.speedlimit_init, n_nodes)
toolbox.register("population", tools.initRepeat,list, toolbox.individual)
def evaluate(individual):
    # individual = individual_value
    # counterfactual_lanes = individual[:n_nodes]
    counterfactual_speedlimit = individual[0]
    lanes_pois = individual[1:]
    counterfactual_poi = lanes_pois[::2]
    counterfactual_lanes = lanes_pois[1::2]
    # counterfactual_poi = individual[2*n_nodes:]    
    input_poi = np.repeat(np.array(counterfactual_poi).reshape(1,n_nodes)[:, np.newaxis,:], 12, axis=1)
    input_lanes = np.repeat(np.array(counterfactual_lanes).reshape(1,n_nodes)[:, np.newaxis,:], 12, axis=1)
    input_speedlimit = np.repeat(np.array(np.repeat(counterfactual_speedlimit,n_nodes)).reshape(1,n_nodes)[:, np.newaxis,:], 12, axis=1)    
    counterfactual_lanes = lanes_pois[1::2]
    return objective_function(counterfactual_poi,counterfactual_lanes,input_poi,input_lanes,input_speedlimit, data_point, target_speed_increase, individual)

toolbox.register("evaluate", evaluate)
toolbox.register("gaussian_mutation", np.random.normal)

def mutate(individual):
    sigma_lanes = 2 # Define standard deviation
    sigma_speedlimit = 30 # Define standard deviation
    sigma_poi = 5 # Define standard deviation
    
    # Mutate the speedlimit, which is the first element in the individual.
    if np.random.random() < 0.1:  # 10% chance to mutate speedlimit
        individual[0] += toolbox.gaussian_mutation(0, sigma_speedlimit)
        individual[0] = max(min(individual[0], 120), 40)
    
    # Mutate the lanes and poi of each node.
    for i in range(1, len(individual)):
        if np.random.random() < 0.1:  # 10% chance to mutate each attribute
            individual[i] += toolbox.gaussian_mutation(0, sigma_lanes if i%2==1 else sigma_poi)
            individual[i] = int(round(max(min(individual[i], 30 if i%2==1 else 6), 0 if i%2==1 else 1)))

    return individual,

def mate(ind1, ind2):
    # The first element is the speed limit, which remains the same in the offspring.
    speedlimit = ind1[0] if np.random.random() < 0.5 else ind2[0]

    # The rest of the elements are lanes and poi, which can be crossed over.
    lanes_pois1 = ind1[1:]
    lanes_pois2 = ind2[1:]

    crossover_point = np.random.randint(1, len(lanes_pois1))
    lanes_pois_offspring1 = lanes_pois1[:crossover_point] + lanes_pois2[crossover_point:]
    lanes_pois_offspring2 = lanes_pois2[:crossover_point] + lanes_pois1[crossover_point:]

    return creator.Individual([speedlimit] + lanes_pois_offspring1), creator.Individual([speedlimit] + lanes_pois_offspring2)


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
pareto_front_df.to_csv('Data/moc/pareto_front_t45.csv', index=False)
pd.DataFrame(pareto_front[0]).to_csv('Data/moc/individual_t45.csv', index=False)
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

