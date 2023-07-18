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
data_point = train_dataset[2748][0]# size(1, 12,3169,17) # 96 #2784-12, thursday afternoon //2664//2676(7am)/2748
data_point = data_point.reshape(1,12,3169,17)

node_index = 1800 #882/ 881 / 1034
# Set target outcome
target_speed_increase = 0.06


# add constraint list: index of the specific road
road_index_df= pd.read_csv("Data/ventura_road/spd_72_north_index.csv", header=None)
road_index = road_index_df[0].to_numpy()

# n_nodes = road_index.shape[0]
n_nodes = 3169
# lanes_value = [1,2,3,4,5,6,7,8]
# parameter_combinations = list(itertools.product(lanes_value))

training_samples = data_point[0,0,:,14:17]

individual_df = pd.read_csv('Data/moc/individual_t26.csv')
individual_value = individual_df.iloc[4].to_numpy()

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
    cf_point_norm = cf_point/original_samples.amax(axis = 0)
    original_samples_norm = original_samples/original_samples.amax(axis = 0)
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(original_samples_norm)
    distances, indices = neigh.kneighbors(cf_point_norm)
    distances = distances.flatten()
    indices = indices.flatten()
    overall_avg_distance = distances.mean()
    # for point in counterfactual
    # for point in cf_point:
    #     point_distances=[l2_distance(point, sample) for sample in original_samples]
    #     closest_samples = np.partition(point_distances, k)[:k]
    #     avg_distance = np.mean(closest_samples)
    #     distances.append(avg_distance)
    # overall_avg_distance = np.mean(distances)
    return overall_avg_distance
# ,day_of_week, 
def objective_function(speed, temp,precip, wind, humidity,day_of_week,poi_data, lanes_data,speedlimit_data, orig_data, target_speed_increase, individual):
    modified_data = orig_data.clone()
    modified_data[:,:,:,14]= torch.from_numpy(poi_data.astype(np.float))
    modified_data[:,:,:,15]= torch.from_numpy(lanes_data.astype(np.float))
    modified_data[:,:,:,16]= torch.from_numpy(speedlimit_data.astype(np.float))
    modified_data[:,:,:,1] = torch.from_numpy(temp).float()
    modified_data[:,:,:,2] = torch.from_numpy(precip).float()
    modified_data[:,:,:,3] = torch.from_numpy(wind).float()
    modified_data[:,:,:,4] = torch.from_numpy(humidity).float()
    modified_data[:,:,node_index,0] = torch.from_numpy(speed).float()
    day_of_week_np = np.array(day_of_week)
    day_of_week_expanded = day_of_week_np[np.newaxis, np.newaxis, np.newaxis, :]
    day_of_week_reshaped = np.repeat(day_of_week_expanded, 1, axis=0)
    day_of_week_reshaped = np.repeat(day_of_week_reshaped, 12, axis=1)
    day_of_week_reshaped = np.repeat(day_of_week_reshaped, 3169, axis=2)
    modified_data[:,:,:,5:12] = torch.from_numpy(day_of_week_reshaped).float()

    # calculate the difference between the predicted speed and target speed
    # pred_speed = model(modified_data)[node_index]#[:,0,:,0]
    orig_speed = model(orig_data)[node_index]#[:,0,:,0]
    pred_speed = model(modified_data)[node_index]#[:,0,:,0]
    # target_speed = orig_speed+target_speed_increase #*(1+target_speed_increase)
    target_speed= 0.4417 # 72/163
    outcome_diff = torch.abs(pred_speed - target_speed)
    outcome_diff = outcome_diff.mean()

    day_of_week_diff = torch.linalg.norm(modified_data[:,:,:,5:12] - orig_data[:,:,:,5:12])
    if day_of_week[3]!=1:
        sparsity_day_of_week = 1
    else:
        sparsity_day_of_week = 0

    #for speed difference, number of time step change + overall change
    # speed_diff = torch.linalg.norm(modified_data[:,:,node_index,0] - orig_data[:,:,node_index,0])
    # speed_sparsity = sum([1 for i in range(12) if (speed[i]!= orig_data[:,:,node_index,0][0,i])])

    # Calculate the similarity between the original input and the counterfactual
    temp_diff = torch.linalg.norm(modified_data[:,:,:,1] - orig_data[:,:,:,1])
    precip_diff = torch.linalg.norm(modified_data[:,:,:,2] - orig_data[:,:,:,2])
    wind_diff = torch.linalg.norm(modified_data[:,:,:,3] - orig_data[:,:,:,3])
    humidity_diff = torch.linalg.norm(modified_data[:,:,:,4] - orig_data[:,:,:,4])
    poi_diff = torch.linalg.norm(torch.from_numpy(poi_data[0,0]).float()- orig_data[:,:,:,14][0,0])
    lanes_diff = torch.linalg.norm(torch.from_numpy(lanes_data[0,0]).float() - orig_data[:,:,:,15][0,0])
    speedlimit_diff = torch.linalg.norm(torch.from_numpy(speedlimit_data[0,0]).float() - orig_data[:,:,:,16][0,0])
    sparsity_poi = sum([1 for i in range(n_nodes) if (poi_data[0,0,i]!= original_poi[0,i])])
    sparsity_lanes = sum([1 for i in range(n_nodes) if (lanes_data[0,0,i]!=original_lanes[0,i])])
    sparsity_speedlimit = sum([1 for i in range(n_nodes) if (speedlimit_data[0,0,i]!=original_speedlimit[0,i])])
    sparsity = sparsity_poi + sparsity_lanes + sparsity_speedlimit
    # plausibility
    cf_points = np.array(individual[4+12+7:]).reshape(3169,3)
    plausibility = calculate_plausibility(cf_points, training_samples, 1)
    # nearest_samples = get_k_nearest(individual)
    # plausibility = sum(l2_distance(individual, sample) for sample in nearest_samples)/len(nearest_samples)

    # Define weights for the objective function
    w_outcome = 163.0
    w_poi = 0.01
    w_lanes = 0.01
    w_speedlimit = 0.01
    w_sparsity = 0.01
    w_plausibility = 0.01
    w_temp = 0.01
    w_dayofweek = 0.01
#+w_temp*torch.from_numpy(np.array(temp)).float().item()#speed_sparsity 
    return (w_outcome*outcome_diff.item(), 
            w_dayofweek*day_of_week_diff + w_temp*temp_diff.mean().item()+w_temp*precip_diff.mean().item()+w_temp*wind_diff.mean().item()+w_temp*humidity_diff.mean().item()+ w_speedlimit*speedlimit_diff.mean().item()+w_poi*poi_diff.mean().item()+w_lanes*lanes_diff.mean().item(),
            sparsity_day_of_week + w_sparsity*sparsity,
            w_plausibility*plausibility
            )
    # return w_outcome*outcome_diff.mean().detach().numpy(),w_speedlimit*speedlimit_diff.mean().detach().numpy(),w_poi*poi_diff.mean().detach().numpy(),w_lanes*lanes_diff.mean().detach().numpy() # w_hour_of_day*hour_of_day_diff.mean() + w_temp*temp_diff.mean() + w_day_of_week*day_of_week_diff

# Initialize the counterfactual input
original_poi = data_point[:,0,:,14]
original_lanes = data_point[:,0,:,15]
original_speedlimit = data_point[:,0,:,16]
original_temperature = data_point[:,:,0,1]
original_precip = data_point[:,:,0,2]
original_wind = data_point[:,:,0,3]
original_humidity = data_point[:,:,0,4]
original_speed = data_point[:,:,node_index,0]
# counterfactual_poi = original_poi[:,road_index]
counterfactual_data = data_point#.clone().detach().requires_grad_(True)
counterfactual_poi = original_poi.clone().detach().requires_grad_(True)
counterfactual_lanes = original_lanes.clone().detach().requires_grad_(True)
counterfactual_speedlimit = original_speedlimit.clone().detach().requires_grad_(True)
counterfactual_temperature = original_temperature.clone().detach().requires_grad_(True)
counterfactual_precip = original_precip.clone().detach().requires_grad_(True)
counterfactual_wind = original_wind.clone().detach().requires_grad_(True)
counterfactual_humidity = original_humidity.clone().detach().requires_grad_(True)
counterfactual_speed = original_speed.clone().detach().requires_grad_(True)
# the change of temperature

####################### Implement NSGA-II optimization #########################
# creator.create("FitnessMulti", base.Fitness, weight = (-1,-1,-1,-1)) # 4 objectives in total TODO
creator.create("FitnessMulti", base.Fitness, weights=(-10.0,-1.0, -1.0,-1.0))
creator.create("Individual", list, fitness = creator.FitnessMulti)
toolbox = base.Toolbox()

toolbox.register("poi_init", np.random.randint, 0, 36)
toolbox.register("lanes_init", np.random.randint, 1, 16)
toolbox.register("speedlimit_init", np.random.uniform, 10, 120)
toolbox.register("temperature_init", np.random.uniform, 0, 1)
toolbox.register("precip_init", np.random.uniform, 0, 1)
toolbox.register("wind_init", np.random.uniform, 0, 1)
toolbox.register("humidity_init", np.random.uniform, 0, 1)
toolbox.register("speed_init", np.random.uniform, 0, 1)
def day_of_week_init():
    day_of_week = [0]*7
    day_of_week[np.random.randint(0, 6)] = 1  # Randomly choose a day
    return day_of_week

def init_individual(speed_init, poi_init, lanes_init, speedlimit_init,temperature_init,precip_init,wind_init,humidity_init,day_of_week_init, n_nodes):
    # Return a single individual, which is a list starting with the lane and speedlimit values, followed by POI values
    temp_change = [temperature_init()]
    precip_change = [precip_init()]
    wind_change = [wind_init()]
    humidity_change = [humidity_init()]
    speedlimit = [speedlimit_init() for _ in range(n_nodes)]
    lanes = [lanes_init() for _ in range(n_nodes)]
    pois = [poi_init() for _ in range(n_nodes)]
    speed_history = [speed_init() for _ in range(12)]
    day_of_week = day_of_week_init()
    return creator.Individual(temp_change + precip_change + wind_change + humidity_change + speed_history + day_of_week+[value for pair in zip(pois, lanes, speedlimit) for value in pair])
    # return creator.Individual([lanes_init() for _ in range(n_nodes)] + [speedlimit_init() for _ in range(n_nodes)] + [poi_init() for _ in range(n_nodes)])

 
toolbox.register("init_individual", init_individual, toolbox.speed_init, toolbox.poi_init, toolbox.lanes_init, toolbox.speedlimit_init, toolbox.temperature_init,
                toolbox.precip_init,toolbox.wind_init,toolbox.humidity_init,day_of_week_init, n_nodes)
toolbox.register("individual", init_individual, toolbox.speed_init, toolbox.poi_init, toolbox.lanes_init, toolbox.speedlimit_init, toolbox.temperature_init, 
                toolbox.precip_init,toolbox.wind_init,toolbox.humidity_init,day_of_week_init, n_nodes)
toolbox.register("population", tools.initRepeat,list, toolbox.individual)
def evaluate(individual):
    individual = individual_value
    # counterfactual_lanes = individual[:n_nodes]
    # counterfactual_temp = individual[0]
    individual_n = individual[4+12+7:]
    counterfactual_poi = individual_n[::3]
    counterfactual_lanes = individual_n[1::3]
    counterfactual_speedlimit = individual_n[2::3] 
    day_of_week = individual[4+12:4+12+7]
    input_temp = np.full((1,12,3169), individual[0])
    input_precip = np.full((1,12,3169), individual[1])
    input_wind = np.full((1,12,3169), individual[2])
    input_humidity = np.full((1,12,3169), individual[3])
    input_speed = np.array(individual[4:4+12])
    input_poi = np.repeat(np.array(counterfactual_poi).reshape(1,n_nodes)[:, np.newaxis,:], 12, axis=1)
    input_lanes = np.repeat(np.array(counterfactual_lanes).reshape(1,n_nodes)[:, np.newaxis,:], 12, axis=1)
    input_speedlimit = np.repeat(np.array(counterfactual_speedlimit).reshape(1,n_nodes)[:, np.newaxis,:], 12, axis=1)    #day_of_week, 
    return objective_function(input_speed, input_temp,input_precip,input_wind,input_humidity,day_of_week,input_poi,input_lanes,input_speedlimit, data_point, target_speed_increase, individual)

toolbox.register("evaluate", evaluate)
toolbox.register("gaussian_mutation", np.random.normal)

def mutate(individual):
    sigma_lanes = 3 # Define standard deviation
    sigma_speedlimit = 30 # Define standard deviation
    sigma_poi = 5 # Define standard deviation
    sigma_temp = 0.205 # Define standard deviation
    sigma_precip = 1 # Define standard deviation
    sigma_wind = 1 # Define standard deviation
    sigma_humidity = 1 # Define standard deviation
    sigma_speed = 15/163
    # Mutate the speedlimit, which is the first element in the individual.
    if np.random.random() < 0.1:  # 10% chance to mutate speedlimit
        individual[0] += toolbox.gaussian_mutation(0, sigma_temp)
        individual[0] = max(min(individual[0], 1), 0)
    if np.random.random() < 0.1:  # 10% chance to mutate speedlimit
        individual[1] += toolbox.gaussian_mutation(0, sigma_precip)
        individual[1] = max(min(individual[1], 1), 0)
    if np.random.random() < 0.1:  # 10% chance to mutate speedlimit
        individual[2] += toolbox.gaussian_mutation(0, sigma_wind)
        individual[2] = max(min(individual[2], 1), 0)
    if np.random.random() < 0.1:  # 10% chance to mutate speedlimit
        individual[3] += toolbox.gaussian_mutation(0, sigma_humidity)
        individual[3] = max(min(individual[3], 1), 0)
    for i in range(4, 4+12):
        if np.random.random() < 0.1: 
            individual[i] += toolbox.gaussian_mutation(0, sigma_speed)
            individual[i] = max(min(individual[0], 1), 0)
    if np.random.random() < 0.1:  # 10% chance to mutate day of week
        day_of_week_index = -7  # Assuming day of week is the last 7 elements in the individual
        day_of_week = individual[day_of_week_index:]
        selected_day = np.random.randint(0, 6)  # Randomly choose a day
        day_of_week = [1 if i == selected_day else 0 for i in range(7)]  # Set one day to 1 and the rest to 0
        individual[day_of_week_index:] = day_of_week
    # Mutate the lanes and poi of each node.
    for i in range(4+12, len(individual)):
        if np.random.random() < 0.1:  # 10% chance to mutate each attribute
            if i%3 == 1:
                individual[i] += toolbox.gaussian_mutation(0, sigma_poi)
                individual[i] = int(round(max(min(individual[i], 30), 0)))
            if i%3 == 2:
                individual[i] += toolbox.gaussian_mutation(0, sigma_lanes)
                individual[i] = int(round(max(min(individual[i], 16), 1)))
            if i%3 == 0:
                individual[i] += toolbox.gaussian_mutation(0, sigma_speedlimit)
                individual[i] = max(min(individual[i], 120), 10)
            # individual[i] += toolbox.gaussian_mutation(0, sigma_lanes if i%2==1 else sigma_poi)
            # individual[i] = int(round(max(min(individual[i], 6 if i%2==1 else 30), 1 if i%2==1 else 0)))

    return individual,

def mate(ind1, ind2):
    # The first element is the speed limit, which remains the same in the offspring.
    temp = ind1[0] if np.random.random() < 0.5 else ind2[0]

    # The rest of the elements are lanes and poi, which can be crossed over.
    lanes_pois1 = ind1[1:]
    lanes_pois2 = ind2[1:]

    crossover_point = np.random.randint(1, len(lanes_pois1))
    lanes_pois_offspring1 = lanes_pois1[:crossover_point] + lanes_pois2[crossover_point:]
    lanes_pois_offspring2 = lanes_pois2[:crossover_point] + lanes_pois1[crossover_point:]

    return creator.Individual([temp] + lanes_pois_offspring1), creator.Individual([temp] + lanes_pois_offspring2)


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
pareto_front_df.to_csv('Data/moc/pareto_front_t27.csv', index=False)
pd.DataFrame(pareto_front[0]).to_csv('Data/moc/individual_t27.csv', index=False)
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

# for ind in pareto_front:
#     poi, lanes, speedlimit = ind[0],ind[1],ind[2]
    
#     print(f"POI: {poi}, Lanes: {lanes}, Speed Limit: {speedlimit}")

