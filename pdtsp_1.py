from src.optimize import initiate_solver, read_data, PDTSP1Model, parse_optimal_path, get_map
import argparse
import json
import time
import webbrowser
import os


parser = argparse.ArgumentParser(description="Solve Single Vehicle Pickup and Delivery problem given cities data")
parser.add_argument("-c", "--config", required=True, type=str, 
                    help="Path to config file, it contains settings related to program")

args = parser.parse_args()
try:
    with open(args.config, "r") as f:
        config = json.load(f)
except Exception as e:
    config = None
    print("Config Parsing error, config is in JSON format")
    print("ERROR: ", e)
    exit(0)

print("Initiating Solver ...")
solver = initiate_solver(config["solver"], config["solver_binary_path"])

print("Reading Data ...")
cities_coords, cities_names, distance_matrix, demand_list = read_data(config["city_details_csv"], 
                                                                    config["city_distances_csv"],
                                                                    config["city_demands_csv"])
model = PDTSP1Model(distance_matrix=distance_matrix, demand_list=demand_list, 
                    vehicle_capacity=float(config["vehicle_capacity"]), 
                    depot_index=int(config["depot_index"]))
model.pprint()

print("Solving PDTSP Problem ...")
start_time = time.time()
result = solver.solve(model)
end_time = time.time()
print("Solver Status: ", result.solver.status)
print("Solver Termination Condition: ", result.solver.termination_condition)
print("Time taken to solve: ", f"{end_time - start_time: .2f}sec")

print("="*200)
print("Minimum travelling distance: ", model.objective(), "\n")
print("Optimal Path Sequence \n")
optimal_cities_seq = parse_optimal_path(model, cities_names.tolist())
for city in optimal_cities_seq:
    print(city, " -> ", end="")
print(optimal_cities_seq[0])
print("="*200)

pdtsp_map = get_map(config["city_details_csv"], config["city_distances_csv"], 
                    config["city_demands_csv"], model)
map_save_path = os.path.join(config["output_map_save_path"], "map.html")
pdtsp_map.save(map_save_path)
webbrowser.open(map_save_path, new=2)



