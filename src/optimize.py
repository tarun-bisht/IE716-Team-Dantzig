import numpy as np
import pandas as pd
import folium
import math
from pyomo.environ import (ConcreteModel, 
                           Var, 
                           ConstraintList, 
                           Objective, 
                           Binary,
                           Reals,
                           PositiveIntegers,
                           SolverFactory)



def initiate_solver(solver: str, bin_path: str):
    return SolverFactory(solver, executable=bin_path)


def read_data(city_data: str, city_distances: str, city_demands: str):
    cities_data = pd.read_csv(city_data, index_col=0)
    cities_distances = pd.read_csv(city_distances, index_col=0)
    cities_demands = pd.read_csv(city_demands, index_col=0)
    cities_coords = cities_data.iloc[:, 2:4].to_numpy()
    cities_names = cities_data.iloc[:, 1].to_numpy()
    distance_matrix = cities_distances.to_numpy()
    demand_list = cities_demands.iloc[:, 2].values
    
    return cities_coords, cities_names, distance_matrix, demand_list


def generate_map(cities_data, cities_distances, cities_demand=None, polylines=None, distances=None):
    cities_coords = cities_data.iloc[:, 2:4].to_numpy()
    cities_names = cities_data.iloc[:, 1].to_numpy()
    cities_dists = cities_distances.to_numpy()
    if cities_demand is not None:
        cities_demand = cities_demand.iloc[:, 2].to_numpy()

    min_pt = (cities_data[['lat', 'lon']].min().values).tolist()
    max_pt = (cities_data[['lat', 'lon']].max().values).tolist()
    
    mymap = folium.Map(location=[18.9690, 72.8777])
    if cities_demand is not None:
        for name, pt, demand in zip(cities_names, cities_coords, cities_demand):
            if demand == 0:
                folium.Marker(list(pt), popup = f'{name} \n {demand}', icon=folium.Icon(color="black", icon="square-parking", prefix="fa")).add_to(mymap)
            elif demand < 0:
                folium.Marker(list(pt), popup = f'{name} \n {demand}', icon=folium.Icon(color="red", icon="truck-pickup", prefix="fa")).add_to(mymap)
            else:
                folium.Marker(list(pt), popup = f'{name} \n {demand}', icon=folium.Icon(color="green", icon="truck-pickup", prefix="fa")).add_to(mymap)
    else:
        for name, pt in zip(cities_names, cities_coords):
            folium.Marker(list(pt), popup = name).add_to(mymap)

    mymap.fit_bounds([min_pt, max_pt])
    if polylines is not None:
        for i in range(len(polylines)):
            line = polylines[i]
            # Calculate the bearing between the two points
            start_point = line[0]
            end_point = line[-1]
            lat1, lon1 = start_point
            lat2, lon2 = end_point
            y = math.sin(lon2-lon1) * math.cos(lat2)
            x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1)
            bearing = math.atan2(y, x) * 180 / math.pi
            # Calculate the midpoint of the polyline
            midpoint = [(line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2]
            if distances is not None:
                folium.PolyLine(line, tooltip=distances[i], popup=distances[i], weight=6, color="#111").add_to(mymap)
            else:
                folium.PolyLine(line, weight=1, color="#111").add_to(mymap)

            # Add an arrow marker to the end of the polyline with the correct rotation
            folium.RegularPolygonMarker(
                location=midpoint,
                color="#111",
                fill_color="#111",
                number_of_sides=3,
                opacity=1,
                radius=10,
                rotation=bearing-90,
                tooltip=''
            ).add_to(mymap)
    return mymap


def parse_optimal_path_index(model: ConcreteModel):
    sol_edges, _ = parse_solution(model)
    sol_dict = {key:val for key, val in sol_edges}
    optimal_path_idx = []
    c = 0
    idx = 0
    while c < len(model.u):
        optimal_path_idx.append(idx)
        idx = sol_dict[idx]
        c += 1
    optimal_path_idx.append(0)
    return optimal_path_idx


def parse_optimal_path(model: ConcreteModel, cities_names: list):
    optimal_path_idx = parse_optimal_path_index(model)
    return [cities_names[i] for i in optimal_path_idx]


def parse_solution(model: ConcreteModel):
    sol_edges = []
    sol_demands = []
    for i in range(len(model.u)):
        for j in range(len(model.u)):
            if model.x[i, j].value:
                if i!=0 or j!=0:
                    sol_edges.append((j, i))
                    sol_demands.append(model.f[i, j].value)
    return sol_edges, sol_demands

def get_map(city_data: str, city_distances: str, city_demands: str, model: ConcreteModel):
    cities_data = pd.read_csv(city_data, index_col=0)
    cities_distances = pd.read_csv(city_distances, index_col=0)
    cities_demands = pd.read_csv(city_demands, index_col=0)
    cities_coords = cities_data.iloc[:, 2:4].to_numpy()
    cities_names = cities_data.iloc[:, 1].to_numpy()
    distance_matrix = cities_distances.to_numpy()
    optimal_path_idx = parse_optimal_path_index(model)

    sol_edges, _ = parse_solution(model)
    polylines = []
    for edge in sol_edges:
        inp, out = edge
        inp_coordinate = cities_coords[inp]
        out_coordinate = cities_coords[out]
        polylines.append((inp_coordinate, out_coordinate))
    distances = []
    for i in range(len(optimal_path_idx)-1):
        distances.append(f"{distance_matrix[optimal_path_idx[i], optimal_path_idx[i+1]] :.2f} K.M.")
    
    return generate_map(cities_data, cities_distances, cities_demands, polylines, distances)


def PDTSP1Model(distance_matrix, demand_list, vehicle_capacity, depot_index):
    n = len(distance_matrix)
    demand_list[depot_index] = -sum(demand_list)

    model = ConcreteModel(name="1-PDTSP")

    model.x = Var(range(n), range(n), initialize=0.0, domain=Binary)
    model.f = Var(range(n), range(n), initialize=np.inf, domain=Reals)
    model.u = Var(range(n), domain=PositiveIntegers)

    model.objective = Objective(expr=sum(distance_matrix[i, j]*model.x[i, j] for i in range(n) for j in range(n)))

    model.constraints = ConstraintList()
    for i in range(n):
        model.constraints.add(sum(model.x[i, k] for k in range(n) if i != k) == 1)

    for j in range(n):
        model.constraints.add(sum(model.x[k, j] for k in range(n) if j != k) == 1)

    model.constraints.add(model.u[0] == 1)
    for i in range(1, n):
        model.u[i].setlb(2)
        model.u[i].setub(n)

    for i in range(n):
        for j in range(n):
            if i != 0 and j != 0:
                model.constraints.add(model.u[i] - model.u[j] + 1 <= (n-1)*(1-model.x[i, j]))

    for i in range(n):
        model.constraints.add(sum(
            model.f[i, k] for k in range(n) if i != k) - sum(
                model.f[k, i] for k in range(n) if i != k) == demand_list[i])

    for i in range(n):
        for j in range(n):
            model.constraints.add(model.f[i, j] >= 0)
            model.constraints.add(model.f[i, j] <= vehicle_capacity*model.x[i, j])
    return model
