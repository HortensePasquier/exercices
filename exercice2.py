import numpy as np
import gurobipy as gp
from gurobipy import GRB

def generate_knapsack(num_items):
    # Fix seed value
    rng = np.random.default_rng(seed=0)
    # Item values, weights
    values = rng.uniform(low=1, high=25, size=num_items)
    weights = rng.uniform(low=5, high=100, size=num_items)
    # Knapsack capacity
    capacity = 0.7 * weights.sum()

    return values, weights, capacity


def solve_knapsack_model(values, weights, capacity):
    num_items = len(values)
    # Turn values and weights numpy arrays to dictionaries for Gurobi
    items = range(num_items)
    values_dict = {i: values[i] for i in items}
    weights_dict = {i: weights[i] for i in items}

    with gp.Env() as env:
        with gp.Model(name="knapsack", env=env) as model:
            # Define decision variables (binary: 0 or 1)
            x = model.addVars(items, vtype=GRB.BINARY, name="x")

            # Objective function: maximize total value
            model.setObjective(x.prod(values_dict), GRB.MAXIMIZE)

            # Capacity constraint: total weight ≤ capacity
            model.addConstr(x.prod(weights_dict) <= capacity, "capacity")

            # Optimize the model
            model.optimize()

            # Extract and print solution
            if model.status == GRB.OPTIMAL:
                selected_items = [i for i in items if x[i].x > 0.5]
                print(f"Optimal value: {model.objVal}")
                #print(f"Selected items: {selected_items}")
                print(f"Total weight: {sum(weights[i] for i in selected_items)}")
            else:
                print("No optimal solution found!")



data = generate_knapsack(10000)  # Generate data for 10,000 items
solve_knapsack_model(*data)
