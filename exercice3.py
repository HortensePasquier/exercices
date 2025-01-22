import json
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Load data
with open("C:\portfolio-example.json","r") as f:
    data = json.load(f)

n = data["num_assets"]
sigma = np.array(data["covariance"])
mu = np.array(data["expected_return"])
mu_0 = data["target_return"]
k = data["portfolio_max_size"]

# Créer le modèle
with gp.Model("portfolio") as model:
    # Variables de décisions
    x = model.addVars(n, lb=0, ub=1, name="x")  #Investissement relatif dans l'actif
    y = model.addVars(n, vtype=GRB.BINARY, name="y")  # Variable binaire contrôlant si l'actif j est échangé
    
    # Objective: Minimiser le risque
    risk = gp.quicksum(sigma[i][j] * x[i] * x[j] for i in range(n) for j in range(n))
    model.setObjective(risk, GRB.MINIMIZE)
    
    # Contraintes
    # Rendement minimal
    model.addConstr(gp.quicksum(mu[i] * x[i] for i in range(n)) >= mu_0, name="return")
    
    # Investissement total
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1, name="investment")
    
    #Le nombre d'actifs sélectionnés ne doit pas dépasser k
    model.addConstr(gp.quicksum(y[i] for i in range(n)) <= k, name="cardinality")
    
    # lien entre x et y
    for i in range(n):
        model.addConstr(x[i] <= y[i], name=f"link_{i}")
    
    
    model.optimize()
    
    
    if model.status == GRB.OPTIMAL:
        portfolio = [x[i].X for i in range(n)]
        risk = model.ObjVal
        expected_return = sum(mu[i] * portfolio[i] for i in range(n))
        df = pd.DataFrame(
            data=portfolio + [risk, expected_return],
            index=[f"asset_{i}" for i in range(n)] + ["risk", "return"],
            columns=["Portfolio"],
        )
        print(df)
    else:
        print("No optimal solution found.")
