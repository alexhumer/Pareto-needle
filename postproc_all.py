# %%
import numpy as np
import pickle
import plotly.express as px
import pandas as pd


# %%
path = r"./"

data = {}
fname = "res_all.pkl."
suffix = "1"

# %% first load the data    
with open(path + fname + suffix, "rb") as f:
    X, F = pickle.load(f)

data = (X, F)

# %%
fl = F.min(axis=0)
fu = F.max(axis=0)

# %% ... then rearrange it appropriately
df_results = None

X_ = X
X = np.array([[x["eta"], x["alpha"], x["beta"]] for x in X_])
S = np.array([[x["shape"]] for x in X_])

# %%
nF = (F - fl) / (fu - fl)

eta_l = 10
eta_u = 50
alpha_l = 0.1
alpha_u = 1.
beta_l = 0.1
beta_u = 0.5

nX = np.zeros_like(X)
nX[:,0] = (X[:,0] - eta_l) / (eta_u - eta_l)
nX[:,1] = (X[:,1] - alpha_l) / (alpha_u - alpha_l)
nX[:,2] = (X[:,2] - beta_l) / (beta_u - beta_l)

for i, S_ in enumerate(S):
    if S_ == "tri":
        X[i,1] = (1.9/0.9*(X[i,1]-0.1) + 0.1)

df = pd.DataFrame({
    "shape": S[:,0], 
    "eta": X[:,0],
    "alpha": X[:,1],
    "beta": X[:,2],
    "f_A" : F[:,0],
    "f_d" : F[:,1],
    "f_P" : F[:,2],
    "f_Q" : F[:,3],
    })

df_n = pd.DataFrame({
    "shape": S[:,0], 
    "eta": nX[:,0],
    "alpha": nX[:,1],
    "beta": nX[:,2],
    "f_A" : nF[:,0],
    "f_d" : nF[:,1],
    "f_P" : nF[:,2],
    "f_Q" : nF[:,3],
    })

# %%
df_results = df
fig = px.scatter_3d(
    df_results,
    x=df_results["f_A"],
    y=df_results["f_d"],
    z=df_results["f_P"],
    color=df_results["shape"],
    width=300*2,
    height=300*2,
    size_max=1,
    )
fig.update_scenes(aspectmode='cube')
fig.update_traces(marker=dict(size=3))
fig.show()
# %%
fig = px.scatter_3d(
    df_results,
    x=df_results["f_A"],
    y=df_results["f_d"],
    z=df_results["f_Q"],
    color=df_results["shape"],
    width=300*2,
    height=300*2,
    size_max=1,
    )
fig.update_scenes(aspectmode='cube')
fig.update_traces(marker=dict(size=3))
fig.show()

# %%
fig = px.scatter_3d(
    df_results,
    x=df_results["f_A"],
    y=df_results["f_P"],
    z=df_results["f_Q"],
    color=df_results["shape"],
    width=300*2,
    height=300*2,
    size_max=1,
    )
fig.update_scenes(aspectmode='cube')
fig.update_traces(marker=dict(size=3))
fig.show()

# %%
fig = px.scatter_3d(
    df_results,
    x=df_results["f_d"],
    y=df_results["f_P"],
    z=df_results["f_Q"],
    color=df_results["shape"],
    width=300*2,
    height=300*2,
    size_max=1,
    )
fig.update_scenes(aspectmode='cube')
fig.update_traces(marker=dict(size=3))
fig.show()

# %%
