# %%
import os
import numpy as np
import pickle, dill

import matplotlib.pyplot as plt
# %matplotlib ipympl
%config InlineBackend.print_figure_kwargs = {'bbox_inches': None}
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

import vtk
import pyvista as pv
from alphashape import alphashape
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pymoo.visualization.scatter import Scatter

# %%
path = r"./results/"

fname = "res_all.pkl."

data = {}

# 1 ... AGEMOEA2
# 1.2 . AGEMOEA2 merged 1000 / 100
# 1.2 . AGEMOEA2 merged 500 / 200
# 2 ... NSGA2

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
# F = nF

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
df_results = df_n
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 12
})
fact = 2
alpha = 0.75
markers = dict(
    ellipse=dict(marker="o", s=5*fact, c="C0", alpha=alpha, clip_on=False, label=r"\textrm{ellipse}"),
    rect=dict(marker="s", s=5*fact, c="C1", alpha=alpha, clip_on=False, label=r"\textrm{rectangle}"),
    tri=dict(marker="^", s=5*fact, c="C2", alpha=alpha, clip_on=False, label=r"\textrm{triangle}"),
    cross=dict(marker="+", s=12*fact, c="C3", alpha=alpha, clip_on=False, label=r"\textrm{cross}"),
    )

# fig, ax = plt.subplots(subplot_kw={'projection': '3d'},)
df_ellipse = df_results[df_results["shape"] == "ellipse"]
df_rect = df_results[df_results["shape"] == "rect"]
df_tri = df_results[df_results["shape"] == "tri"]
df_cross = df_results[df_results["shape"] == "cross"]

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8,8*2/3))
ax[0,0].scatter(df_cross.f_d, df_cross.f_A, **markers["cross"])
ax[0,0].scatter(df_rect.f_d, df_rect.f_A, **markers["rect"])
ax[0,0].scatter(df_ellipse.f_d, df_ellipse.f_A, **markers["ellipse"])
ax[0,0].scatter(df_tri.f_d, df_tri.f_A, **markers["tri"])
ax[0,0].set_xlabel(r"$\tilde f_d / 1$")
ax[0,0].set_ylabel(r"$\tilde f_A / 1$")
ax[0,0].set_xlim(0,1)
ax[0,0].set_ylim(0,1)
ax[0,0].set_xticks([0,0.2,0.4,0.6,0.8,1])

# ax2 = fig.add_subplot(332,sharey=ax1)
ax[0,1].scatter(df_cross.f_P, df_cross.f_A, **markers["cross"])
ax[0,1].scatter(df_rect.f_P, df_rect.f_A, **markers["rect"])
ax[0,1].scatter(df_ellipse.f_P, df_ellipse.f_A, **markers["ellipse"])
ax[0,1].scatter(df_tri.f_P, df_tri.f_A, **markers["tri"])
ax[0,1].set_xlabel(r"$\tilde f_P / 1$")
ax[0,1].set_ylabel(r"$\tilde f_A / 1$")
ax[0,1].set_xlim(0,1)
ax[0,1].set_ylim(0,1)

# ax3 = fig.add_subplot(333,sharey=ax1)
# ax[0,2].scatter(df_cross.f_Q, df_cross.f_A, s=5, alpha=1)
ax[0,2].scatter(df_cross.f_Q, df_cross.f_A, **markers["cross"])
ax[0,2].scatter(df_rect.f_Q, df_rect.f_A, **markers["rect"])
ax[0,2].scatter(df_ellipse.f_Q, df_ellipse.f_A, **markers["ellipse"])
ax[0,2].scatter(df_tri.f_Q, df_tri.f_A, **markers["tri"])
ax[0,2].set_xlabel(r"$\tilde f_Q / 1$")
ax[0,2].set_ylabel(r"$\tilde f_A / 1$")
ax[0,2].set_xlim(0,1)
ax[0,2].set_ylim(0,1)


# ax[1,0].scatter(df_cross.f_P, df_cross.f_d, s=5, alpha=1)
ax[1,0].scatter(df_cross.f_P, df_cross.f_d, **markers["cross"])
ax[1,0].scatter(df_rect.f_P, df_rect.f_d, **markers["rect"])
ax[1,0].scatter(df_ellipse.f_P, df_ellipse.f_d, **markers["ellipse"])
ax[1,0].scatter(df_tri.f_P, df_tri.f_d, **markers["tri"])
ax[1,0].set_xlabel(r"$\tilde f_P / 1$")
ax[1,0].set_ylabel(r"$\tilde f_d / 1$")
ax[1,0].set_xlim(0,1)
ax[1,0].set_ylim(0,1)

# ax[1,1].scatter(df_cross.f_Q, df_cross.f_d, s=5, alpha=1)
ax[1,1].scatter(df_cross.f_Q, df_cross.f_d, **markers["cross"])
ax[1,1].scatter(df_rect.f_Q, df_rect.f_d, **markers["rect"])
ax[1,1].scatter(df_ellipse.f_Q, df_ellipse.f_d, **markers["ellipse"])
ax[1,1].scatter(df_tri.f_Q, df_tri.f_d, **markers["tri"])
ax[1,1].set_xlabel(r"$\tilde f_Q / 1$")
ax[1,1].set_ylabel(r"$\tilde f_d / 1$")
ax[1,1].set_xlim(0,1)
ax[1,1].set_ylim(0,1)


# ax[1,2].scatter(df_cross.f_Q, df_cross.f_P, s=5, alpha=1)
ax[1,2].scatter(df_cross.f_P, df_cross.f_Q, **markers["cross"])
ax[1,2].scatter(df_rect.f_P, df_rect.f_Q, **markers["rect"])
ax[1,2].scatter(df_ellipse.f_P, df_ellipse.f_Q, **markers["ellipse"])
ax[1,2].scatter(df_tri.f_P, df_tri.f_Q, **markers["tri"])
ax[1,2].set_xlabel(r"$\tilde f_P / 1$")
ax[1,2].set_ylabel(r"$\tilde f_Q / 1$")
ax[1,2].set_xlim(0,1)
ax[1,2].set_ylim(0,1)
fig.tight_layout()

handles, labels = ax[0,0].get_legend_handles_labels()
# order = ["ellipse", "triangle", "rectangle", "cross"]
# handles = [handles[labels.index(label)] for label in order]
# labels = [label for label in order]

fig.legend(
    handles, 
    labels, 
    loc='lower center', 
    bbox_to_anchor=(0.5, 0.), 
    ncol=4,
    frameon=False,
    prop={"size": 12},
    markerscale=1.5)

# fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.1))
plt.subplots_adjust(bottom=0.175)
plt.show()
# fig.savefig("../manuscript/figures/pareto_2d.2.pdf")

# %%
# Generate random 4D points
np.random.seed(42)  # For reproducibility
# points = df_results[df_results["shape"] == "ellipse"].to_numpy()[:,-4:]
points = df_results.to_numpy()[:,-4:]

# Compute the convex hull

# Function to plot a 3D projection of the convex hull
def plot_3d_projection(dim1, dim2, dim3):
    # Project points to 3D
    projected_points = points[:, [dim1, dim2, dim3]]
    
    # Create a Plotly figure
    fig = go.Figure()

    # Add points to the plot
    fig.add_trace(go.Scatter3d(
        x=projected_points[:, 0],
        y=projected_points[:, 1],
        z=projected_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Points'
    ))

    mesh = go.Mesh3d(
        x=projected_points[:, 0],
        y=projected_points[:, 1],
        z=projected_points[:, 2],
        alphahull=1,  # Convex hull
        opacity=0.5,
        color='cyan',
        name='Convex Hull'
    )
    fig.add_trace(mesh)

    # Set axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title=f"Dimension {dim1+1}",
            yaxis_title=f"Dimension {dim2+1}",
            zaxis_title=f"Dimension {dim3+1}"
        ),
        title=f"Projection onto Dimensions {dim1+1}, {dim2+1}, {dim3+1}",
    )

    fig.show()

# Plot projections onto different 3D subspaces
# plot_3d_projection(0, 1, 2)  # Projection onto first three dimensions
# plot_3d_projection(0, 1, 3)  # Projection with first, second, and fourth dimensions
# plot_3d_projection(1, 2, 3)  # Projection with second, third, and fourth dimensions

# %%
# shapes = ["ellipse", "rect", "tri", "cross"]
shapes = ["cross", "rect", "ellipse", "tri"]

fact = 1.
alpha = 0.75
markers_3d = dict(
    ellipse=dict(marker="o", s=5*fact, c="C0", alpha=alpha, clip_on=False, label=r"\textrm{ellipse}"),
    rect=dict(marker="s", s=5*fact, c="C1", alpha=alpha, clip_on=False, label=r"\textrm{rectangle}"),
    tri=dict(marker="^", s=5*fact, c="C2", alpha=alpha, clip_on=False, label=r"\textrm{triangle}"),
    cross=dict(marker="+", s=12*fact, c="C3", alpha=alpha, clip_on=False, label=r"\textrm{cross}"),
    )

labels = [r"$\tilde f_A / 1$", r"$\tilde f_d / 1$", r"$\tilde f_P / 1$", r"$\tilde f_Q / 1$"]
colors_i = [
    [[0., 1., 0., 1]],
    [[1., 0., 0., 1]],
    [[0.5, 0.8, 1, 1]],
    [[1., 0.67, 0., 1]]]
alpha_i = [2, 1.5, 2, 1.2]

dims = [0,1,3]
# Create a Matplotlib 3D plot
# %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}
fig = plt.figure(figsize=(8., 8.5), constrained_layout=False)
# fig.subplots_adjust(wspace=0.2)

from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 2, figure=fig, hspace=0., top=1, bottom=0.05, left=0.0, right=0.94)  # Adjust spacing

ax = fig.add_subplot(gs[0,0], projection='3d')

for i, shape in enumerate(shapes):
    points = df_results[df_results["shape"]==shape].to_numpy()[:,-4:].astype(np.double)

    # Generate the alpha hull
    hull = alphashape(points[:,dims], alpha_i[i])

    vertices = np.array(hull.vertices)
    faces = np.array(hull.faces)

    faces_pv = np.hstack([[3] + list(face) for face in faces])
    hull_mesh = pv.PolyData(vertices, faces_pv) #.smooth_taubin(n_iter=5, pass_band=0.5)

    points_hull = hull_mesh.points  # Array of shape (N, 3), where N is the number of points
    faces_hull = hull_mesh.faces.reshape(-1, 4)[:, 1:]  

    # Map intensities to a color map
    colors = np.array(colors_i[i] * len(faces_hull))

    # Convert faces to a format compatible with Matplotlib's Poly3DCollection
    poly3d = np.array([[points_hull[vertex] for vertex in face] for face in faces_hull])

    collection = Poly3DCollection(
        poly3d, 
        # edgecolor="k", 
        facecolors=colors, 
        linewidth=0.2, 
        alpha=0.1, 
        shade=True)
    # ax.add_collection3d(collection)

    ax.scatter(
        points[:,dims[0]],
        points[:,dims[1]],
        points[:,dims[2]],
        # s=1,
        # c="C"+str(i),
        # alpha=0.75,
        **markers_3d[shape],
    )
ax.view_init(elev=25, azim=-1*45)  # Elevation and azimuth in degrees

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel(labels[dims[0]])
ax.set_ylabel(labels[dims[1]])
ax.set_zlabel(labels[dims[2]])
# ax.invert_yaxis()
ax.set_aspect("equal")
# fig.savefig("pareto_3d_AdQ.pdf")

# %%
dims = [0,1,2]
# Create a Matplotlib 3D plot
# fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(gs[0,1], projection='3d')

for i, shape in enumerate(shapes):
    points = df_results[df_results["shape"]==shape].to_numpy()[:,-4:].astype(np.double)

    # Generate the alpha hull
    hull = alphashape(points[:,dims], alpha_i[i])

    vertices = np.array(hull.vertices)
    faces = np.array(hull.faces)

    faces_pv = np.hstack([[3] + list(face) for face in faces])
    hull_mesh = pv.PolyData(vertices, faces_pv) #.smooth_taubin(n_iter=5, pass_band=0.5)

    points_hull = hull_mesh.points  # Array of shape (N, 3), where N is the number of points
    faces_hull = hull_mesh.faces.reshape(-1, 4)[:, 1:]  

    # Map intensities to a color map
    colors = np.array(colors_i[i] * len(faces_hull))

    # Convert faces to a format compatible with Matplotlib's Poly3DCollection
    poly3d = np.array([[points_hull[vertex] for vertex in face] for face in faces_hull])

    collection = Poly3DCollection(
        poly3d, 
        # edgecolor="k", 
        facecolors=colors, 
        linewidth=0.2, 
        alpha=0.15, 
        shade=True)
    # ax.add_collection3d(collection)

    ax.scatter(
        points[:,dims[0]],
        points[:,dims[1]],
        points[:,dims[2]],
        # alpha=0.75,
        **markers_3d[shape]
    )
ax.view_init(elev=25, azim=-1*45)  # Elevation and azimuth in degrees

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel(labels[dims[0]])
ax.set_ylabel(labels[dims[1]])
ax.set_zlabel(labels[dims[2]])
# ax.invert_yaxis()
ax.set_aspect("equal")
# fig.savefig("pareto_3d_AdP.pdf")

# %%
dims = [0,2,3]
alpha_i = [1.5, 1.5, 1.5, 1.2]

# Create a Matplotlib 3D plot
# fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(gs[1,0], projection='3d')

for i, shape in enumerate(shapes):
    points = df_results[df_results["shape"]==shape].to_numpy()[:,-4:].astype(np.double)

    # Generate the alpha hull
    hull = alphashape(points[:,dims], alpha_i[i])

    vertices = np.array(hull.vertices)
    faces = np.array(hull.faces)

    faces_pv = np.hstack([[3] + list(face) for face in faces])
    hull_mesh = pv.PolyData(vertices, faces_pv) #.smooth_taubin(n_iter=5, pass_band=0.5)

    points_hull = hull_mesh.points  # Array of shape (N, 3), where N is the number of points
    faces_hull = hull_mesh.faces.reshape(-1, 4)[:, 1:]  

    # Map intensities to a color map
    colors = np.array(colors_i[i] * len(faces_hull))

    # Convert faces to a format compatible with Matplotlib's Poly3DCollection
    poly3d = np.array([[points_hull[vertex] for vertex in face] for face in faces_hull])

    collection = Poly3DCollection(
        poly3d, 
        # edgecolor="k", 
        facecolors=colors, 
        linewidth=0.2, 
        alpha=0.15, 
        shade=True)
    # ax.add_collection3d(collection)

    ax.scatter(
        points[:,dims[0]],
        points[:,dims[1]],
        points[:,dims[2]],
        # alpha=0.7,
        **markers_3d[shape]
    )
ax.view_init(elev=25, azim=-45)  # Elevation and azimuth in degrees

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel(labels[dims[0]])
ax.set_ylabel(labels[dims[1]])
ax.set_zlabel(labels[dims[2]])
# ax.invert_yaxis()
ax.set_aspect("equal")
# fig.savefig("pareto_3d_APQ.pdf")

# %%
dims = [1,2,3]
alpha_i = [1.7, 1.5, 1.7, 1.7]

# Create a Matplotlib 3D plot
# fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(gs[1,1], projection='3d')

for i, shape in enumerate(shapes):
    points = df_results[df_results["shape"]==shape].to_numpy()[:,-4:].astype(np.double)

    # Generate the alpha hull
    hull = alphashape(points[:,dims], alpha_i[i])

    vertices = np.array(hull.vertices)
    faces = np.array(hull.faces)

    faces_pv = np.hstack([[3] + list(face) for face in faces])
    hull_mesh = pv.PolyData(vertices, faces_pv) #.smooth_taubin(n_iter=5, pass_band=0.5)

    points_hull = hull_mesh.points  # Array of shape (N, 3), where N is the number of points
    faces_hull = hull_mesh.faces.reshape(-1, 4)[:, 1:]  

    # Map intensities to a color map
    colors = np.array(colors_i[i] * len(faces_hull))

    # Convert faces to a format compatible with Matplotlib's Poly3DCollection
    poly3d = np.array([[points_hull[vertex] for vertex in face] for face in faces_hull])

    collection = Poly3DCollection(
        poly3d, 
        # edgecolor="k", 
        facecolors=colors, 
        linewidth=0.2, 
        alpha=0.15, 
        shade=True)
    # ax.add_collection3d(collection)

    ax.scatter(
        points[:,dims[0]],
        points[:,dims[1]],
        points[:,dims[2]],
        # alpha=1,
        **markers_3d[shape]
    )
    ax.view_init(elev=25, azim=-1*45)  # Elevation and azimuth in degrees

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel(labels[dims[0]])
ax.set_ylabel(labels[dims[1]])
ax.set_zlabel(labels[dims[2]])
ax.set_aspect("equal")

handles, labels = ax.get_legend_handles_labels()
# order = ["ellipse", "triangle", "rectangle", "cross"]
# handles = [handles[labels.index(label)] for label in order]
# labels = [label for label in order]

fig.legend(
    handles, 
    labels, 
    loc='lower center', 
    bbox_to_anchor=(0.5, 0.), 
    ncol=4,
    frameon=False,
    prop={"size": 12},
    markerscale=2)

# fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.1))

# ax.invert_yaxis()
# fig.savefig("pareto_3d_dPQ.pdf")
# fig.savefig("../manuscript/figures/pareto_3d.2.pdf")

display(fig)

# %%
# from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights

# dm = HighTradeoffPoints()

shape = "ellipse"
# f_shape = df_results[df_results["shape"]==shape].to_numpy()[:,-4:].astype(np.double)
f_shape = df_n.to_numpy()[:,-4:].astype(np.double)
# idx_hto = dm(f_shape)

asf = ASF()
weights = np.array([0.25, 0.25, 0.25, 0.25])
idx_asf = asf.do(f_shape, 1/weights).argmin()

pw = PseudoWeights(weights)
idx_pw, p_weights = pw.do(f_shape, return_pseudo_weights=True)
print(p_weights.shape)
print(idx_asf, idx_pw)

# %%
decomp = ASF()

#  %%
w_i = np.linspace(0.05,0.95,25)
# %%
w_1, w_2, w_3 = np.meshgrid(w_i, w_i, w_i)
w_123 = np.column_stack((w_1.reshape(-1,1), w_2.reshape(-1,1), w_3.reshape(-1,1)))

# w_123 = np.random.rand(5000, 3)
# %%
w_123_ = w_123[np.sum(w_123, axis=-1) <= 1.]
w_4 = 1 - np.sum(w_123_, axis=1)

w_1234 = np.column_stack((w_123_, w_4))[:,[-1,0,1,2]]
# print(np.sum(w_1234, axis=1)) # check -> OK
print(len(w_1234))

# %%
decomp = ASF()
f_merged = df_results.to_numpy()[:,-4:].astype(np.double)

shapes_asf = []
for w in w_1234:
    idx_asf = decomp.do(f_merged, 1/w).argmin()
    idx_pw = PseudoWeights(w).do(f_merged)
    idx_ = idx_pw
    # idx_ = idx_asf
    shapes_asf.append(df_results.to_numpy()[idx_,0])
# %%
fact = 2
markers_asf = dict(
    ellipse=dict(marker="o", s=5*fact, c="C0"),
    rect=dict(marker="s", s=5*fact, c="C1"),
    tri=dict(marker="^", s=5*fact, c="C2"),
    cross=dict(marker="+", s=15*fact, c="C3"),
    )

fig = px.scatter_3d(
    df_results,
    x=w_1234[:,0],
    y=w_1234[:,1],
    z=w_1234[:,2],
    size=np.full_like(w_1234[:,0], 200.),
    color=shapes_asf,
    width=300*2,
    height=300*2,
    opacity=0.4
)
fig.show()
# %%
# %matplotlib ipympl
# %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}
fig = plt.figure(figsize=(8., 8.), constrained_layout=False)
# fig.subplots_adjust(wspace=0.2)

gs = GridSpec(2, 2, figure=fig, hspace=0.05, top=1, bottom=0.0, left=0.0, right=0.94)  # Adjust spacing

ax = fig.add_subplot(gs[0,0], projection='3d')   

for shape in shapes:
    idx_shape = np.equal(shapes_asf, shape)
    ax.scatter(
        w_1234[idx_shape,0],
        w_1234[idx_shape,1],
        w_1234[idx_shape,2],
        # s=1,
        # c="C"+str(i),
        alpha=0.75,
        **markers_asf[shape],
    )
ax.view_init(elev=25, azim=1*45)  # Elevation and azimuth in degrees

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
# ax.set_xlabel(labels[dims[0]])
# ax.set_ylabel(labels[dims[1]])
# ax.set_zlabel(labels[dims[2]])
# ax.invert_yaxis()
ax.set_aspect("equal")

# %%
vertices = np.array([
    [0, 0, 0],  # Vertex 1
    [1, 0, 0],  # Vertex 2
    [0, 1, 0],  # Vertex 3
    [0, 0, 1],  # Vertex 4
])

# Define the faces of the tetrahedron by connecting the vertices
faces = [
    [vertices[0], vertices[1], vertices[2]],  # Face 1
    [vertices[0], vertices[1], vertices[3]],  # Face 2
    [vertices[0], vertices[2], vertices[3]],  # Face 3
    [vertices[1], vertices[2], vertices[3]]   # Face 4
]


# %%
dims_w = [1,3,2]
labels_w = [r"$w_A / 1$", r"$w_d / 1$", r"$w_P / 1$", r"$w_Q / 1$"]
fig = plt.figure(figsize=(8., 8.), constrained_layout=False)
# fig.subplots_adjust(wspace=0.2)

gs = GridSpec(2, 2, figure=fig, hspace=0., top=1., bottom=0.0, left=0.04, right=1)  # Adjust spacing
idx_grid = [(0,0),(0,1),(1,0),(1,1)]

for i, shape in enumerate(["ellipse", "rect", "tri", "cross"]):
    ax = fig.add_subplot(gs[*idx_grid[i]], projection='3d') 
    poly3d = Poly3DCollection(faces, alpha=0.1, linewidths=0, edgecolors='r')
    poly3d.set_facecolor([0.5, 0.5, 0.5])  
    ax.add_collection3d(poly3d)
    idx_shape = np.equal(shapes_asf, shape)
    ax.scatter(
        w_1234[idx_shape,dims_w[0]],
        w_1234[idx_shape,dims_w[1]],
        w_1234[idx_shape,dims_w[2]],
        # s=1,
        # c="C"+str(i),
        alpha=0.5,
        **markers_asf[shape],
    )
    ax.view_init(elev=25, azim=1*55)  # Elevation and azimuth in degrees

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xlabel(labels_w[dims_w[0]])
    ax.set_ylabel(labels_w[dims_w[1]])
    ax.set_zlabel(labels_w[dims_w[2]])
    # ax.invert_yaxis()
    ax.set_aspect("equal")
# fig.savefig(r"../manuscript/figures/weights_123_pw.pdf")

# %%
l = 10

table_sol = ""

weights = [
    np.ones(4) / 4,
    np.array((1.,0.,0.,0.)),
    np.array((0.,1.,0.,0.)),
    np.array((0.,0.,1.,0.)),
    np.array((0.,0.,0.,1.)),
]
for w in weights:
    idx_asf = decomp.do(f_merged, 1/w).argmin()
    idx_pw, pw = PseudoWeights(w).do(f_merged, return_pseudo_weights=True)
    idx_ = idx_pw
    # idx_ = idx_asf
    # print(df_results.to_numpy()[idx_asf])
    shape = df_results.to_numpy()[idx_][0]

    X = df.to_numpy()[idx_,1:4].reshape(-1,3).astype(np.double)
    a = l / X[:,0]
    b = a * X[:,1]
    h = a * X[:,2]

    F = df.to_numpy()[idx_,4:].reshape(-1,4).astype(np.double)

    print(f"{shape}:")
    print(X[:,[0,1,2]])
    print(F[:])
    print(np.column_stack((a, b, h)))
    print(f"{np.array2string(pw[idx_], formatter={'float': lambda x: f'{x:.3f}'}, separator=', ')}")

    sol_weight = f"{shape:<8} & \t \\num{{{X[0,0]:.6}}} & \t \\num{{{X[0,1]:.6}}} & \t \\num{{{X[0,2]:.6}}} " \
        + f"&  \\num{{{F[0,0]:.6e}}} & \t \\num{{{F[0,1]:.6e}}} & \t \\num{{{-F[0,2]:.6e}}} & \t \\num{{{-F[0,3]:.6e}}}"
    table_sol += sol_weight + " \\\\ \n"
print(table_sol)

# %%
max_weight = 0.4
weights_ = np.array([max_weight, (1-max_weight)/3, (1-max_weight)/3, (1-max_weight)/3])
weights = [np.roll(weights_, i) for i in range(0,4)]

table_sol = ""

for w in weights:
    idx_asf = decomp.do(f_merged, 1/w).argmin()
    idx_pw = PseudoWeights(w).do(f_merged)
    idx_ = idx_pw
    # print(df_results.to_numpy()[idx_asf])
    shape = df_results.to_numpy()[idx_][0]

    X = df.to_numpy()[idx_,1:4].reshape(-1,3).astype(np.double)
    a = l / X[:,0]
    b = a * X[:,1]
    h = a * X[:,2]

    F = df.to_numpy()[idx_,4:].reshape(-1,4).astype(np.double)

    print(f"{shape}:")
    print(X[:,[0,1,2]])
    print(F[:])
    print(np.column_stack((a, b, h)))
    print(f"{np.array2string(pw[idx_], formatter={'float': lambda x: f'{x:.3f}'}, separator=', ')}")

    sol_weight = f"{shape:<8} & \t \\num{{{X[0,0]:.6}}} & \t \\num{{{X[0,1]:.6}}} & \t \\num{{{X[0,2]:.6}}} " \
        + f"&  \\num{{{F[0,0]:.6e}}} & \t \\num{{{F[0,1]:.6e}}} & \t \\num{{{-F[0,2]:.6e}}} & \t \\num{{{-F[0,3]:.6e}}}"
    table_sol += sol_weight + " \\\ \n"
print(table_sol)
weights_04 = weights

# %%
max_weight = 0.1
weights_ = np.array([max_weight, (1-max_weight)/3, (1-max_weight)/3, (1-max_weight)/3])
weights = [np.roll(weights_, i) for i in range(0,4)]

table_sol = ""

for w in weights:
    idx_asf = decomp.do(f_merged, 1/w).argmin()
    idx_pw = PseudoWeights(w).do(f_merged)
    idx_ = idx_pw
    # print(df_results.to_numpy()[idx_asf])
    shape = df_results.to_numpy()[idx_][0]

    X = df.to_numpy()[idx_,1:4].reshape(-1,3).astype(np.double)
    a = l / X[:,0]
    b = a * X[:,1]
    h = a * X[:,2]

    F = df.to_numpy()[idx_,4:].reshape(-1,4).astype(np.double)

    print(f"{shape}:")
    print(X[:,[0,1,2]])
    print(F[:])
    print(np.column_stack((a, b, h)))
    print(f"{np.array2string(pw[idx_], formatter={'float': lambda x: f'{x:.3f}'}, separator=', ')}")

    sol_weight = f"{shape:<8} & \t \\num{{{X[0,0]:.6}}} & \t \\num{{{X[0,1]:.6}}} & \t \\num{{{X[0,2]:.6}}} " \
        + f"&  \\num{{{F[0,0]:.6e}}} & \t \\num{{{F[0,1]:.6e}}} & \t \\num{{{-F[0,2]:.6e}}} & \t \\num{{{-F[0,3]:.6e}}}"
    table_sol += sol_weight + " \\\ \n"
print(table_sol)

weights_01 = weights

# %%
df_results = df
fact = 2
alpha = 0.75
markers_X = dict(
    ellipse=dict(marker="o", s=5*fact, c="C0", alpha=alpha, clip_on=False, label=r"\textrm{ellipse}"),
    rect=dict(marker="s", s=5*fact, c="C1", alpha=alpha, clip_on=False, label=r"\textrm{rectangle}"),
    tri=dict(marker="^", s=5*fact, c="C2", alpha=alpha, clip_on=False, label=r"\textrm{triangle}"),
    cross=dict(marker="+", s=12*fact, c="C3", alpha=alpha, clip_on=False, label=r"\textrm{cross}"),
    )

fig, ax = plt.subplots(1,1,subplot_kw={'projection': '3d'},figsize=(4,4))
for shape in shapes:
    df_results_shape = df_results.query(f"shape=='{shape}'")
    ax.scatter(df_results_shape.alpha, df_results_shape.beta, df_results_shape.eta, **markers_X[shape])
# ax.scatter(df_rect.alpha, df_rect.beta, df_rect.eta, **markers_X["rect"])
# ax.scatter(df_ellipse.alpha, df_ellipse.beta, df_ellipse.eta, **markers_X["ellipse"])
# ax.scatter(df_tri.alpha, df_tri.beta, df_tri.eta, **markers_X["tri"])
ax.set_zlim(10, 50)
ax.set_xlim(0.1, 2)
ax.set_ylim(0.1, 0.5)
ax.set_box_aspect((1, 1, 1))
ax.set_zlabel(r"$\eta / 1$")
ax.set_xlabel(r"$\alpha / 1$")
ax.set_ylabel(r"$\beta / 1$")
ax.view_init(elev=25, azim=-1*45)
plt.show()

# %%
# eta_cross = (df_cross.eta - 10) / 40
# alpha_cross = (df_cross.alpha - 0.1) / 0.9
# beta_cross = (df_cross.beta - 0.1) / 0.4

# eta_rect = (df_rect.eta - 10) / 40
# alpha_rect = (df_rect.alpha - 0.1) / 0.9
# beta_rect = (df_rect.beta - 0.1) / 0.4

# eta_ellipse = (df_ellipse.eta - 10) / 40
# alpha_ellipse = (df_ellipse.alpha - 0.1) / 0.9
# beta_ellipse = (df_ellipse.beta - 0.1) / 0.4

# eta_tri = (df_tri.eta - 10) / 40
# alpha_tri = (df_tri.alpha - 0.1) / 1.9
# beta_tri = (df_tri.beta - 0.1) / 0.4

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8,8*2/3/1.75))
for shape in shapes:
    df_results_shape = df_n.query(f"shape=='{shape}'")
    ax[0].scatter(df_results_shape.eta, df_results_shape.alpha, **markers_X[shape])
# ax[0].scatter(eta_cross, alpha_cross, **markers_X["cross"])
# ax[0].scatter(eta_rect, alpha_rect, **markers_X["rect"])
# ax[0].scatter(eta_ellipse, alpha_ellipse, **markers_X["ellipse"])
# ax[0].scatter(eta_tri, alpha_tri, **markers_X["tri"])
ax[0].set_xlabel(r"$\tilde \eta / 1$")
ax[0].set_ylabel(r"$\tilde \alpha / 1$")
ax[0].set_xlim(0,1)
ax[0].set_ylim(0,1)
ax[0].set_xticks([0,0.2,0.4,0.6,0.8,1])
ax[0].set_aspect("equal")

for shape in shapes:
    df_results_shape = df_n.query(f"shape=='{shape}'")
    ax[1].scatter(df_results_shape.eta, df_results_shape.beta, **markers_X[shape])
ax[1].set_xlabel(r"$\tilde \eta / 1$")
ax[1].set_ylabel(r"$\tilde \beta / 1$")
ax[1].set_xlim(0,1)
ax[1].set_ylim(0,1)
ax[1].set_aspect("equal")

for shape in shapes:
    df_results_shape = df_n.query(f"shape=='{shape}'")
    ax[2].scatter(df_results_shape.alpha, df_results_shape.beta, **markers_X[shape])
ax[2].set_xlabel(r"$\tilde \alpha / 1$")
ax[2].set_ylabel(r"$\tilde \beta / 1$")
ax[2].set_xlim(0,1)
ax[2].set_ylim(0,1)
ax[2].set_aspect("equal")

handles, labels = ax[0].get_legend_handles_labels()

fig.legend(
    handles, 
    labels, 
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.025), 
    ncol=4,
    frameon=False,
    prop={"size": 12},
    markerscale=1.5)

# fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.1))
fig.tight_layout()
fig.subplots_adjust(top=1, bottom=0.175)
# fig.savefig(r"../manuscript/figures/pareto_2d_variables.pdf")
plt.show()

# %%
df_results = df_n
l = 10
table_extremal_designs = ""
indices = [
    np.argmin(df_results.eta),
    np.argmax(df_results.eta),
    np.argmin(df_results.alpha),
    np.argmax(df_results.alpha),
    np.argmin(df_results.beta),
    np.argmax(df_results.beta),
]

df_results = df
for idx in indices:
    shape = df_results.iloc[idx,0]
    X = df_results.to_numpy()[idx,1:4].reshape(-1,3).astype(np.double)
    a = l / X[:,0]
    # b = a * (1.9/0.9*(X[:,1]-0.1) + 0.1) if shape == "tri" else a * X[:,1] 
    b = a * X[:,1] 
    h = a * X[:,2]

    F = df_results.to_numpy()[idx,4:].reshape(-1,4).astype(np.double)
    # nF = (F - fl) / (fu - fl)
    # F = F * (fu - fl) + fl

    extremal = f"\\num{{{X[0,0]:.6}}} & \t \\num{{{X[0,1]:.6}}} & \t \\num{{{X[0,2]:.6}}}  & \t {shape:<8} \t " \
        + f"&  \\num{{{F[0,0]:.6e}}} & \t \\num{{{F[0,1]:.6e}}} & \t \\num{{{-F[0,2]:.6e}}} & \t \\num{{{-F[0,3]:.6e}}}"
    table_extremal_designs += extremal + " \\\ \n"
print(table_extremal_designs)

# %%
# get some understanding for pseudo weights
y_ = np.array((1.5, 0.2))
y_ideal = np.array((0.1,0.2))
y_nadir = np.array((1.5,2))

(y_nadir - y_) / (y_nadir - y_ideal) / (np.sum((y_nadir - y_) / (y_nadir - y_ideal)))
# %%
y_n = (y_ - y_ideal) / (y_nadir - y_ideal)
(1 - y_n) / np.sum(1 - y_n)

# %%
# check pseudo weights of extremal points in objective space
idx_A = np.argmin(df.f_A)
idx_d = np.argmin(df.f_d)
idx_P = np.argmin(df.f_P)
idx_Q = np.argmin(df.f_Q)
print(df.iloc[idx_A])
print(df.iloc[idx_d])
print(idx_A, idx_d)

idx_eta = np.argmax(df.eta)
idx_eta = np.argmax(df.eta)

idx_pw_A, pw = PseudoWeights([1.,0.,0.,0.]).do(f_merged, return_pseudo_weights=True)
idx_pw_d, pw = PseudoWeights([0.,1.,0.,0.]).do(f_merged, return_pseudo_weights=True)
idx_pw_P, pw = PseudoWeights([0.,0.,1.,0.]).do(f_merged, return_pseudo_weights=True)
idx_pw_Q, pw = PseudoWeights([0.,0.,0.,1.]).do(f_merged, return_pseudo_weights=True)
# %%
print(df_n.iloc[idx_pw_A])
print(pw[idx_pw_A], pw[idx_A])
print(pw[idx_eta])

# %%
print(idx_P, idx_pw_P)
print(pw[idx_pw_P])
# %%
print(idx_Q, idx_pw_Q)
print(pw[idx_pw_Q])

# %%
fig = plt.figure(figsize=(4., 4.), constrained_layout=False)
# fig.subplots_adjust(wspace=0.2)

gs = GridSpec(1, 1, figure=fig, hspace=0., top=1, bottom=0.05, left=0.1, right=1)  # Adjust spacing
ax = fig.add_subplot(gs[0,0], projection='3d')   

poly3d = Poly3DCollection(faces, alpha=0.1, linewidths=0, edgecolors='r')
poly3d.set_facecolor([0.5, 0.5, 0.5])  
ax.add_collection3d(poly3d)
for i, pw_ in enumerate(pw):
    ax.scatter(
        pw_[dims_w[0]],
        pw_[dims_w[1]],
        pw_[dims_w[2]],
        # s=1,
        # c="C"+str(i),
        alpha=0.75,
        **markers_asf[df.iloc[i]["shape"]],
    )
ax.view_init(elev=25, azim=1*45)  # Elevation and azimuth in degrees

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel(labels_w[dims_w[0]])
ax.set_ylabel(labels_w[dims_w[1]])
ax.set_zlabel(labels_w[dims_w[2]])

ax.set_aspect("equal")
# fig.savefig("../manuscript/figures/weights_pareto_set.pdf")

# %%
idx_A_asf = decomp.do(f_merged, 1/np.array([1.,0.,0.,0.])).argmin()
idx_d_asf = decomp.do(f_merged, 1/np.array([0.,1.,0.,0.])).argmin()
idx_P_asf = decomp.do(f_merged, 1/np.array([0.,0.,1.,0.])).argmin()
idx_Q_asf = decomp.do(f_merged, 1/np.array([0.,0.,0.,1.])).argmin()
# %%
print(idx_A_asf, df.iloc[idx_A_asf])
print(df.f_A.argmin(), df.iloc[df.f_A.argmin()])
# %%
print(idx_d_asf, df.iloc[idx_d_asf])
print(df.f_d.argmin(), df.iloc[df.f_d.argmin()])
# %%
print(idx_P_asf, df.iloc[idx_P_asf])
print(df.f_P.argmin(), df.iloc[df.f_P.argmin()])
# %%
print(idx_Q_asf, df.iloc[idx_Q_asf])
print(df.f_Q.argmin(), df.iloc[df.f_Q.argmin()])

# %%
idx_uniform_asf = decomp.do(f_merged, 1/np.array([0.25,0.25,0.25,0.25])).argmin()
print(idx_uniform_asf, df.iloc[idx_uniform_asf])

# %%
# %matplotlib ipympl
# %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}
idx_shapes = [
    np.argmin(df_results.eta),
    np.argmin(df_results.alpha),
    np.argmin(df_results.beta),
    np.argmax(df_results.eta),
    np.argmax(df_results.alpha),
    np.argmax(df_results.beta),
    PseudoWeights([1/4, 1/4, 1/4, 1/4]).do(f_merged),
    idx_pw_A,
    idx_pw_d,
    idx_pw_P,
    idx_pw_Q,
] + [PseudoWeights(w).do(f_merged) for w in weights_04 + weights_01] + [idx_A_asf]

labels_pw = [
    dict(x=pw[idx_shapes[0],dims_w[0]], y=pw[idx_shapes[0],dims_w[1]]+0.03, z=pw[idx_shapes[0],dims_w[2]]-0.05, s="$\\mathrm{(a)}$"),
    dict(x=pw[idx_shapes[1],dims_w[0]], y=pw[idx_shapes[1],dims_w[1]]-0.15, z=pw[idx_shapes[1],dims_w[2]]-0.02, s="$\\mathrm{(b)}$"),
    dict(x=pw[idx_shapes[2],dims_w[0]]+0.05, y=pw[idx_shapes[2],dims_w[1]]+0.08, z=pw[idx_shapes[2],dims_w[2]]-0.02, s="$\\mathrm{(c)}$"),
    dict(x=pw[idx_shapes[3],dims_w[0]], y=pw[idx_shapes[3],dims_w[1]], z=pw[idx_shapes[3],dims_w[2]]-0.08, s="$\\mathrm{(d)}$"),
    dict(x=pw[idx_shapes[4],dims_w[0]]-0.03, y=pw[idx_shapes[4],dims_w[1]], z=pw[idx_shapes[4],dims_w[2]]-0.02, s="$\\mathrm{(e)}$"),
    dict(x=pw[idx_shapes[5],dims_w[0]], y=pw[idx_shapes[5],dims_w[1]]+0.03, z=pw[idx_shapes[5],dims_w[2]]+0.03, s="$\\mathrm{(f)}$"),
    dict(x=pw[idx_shapes[6],dims_w[0]], y=pw[idx_shapes[6],dims_w[1]]+0.02, z=pw[idx_shapes[6],dims_w[2]]+0.01, s="$\\mathrm{(g)}$"),
    dict(x=pw[idx_shapes[7],dims_w[0]], y=pw[idx_shapes[7],dims_w[1]]-0.19, z=pw[idx_shapes[7],dims_w[2]]-0.09, s="$\\mathrm{(h)}$"),
    dict(x=pw[idx_shapes[8],dims_w[0]], y=pw[idx_shapes[8],dims_w[1]]-0.15, z=pw[idx_shapes[8],dims_w[2]]-0.05, s="$\\mathrm{(i)}$"),
    dict(x=pw[idx_shapes[9],dims_w[0]]-0.04, y=pw[idx_shapes[9],dims_w[1]], z=pw[idx_shapes[9],dims_w[2]], s="$\\mathrm{(j)}$"),
    dict(x=pw[idx_shapes[10],dims_w[0]]-0.04, y=pw[idx_shapes[10],dims_w[1]], z=pw[idx_shapes[10],dims_w[2]], s="$\\mathrm{(k)}$"),
    dict(x=pw[idx_shapes[11],dims_w[0]]+0.09, y=pw[idx_shapes[11],dims_w[1]]-0.05, z=pw[idx_shapes[11],dims_w[2]]+0.03, s="$\\mathrm{(l)}$"),
    dict(x=pw[idx_shapes[12],dims_w[0]]+0.05, y=pw[idx_shapes[12],dims_w[1]]-0.15, z=pw[idx_shapes[12],dims_w[2]]-0.05, s="$\\mathrm{(m)}$"),
    dict(x=pw[idx_shapes[13],dims_w[0]], y=pw[idx_shapes[13],dims_w[1]]+0.02, z=pw[idx_shapes[13],dims_w[2]]+0.02, s="$\\mathrm{(n)}$"),
    dict(x=pw[idx_shapes[14],dims_w[0]]-0.04, y=pw[idx_shapes[14],dims_w[1]], z=pw[idx_shapes[14],dims_w[2]]-0.08, s="$\\mathrm{(o)}$"),
    dict(x=pw[idx_shapes[15],dims_w[0]]+0.06, y=pw[idx_shapes[15],dims_w[1]], z=pw[idx_shapes[15],dims_w[2]]+0.06, s="$\\mathrm{(p)}$"),
    dict(x=pw[idx_shapes[16],dims_w[0]]-0.03, y=pw[idx_shapes[16],dims_w[1]], z=pw[idx_shapes[16],dims_w[2]], s="$\\mathrm{(q)}$"),
    dict(x=pw[idx_shapes[17],dims_w[0]], y=pw[idx_shapes[17],dims_w[1]]+0.02, z=pw[idx_shapes[17],dims_w[2]]-0.08, s="$\\mathrm{(r)}$"),
    dict(x=pw[idx_shapes[18],dims_w[0]], y=pw[idx_shapes[18],dims_w[1]]-0.17, z=pw[idx_shapes[18],dims_w[2]]-0.05, s="$\\mathrm{(s)}$"),
    dict(x=pw[idx_shapes[19],dims_w[0]], y=pw[idx_shapes[19],dims_w[1]]-0.12, z=pw[idx_shapes[19],dims_w[2]]-0.15, s="$\\mathrm{(t)}$"),
]
# # %%
fig = plt.figure(figsize=(4., 4.), constrained_layout=False)
# fig.subplots_adjust(wspace=0.2)

gs = GridSpec(1, 1, figure=fig, hspace=0., top=1, bottom=0.05, left=0.1, right=1)  # Adjust spacing
ax = fig.add_subplot(gs[0,0], projection='3d')   

poly3d = Poly3DCollection(faces, alpha=0.1, linewidths=0, edgecolors='r')
poly3d.set_facecolor([0.5, 0.5, 0.5])  
ax.add_collection3d(poly3d)
for i, idx in enumerate(idx_shapes):
    ax.scatter(
        pw[idx,dims_w[0]],
        pw[idx,dims_w[1]],
        pw[idx,dims_w[2]],
        # s=1,
        # c="C"+str(i),
        alpha=1,
        **markers_asf[df.iloc[idx]["shape"]],
    )
    ax.text(**labels_pw[i], c=markers_asf[df.iloc[idx]["shape"]]["c"])
    
ax.view_init(elev=25, azim=1*45)  # Elevation and azimuth in degrees

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel(labels_w[dims_w[0]])
ax.set_ylabel(labels_w[dims_w[1]])
ax.set_zlabel(labels_w[dims_w[2]])

ax.set_aspect("equal")
# fig.savefig("../manuscript/figures/weights_a-t.pdf")

# %%
print(np.min(pw, axis=0))
print(np.max(pw, axis=0))
# %%
