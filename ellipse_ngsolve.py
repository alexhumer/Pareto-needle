# %%
import os
os.environ["OMP_NUM_THREADS"] = "2"
from netgen.occ import *
from ngsolve import *
import numpy as np

SetNumThreads(2)

def compute_ellipse(major, minor, wall_thickness, length, E):
    mesh = create_geometry(major, minor, wall_thickness)
    if mesh == None:
        A = np.inf
        d_in_max = np.inf
        P_crit = -np.inf
        f = -np.inf
    else:
        A = area(mesh)
        d_in_max = incircle(major, minor)
        Imin = np.min(second_moments(mesh))
        P_crit = critical_load(length, Imin, E)
        f = flow(mesh, length)
    return A, d_in_max, P_crit, f

def create_geometry(major, minor, wall_thickness):
    if minor - 2*wall_thickness < 1e-2:
        return None
     
    ellipse = Ellipse(gp_Ax2d((0,0), (1,0)), major/2, minor/2).Face()
    hole = ellipse.Scale((0,0,0), 1 - 2*wall_thickness/minor)

    ellipse.name = "cannula"
    hole.name = "fluid"
    
    for edge in ellipse.edges:
        edge.name = "outer"

    for edge in hole.edges:
        edge.name = "inner"

    cannula = ellipse-hole
    cannula.maxh = wall_thickness/2
    hole.maxh = (minor-2*wall_thickness)/4 

    geo = OCCGeometry(Glue([cannula, hole]), dim=2)
    ng_mesh = geo.GenerateMesh()
    mesh = Mesh(ng_mesh)
    mesh.Curve(3)
    return mesh

def critical_load(l, Imin, E=1.):
    EI = E*Imin
    return np.pi**2 * EI/(0.699*l)

def area(mesh):
    with TaskManager():
        A = Integrate(1, mesh)
    return A

def circumference(mesh):
    with TaskManager():
        u = Integrate(1, mesh, definedon=mesh.Boundaries("outer"))
    return u

def second_moments(mesh):
    with TaskManager():
        A = Integrate(1, mesh, definedon=mesh.Materials("cannula"))
        xc = 1/A * Integrate(x, mesh, definedon=mesh.Materials("cannula"))
        yc = 1/A * Integrate(y, mesh, definedon=mesh.Materials("cannula"))

        Ixx = Integrate((y-yc)**2, mesh, definedon=mesh.Materials("cannula"))
        Iyy = Integrate((x-xc)**2, mesh, definedon=mesh.Materials("cannula"))
        Ixy = -Integrate((x-xc)*(y-yc), mesh, definedon=mesh.Materials("cannula"))
        w, v = np.linalg.eig(np.array((Ixx, Ixy, Ixy, Iyy)).reshape(-1,2))
    return w

def flow(mesh, l, delta_p=1, mu=1):
    G = -delta_p/l
    mu = 1

    Vu = H1(mesh, order=3, dirichlet='inner', definedon=mesh.Materials('fluid'))
    u,v = Vu.TnT()
    U = GridFunction(Vu)

    a = BilinearForm(Vu, symmetric=True)
    a += SymbolicBFI(InnerProduct(grad(u), grad(v)), definedon=mesh.Materials('fluid'))
    a += SymbolicBFI(G/mu*v, definedon=mesh.Materials('fluid'))

    with TaskManager():
        res = U.vec.CreateVector()
        a.Assemble()
        a.Apply(U.vec, res)
        U.vec.data -= a.mat.Inverse(Vu.FreeDofs()) * res
        flow = Integrate(U, mesh, definedon=mesh.Materials('fluid'))
    return flow

def incircle(major, minor):
    return minor
