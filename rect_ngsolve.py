# %%
import os
os.environ["OMP_NUM_THREADS"] = "2"
from netgen.occ import *
from ngsolve import *
import numpy as np

SetNumThreads(2)

def compute_rect(width, height, wall_thickness, length, E):
    mesh = create_geometry(width, height, wall_thickness)
    if mesh == None:
        A = np.inf
        d_in = np.inf
        P_crit = -np.inf
        f = -np.inf
    else:
        A = area(mesh)
        d_in = incircle(width, height)
        Imin = np.min(second_moments(mesh))
        P_crit = critical_load(length, Imin, E)
        f = flow(mesh, length)
    return A, d_in, P_crit, f

def create_geometry(width, height, wall_thickness):
    if width - 2*wall_thickness < 1e-2 or height - 2*wall_thickness < 1e-2:
        return None
    
    wp = WorkPlane()
    rect = wp.RectangleC(width, height).Face()
    hole = wp.RectangleC(width-2*wall_thickness, height-2*wall_thickness).Face()
    rect.name = "cannula"
    hole.name = "fluid"
    
    for edge in rect.edges:
        edge.name = "outer"

    for edge in hole.edges:
        edge.name = "inner"

    cannula = rect-hole
    cannula.maxh = wall_thickness/2
    hole.maxh = np.min([(width-2*wall_thickness)/3, (height-2*wall_thickness)/3])

    geo = OCCGeometry(Glue([cannula, hole]), dim=2)
    ng_mesh = geo.GenerateMesh()
    mesh = Mesh(ng_mesh)
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
        Draw(U, mesh, "u")
    return flow

def incircle(width, height):
    return np.min([width, height])
