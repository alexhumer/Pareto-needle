# %%
import numpy as np
import pickle
# %%
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

import multiprocessing
from pymoo.core.problem import StarmapParallelization
from pymoo.core.variable import Real, Integer, Choice

from ellipse_ngsolve import compute_ellipse
from rect_ngsolve import compute_rect
from tri_ngsolve import compute_tri
from cross_ngsolve import compute_cross

# %%
class Cannula(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(
            vars = {
                "shape": Choice(options=["ellipse", "rect", "tri", "cross"]),
                "eta": Real(bounds=(10., 50.)),
                "alpha": Real(bounds=(0.1, 1)),
                "beta": Real(bounds=(0.1, 0.5)),
            }, 
            n_obj=4, 
            n_ieq_constr=0, 
            **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        l = 1.
        E = 1
        mu = 1
        delta_p = 1

        shape, eta, alpha, beta = x["shape"], x["eta"], x["alpha"], x["beta"]
        if shape == "ellipse":
            major = l / eta 
            minor = major * alpha  
            h = major * beta       
            A, d_in_max, P_crit, U = compute_ellipse(major, minor, h, l, E)
        elif shape == "rect":
            a = l / eta
            b = a * alpha
            h = a * beta
            A, d_in_max, P_crit, U = compute_rect(a, b, h, l, E)
        elif shape == "tri":
            a = l / eta
            b = a * (1.9/0.9*(alpha-0.1) + 0.1) 
            h = a * beta
            A, d_in_max, P_crit, U = compute_tri(a, b, h, l, E)
        elif shape == "cross":
            width = l / eta
            bar = width * alpha
            h = width * beta
            A, d_in_max, P_crit, U = compute_cross(width, bar, h, l, E)

        if 1:
            l_ref = l
        else:
            l_ref = major

        t_ref = mu / delta_p
        F_ref = E*l_ref**2

        A_nd = A / l_ref**2             # min
        d_in_max_nd = d_in_max / l_ref  # min
        P_nd = P_crit / F_ref           # max
        U_nd = U * t_ref / l_ref**4     # max
        
        out["F"] = [A_nd, d_in_max_nd, -P_nd, -U_nd]

# %%
if __name__ == '__main__':
    
    method = "AGEMOEA2"

    if method == "NSGA2":
        algorithm = NSGA2(
            pop_size=1000,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

    elif method == "AGEMOEA2":
        from pymoo.algorithms.moo.age2 import AGEMOEA2
        from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination

        algorithm = AGEMOEA2(
            pop_size=1000,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )

    n_processes = 8 
    pool = multiprocessing.Pool(n_processes)
    runner = StarmapParallelization(pool.starmap)

    problem = Cannula(elementwise_runner=runner)

    res = minimize(problem,
                algorithm,
                termination=("n_gen", 5*100),
                seed=1,
                copy_algorithm=False,
                save_history=True,
                verbose=True)

    
    path = r"./"
    fname = path + "res_all.pkl"

    if 1:
        with open(fname, "wb") as f:
            pickle.dump((res.X, res.F), f)

    pool.close()
