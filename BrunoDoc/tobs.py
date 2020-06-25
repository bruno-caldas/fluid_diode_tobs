from oct2py import octave
from dolfin import *
from BrunoDoc.read_param_file import *
# if linha[9][:2].upper() == 'DA': from dolfin_adjoint import *
import numpy as np
import os

import cplex
from cplex.exceptions import CplexError
import BrunoDoc.smooth as sm

octave.addpath(os.getcwd()+'/BrunoDoc')
def create(nvar,
            x_L_int,
            x_U_int,
            cst_num,
            acst_L,
            acst_U,
            constraints_nnz,
            zero,
            obj_fun,
            dobj_fun,
            cst_fun,
            jac_fun):
    obj = TobsObj()
    obj.nvar = nvar
    obj.cst_num = cst_num
    obj.x_L_int = x_L_int
    obj.x_U_int = x_U_int
    obj.cst_L = acst_L
    obj.cst_U = acst_U
    obj.constraints_nnz = constraints_nnz
    obj.obj_fun = obj_fun
    obj.dobj_fun = dobj_fun
    obj.cst_fun = cst_fun
    obj.jac_fun = jac_fun

    return obj

class TobsObj():
    """
    This Class supplies a python object that has all Ipopt needs,
    such as ObjFun, Gradientes and Constraints
    """
    iteration = 0
    iterations = 5
    subiterations = 5

    def __init__(self):
        pass

    def reshape_to_matlab(self):
        self.x_L = np.ones((self.nvar), dtype=np.float) * self.x_L_int
        self.x_U = np.ones((self.nvar), dtype=np.float) * self.x_U_int
        self.acst_L = np.array(self.cst_L)
        self.acst_U = np.array(self.cst_U)
        self.jd = np.array(self.jd).reshape((-1,1))
        if self.iteration == 0: self.jd_previous = self.jd
        # self.jd = (self.jd + self.jd_previous)/2 #stabilization #FIXME: tirei a estabilizacao

        # self.jac = np.concatenate([self.jac[0].reshape((-1,1)), self.jac[1].reshape((-1,1)), self.jac[2].reshape((-1,1))], axis=1)
        self.jac = np.concatenate([
            self.jac[0].reshape((-1,1)),
            self.jac[1].reshape((-1,1)),
            self.jac[2].reshape((-1,1)),
            self.jac[3].reshape((-1,1))
            ], axis=1)
        # self.jac = np.array(self.jac).reshape((-1,1))

    @staticmethod
    def cplex_optimize(prob, nvar, my_obj, my_constcoef, my_rlimits, my_ll, my_ul, minimize):
        prob.objective.set_sense(prob.objective.sense.minimize)

        my_ctype = "I"*nvar
        my_colnames = ["x"+str(item) for item in range(nvar)]
        # my_sense = ["L", "G"]
        if not minimize: my_sense = ["L", "L", "L", "L"]
        else: my_sense = ["G", "G", "L", "G", "L"]
        # else: my_sense = ["G", "L", "G", "L"]

        my_rownames = ["r1", "r2", "r3", "r4", "r5"]
        # my_rownames = ["r1", "r3", "r4", "r5"]
        # my_rlimits.pop(1)

        prob.variables.add(obj=my_obj, lb=my_ll, ub=my_ul, types=my_ctype,
                               names=my_colnames)

        rows = [cplex.SparsePair(ind=["x"+str(item) for item in range(nvar)], val = my_constcoef[0]),
                cplex.SparsePair(ind=["x"+str(item) for item in range(nvar)], val = my_constcoef[1]),
                cplex.SparsePair(ind=["x"+str(item) for item in range(nvar)], val = my_constcoef[1]),
                cplex.SparsePair(ind=["x"+str(item) for item in range(nvar)], val = my_constcoef[2]),
                cplex.SparsePair(ind=["x"+str(item) for item in range(nvar)], val = my_constcoef[3])]

        prob.linear_constraints.add(lin_expr=rows, senses=my_sense, rhs=my_rlimits, names=my_rownames)

    @staticmethod
    def post_evaluation(rho):
        print("call_back function should be defined")

    def solve(self, rho, minimize=True, filter_fun=None, call_back=post_evaluation):
        flip_limits = 0.0001525
        self.j_previous = None
        while True:
            self.control = np.copy(rho.vector())
            # set_working_tape(Tape())
            self.jd = self.dobj_fun(self.control)
            self.j = self.obj_fun(self.control)
            # if pesq > 3:
            # if self.j_previous is None: self.j_previous = self.j
            # if self.j/self.j_previous > 50:
            #     flip_limits /= 10
            self.cs = self.cst_fun(self.control)
            self.jac = self.jac_fun(self.control)
            self.reshape_to_matlab()
            self.volume_constraint = np.array([self.acst_L[0], self.acst_L[1], self.acst_U[1], self.acst_L[2]])
            # ep = 0.00001 # 12
            ep = 0.02 # 12
            self.epsilons = np.array([ep, ep, ep, ep]) #O benchmarking estava em 0.2
            '''if self.iteration >= 2 and self.j > self.j_previous:
                print("o j atual")
                print(self.j)
                print("")
                print("o j anterior")
                print(self.j_previous)
                self.epsilons = np.array([0.005, 0.01, 0.005]) #O benchmarking estava em 0.2
                flip_limits = 0
                import pdb; pdb.set_trace()'''
            # self.epsilons = np.array([0.005, 0.005, 0.005])
            # self.epsilons = np.array([0.001, 0.001, 0.001])

            # self.volume_constraint = np.array([0.5])
            # self.epsilons = np.array([0.2])
            ans = octave.tobs_from_matlab(
                    self.nvar,
                    self.x_L,
                    self.x_U,
                    self.cst_num,
                    self.acst_L,
                    self.acst_U,
                    self.j,
                    self.jd,
                    self.cs,
                    self.jac,
                    self.iteration,
                    self.epsilons,
                    self.control,
                    self.volume_constraint,
                    flip_limits
                    )
            PythonObjCoeff = ans[0][1] #because [0][0] is the design variable
            PythonConstCoeff = ans[0][2]
            PythonRelaxedLimits = ans[0][3]
            PythonLowerLimits = ans[0][4]
            PythonUpperLimits = ans[0][5]
            PythonnDesignVariables = ans[0][6]
            my_prob = cplex.Cplex()
            my_prob.parameters.mip.strategy.variableselect.set(2)
            coef = [item[0] for item in PythonObjCoeff.tolist()]
            constcoef = PythonConstCoeff.tolist()
            rlimits = [item[0] for item in PythonRelaxedLimits.tolist()]
            ll = [item[0] for item in PythonLowerLimits.tolist()]
            ul = [item[0] for item in PythonUpperLimits.tolist()]
            self.cplex_optimize(my_prob, self.nvar, coef, constcoef, rlimits, ll, ul, minimize)

            my_prob.solve()
            design_variables = my_prob.solution.get_values()

            rho.vector().add_local(np.array(design_variables))
            # rho.vector().set_local(np.array(design_variables)) # achava q era esse

            rho_notfiltered = rho.copy(deepcopy=True)
            '''if filter_fun is not None:
                rho = filter_fun(rho)# FIXME ta com filtro mas sem adjunto'''

            '''if self.iteration % 1000 == 0:
                mesh_adapted, domain= sm.generate_polygon(rho, accept_holes=False)'''

            # call_back(rho, rho_notfiltered, mesh_adapted, domain, self.iteration)
            call_back(rho, rho_notfiltered, None, None, self.iteration)

            vetor_rho = np.array([0 if item<0.5 else 1 for item in np.array(rho.vector())])
            rho.vector().set_local(vetor_rho)

            if self.iteration == 240:
                break
            elif self.iteration == 6:
                #q.assign(0.01)
                pass
            self.iteration += 1
            self.jd_previous = self.jd
            # if self.j_previous is not None and self.j_previous/self.j >= 10:
            self.j_previous = self.j

            # new_mesh_refined, domain = sm.generate_polygon_refined(control, mesh)
            # file_new_mesh_refined << new_mesh_refined
            # file_regions << domain

