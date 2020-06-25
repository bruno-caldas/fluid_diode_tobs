"""
This module creates the Optimization Object
"""
import BrunoDoc.adjoint_problem as AProb
import BrunoDoc.parameters as Par
import numpy
from dolfin import *
from dolfin_adjoint import *

class OP(AProb.AP):
    """
    This Class supplies a python object that has all Ipopt needs,
    such as ObjFun, Gradientes and Constraints
    """
    iterations = 5
    subiterations = 5
    opt_mode = "DA"
    def __init__(self, rho_initial=None, hash=''):
        AProb.AP.__init__(self, hash)
        self.rho_initial = rho_initial
        self.file_out = File(self.workdir + "/control_da.pvd")

    def add_volf_constraint(self, upp, lwr):
        self.V1_up = upp
        self.V1_lwr = lwr

    def add_volf_constraint2(self, upp, lwr):
        self.V2_up = upp
        self.V2_lwr = lwr

    def add_volf_constraint3(self, upp, lwr):
        self.V3_up = upp
        self.V3_lwr = lwr

    def sim(self, max_iter=500):
        super().sim()
        if self.rho_initial != None:
            self.rho = Function(self.A, self.rho_initial.vector(), name="Control")
        self.w = self.get_forward_solution(self.rho)
        (u, p) = split(self.w)
        (v, q) = TestFunctions(self.W)
        # self.w2 = self.get_forward_solution2(self.rho)
        # (u2, p2) = split(self.w2)
        # (v2, q2) = TestFunctions(self.W2)
        self.J, self.J2, visc_term = self.Funcional(self.rho, self.w, self.w2)
        self.J = assemble(self.J)
        # self.J = assemble(self.J) / (assemble(self.J2) + 1e-9) + assemble(self.termo_add)
        self.m = Control(self.rho)
        self.rho_viz = Function(self.A, name="ControlVisualisation")
        self.seq_mu = [self.seq_mu[-1]] # Matar a continuidade para posteriores simulacoes

        def eval_cb(j, rho):
            self.rho_viz.assign(rho)
            self.file_out << self.rho_viz
            self.anotar = False
            self.w = self.get_forward_solution(rho)
            self.J, self.J2, visc_term = self.Funcional(rho, self.w, self.w2)
            self.J = assemble(self.J)
            # self.J = assemble(self.J) / (assemble(self.J2) + 1e-9) + assemble(self.termo_add)

            '''vetor1 = interpolate(Constant(0), self.A)
            vetor2 = interpolate(Constant(1), self.A)
            conv_rate = taylor_test(self.Jhat, vetor1, vetor2)'''

            self.file_obj = open(self.workdir + "/fun_obj.txt", "a+")
            self.file_obj.write(str(self.J) + "\n")
            self.file_obj.close()

        self.Jhat = ReducedFunctional(self.J, self.m, eval_cb_post=eval_cb)
        self.lb = 0.0
        self.ub = 1.0
        self.v_cstr1up = UFLInequalityConstraint((self.V1_up - self.rho)*self.dxx(1), self.m)
        self.v_cstr1lwr = UFLInequalityConstraint((- self.V1_lwr + self.rho)*self.dxx(1), self.m)
        self.v_cstr2up = UFLInequalityConstraint((self.V2_up - self.rho)*self.dxx(2), self.m)
        self.v_cstr2lwr = UFLInequalityConstraint((- self.V2_lwr + self.rho)*self.dxx(2), self.m)
        self.v_cstr3up = UFLInequalityConstraint((self.V3_up - self.rho)*self.dxx(3), self.m)
        self.v_cstr3lwr = UFLInequalityConstraint((-self.V3_lwr + self.rho)*self.dxx(3), self.m)
        self.problem = MinimizationProblem(self.Jhat, bounds=(self.lb, self.ub), constraints=[
            self.v_cstr1up,
            self.v_cstr1lwr,
            self.v_cstr2up,
            self.v_cstr2lwr,
            self.v_cstr3up,
            self.v_cstr3lwr,
            ])
        self.parameters = {'maximum_iterations': max_iter}
        self.solver = IPOPTSolver(self.problem, parameters=self.parameters)

        rho_opt = self.solver.solve()
        rho_opt_xdmf = XDMFFile(self.workdir + "/control_solution_guess.xdmf")
        rho_opt_xdmf.write(rho_opt)
        self.rho.assign(rho_opt)

        # set_working_tape(Tape())
        # rho_intrm = XDMFFile(self.workdir + "intermediate-guess-%s.xdmf" )
        # rho_intrm.write(self.rho)
        # self.w = self.get_forward_solution(self.rho)
        # (u, p) = split(self.w)
        # (v, q) = TestFunctions(self.W)
        # self.w2 = self.get_forward_solution2(self.rho)
        # (u2, p2) = split(self.w2)
        # (v2, q2) = TestFunctions(self.W2)
        # self.J2 = assemble(\
        #         (0.5 * inner(self.alpha(self.rho) * u, u) * dx + 0.5 * self.mu * inner(grad(u)+grad(u).T, grad(u)) * dx)
        #         ) / assemble(\
        #         (0.5 * inner(self.alpha(self.rho) * u2, u2) * dx + 0.5 * self.mu * inner(grad(u2)+grad(u2).T, grad(u2)) * dx)\
        #         )
        # self.m = Control(self.rho)
        # self.allctrls = File(self.workdir + "/final.pvd")
        # self.rho_viz = Function(self.A, name="ControlVisualisation")
        # def eval_cb2(j, rho):
        #     self.rho_viz.assign(rho)
        #     self.allctrls << self.rho_viz
        # self.Jhat = ReducedFunctional(self.J2, self.m, eval_cb_post=eval_cb2)
        # self.problem2 = MinimizationProblem(self.Jhat, bounds=(self.lb, self.ub), constraints=[self.v_cstr1, self.v_cstr2, self.v_cstr3])
        # self.parameters2 = {'maximum_iterations': 200}
        # self.solver2 = IPOPTSolver(self.problem2, parameters=self.parameters2)
        # rho_opt = self.solver2.solve()


