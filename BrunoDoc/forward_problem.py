"""
This module creates the forward problem object
"""
from mshr import *
from dolfin import *
import BrunoDoc.properties as BProp
from BrunoDoc.read_param_file import *
from fenicstools import interpolate_nonmatching_mesh
import numpy as np

if linha[9][:2].upper() == 'DA': from dolfin_adjoint import *

delta = float(linha[2])
gap = float(linha[3])
radius = float(linha[4])
altura = 2.0

class FP(BProp.PP):
    """
    Creates the Forward Problem Object extendig the basic properties
    """
    anotar = True
    def __init__(self, hash=''):
        BProp.PP.__init__(self, hash)
        print('Creating the forward problem')
        self.veloc_file = File(self.workdir + '/velocA.pvd')
        self.press_file = File(self.workdir + '/pressA.pvd')
        self.veloc_file2 = File(self.workdir + '/velocB.pvd')
        self.press_file2 = File(self.workdir + '/pressB.pvd')
        self.filter_file = File(self.workdir + '/filter.pvd')
        self.veloc_plano_file = File(self.workdir + '/velocPlano.pvd')
        self.veloc_perp_file = File(self.workdir + '/velocPerp.pvd')
        self.veloc_planoVolta_file = File(self.workdir + '/velocPlanoVolta.pvd')
        self.veloc_perpVolta_file = File(self.workdir + '/velocPerpVolta.pvd')
        # self.w = self.get_forward_solution()
        return

    def get_forward_solution(self, rho, save_results):
        #Fe.set_log_level(30)

        # CRITICAL  = 50, // errors that may lead to data corruption and suchlike
        # ERROR     = 40, // things that go boom
        # WARNINg   = 30, // things that may go boom later
        # INFO      = 20, // information of general interest
        # PROGRESS  = 16, // what's happening (broadly)
        # TRACE     = 13, // what's happening (in detail)
        # DBG       = 10  // sundry

        #self.w = Function(self.W) #Isso ajuda a convergir durante a otimizacao por causa do erro(1e-7)
        BondConditions = self.boundaries_cond()

        if self.forward_problem == '2D' or self.forward_problem == '3D':

            (u, p) = split(self.w)
            (v, q) = TestFunctions(self.W)
            epsilon = sym(grad(u))
            F = (self.alpha(rho) * inner(u, v) * dx \
                + self.mu*inner(grad(u), grad(v)) * dx \
                - div(v)*p* dx  \
                - inner(div(u), q) * dx) \
                + inner(epsilon*u,v) * dx
            #Fe.solve(F == 0, self.w, BondConditions)
            Jacob = derivative(F, self.w)
            problem = NonlinearVariationalProblem(F, self.w, BondConditions, Jacob)
            solver = NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-7
        prm['newton_solver']['relative_tolerance'] = 1E-9
        prm['newton_solver']['maximum_iterations'] = 2000
        prm['newton_solver']['relaxation_parameter'] = 1.0
        for valor_mu in self.seq_mu:
            try:
                self.mu.assign(valor_mu)
                print("Resolvendo valor de mu {}".format(float(self.mu)))
                print("Resolvendo valor de kmax {}".format(float(self.alphabar)))
                if linha[9].upper() == 'DA' and valor_mu == self.seq_mu[-1]: set_working_tape(Tape()) ;print("LIMPOU A FITA")
                if self.anotar: solver.solve()
                else: solver.solve(annotate=False)
            except:
                print("ENTROU NO EXCEPT")
                print("Resolvendo valor de mu {}".format(float(self.mu)))
                print("Resolvendo valor de kmax {}".format(float(self.alphabar)))
                prm['newton_solver']['absolute_tolerance'] = 1E-6
                prm['newton_solver']['maximum_iterations'] = 500
                prm['newton_solver']['relaxation_parameter'] = 0.3
                if linha[9].upper() == 'DA' and valor_mu == self.seq_mu[-1]: set_working_tape(Tape()) ;print("LIMPOU A FITA")
                if self.anotar: solver.solve()
                else: solver.solve(annotate=False)
        '''

        (u, p) = TrialFunctions(self.W)
        F = self.alpha(rho) * inner(u, v) * dx \
                + self.mu*inner(grad(u)+grad(u).T, grad(v)) * dx \
                + inner(grad(p), v) * dx  \
                + inner(div(u), q) * dx
        problem = LinearVariationalProblem(lhs(F), rhs(F), self.w, BondConditions)
        solver = LinearVariationalSolver(problem)
        prm=solver.parameters
        prm['linear_solver'] = 'mumps'
        solver.solve()'''


        (u, p) = self.w.split()
        u.rename("velocidade", "conforme_tempo")
        p.rename("pressao", "conforme_tempo")
        if save_results:
            self.veloc_file << u
            self.press_file << p

        self.w_old = self.w
        return self.w

