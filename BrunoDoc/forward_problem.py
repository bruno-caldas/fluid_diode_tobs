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
altura = 1.5

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

    def density_filter(self, rho_nf):
        mtrial = TrialFunction(self.A )
        mtresp = Function(self.A )
        m_test = TestFunction(self.A )
        n = FacetNormal(self.mesh)
        ds = Measure('ds', domain = self.mesh)
        eq_l = (
                (self.r_min**2)*inner(grad(mtrial), grad(m_test)) *dx# * 2 * pi * self.r_n * dx
                + mtrial*m_test*dx - (rho_nf)*m_test *dx# * 2 * pi * self.r_n * dx
                # -self.r_min**2*inner(n , grad(mtrial))*m_test*ds
                )
        solve(lhs(eq_l)== rhs(eq_l), mtresp )

        # mtresp_rounded = np.clip(np.array(mtresp.vector()).round(0), a_min = 0, a_max = 1)
        # rho_nf.vector().set_local(mtresp_rounded)

        # rho_nf.rename("controlNotFiltered", "controlNotFiltered")
        # self.file_out << rho_nf
        vetor_rho = np.array(mtresp.vector())

        # vetor_rho = np.array([0 if item<0.5 else 1 for item in vetor_rho])
        vetor_rho = np.array([item if item<0.5 else item for item in vetor_rho])
        rho_nf.vector().set_local(vetor_rho)

        # Set up a vertex_2_dof list
        dofmap = self.A.dofmap()
        nvertices = self.mesh.ufl_cell().num_vertices()
        indices = [dofmap.tabulate_entity_dofs(0, i)[0] for i in range(nvertices)]
        vertex_2_dof = dict()
        [vertex_2_dof.update(dict(vd for vd in zip(cell.entities(0),
                                                dofmap.cell_dofs(cell.index())[indices])))
                                for cell in cells(self.mesh)]
        '''index_to_change = np.where(
                (self.mesh.coordinates()[:,0]<gap) |
                (self.mesh.coordinates()[:,1]>radius+altura-gap) |
                (self.mesh.coordinates()[:,0]>delta-gap)
                )[0]
        for i in index_to_change:
            dof_idx = vertex_2_dof[i]
            rho_nf.vector()[dof_idx] = 1'''

        rho_nf.rename("Controlfiltered", "Controlfiltered")
        self.filter_file << rho_nf
        # import pdb; pdb.set_trace()
        return rho_nf


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

        self.w2_old = self.w2
        (u, p) = self.w.split()
        u.rename("velocidade", "conforme_tempo")
        p.rename("pressao", "conforme_tempo")
        if save_results:
            self.veloc_file << u
            self.press_file << p

        self.w_old = self.w
        return self.w

    def get_forward_solution2(self, rho, save_results=True):
        BondConditions2 = self.boundaries_cond2()

        if self.forward_problem == '2D':
            (u2, p2) = split(self.w2)
            (v2, q2) = TestFunctions(self.W2)
            epsilon2 = sym(grad(u2))
            F2 = (self.alpha(rho) * inner(u2, v2) * dx \
                + self.mu*inner(grad(u2), grad(v2)) * dx \
                - div(v2)*p2* dx  \
                - inner(div(u2), q2) * dx) \
                + inner(epsilon2*u2,v2) * dx
            #Fe.solve(F2 == 0, self.w2, BondConditions2)
            Jacob2 = derivative(F2, self.w2)
            problem2 = NonlinearVariationalProblem(F2, self.w2, BondConditions2, Jacob2)
            solver2 = NonlinearVariationalSolver(problem2)

        prm2 = solver2.parameters
        prm2['newton_solver']['absolute_tolerance'] = 1E-7
        prm2['newton_solver']['relative_tolerance'] = 1E-9
        prm2['newton_solver']['maximum_iterations'] = 2000
        prm2['newton_solver']['relaxation_parameter'] = 1.0
        for valor_mu in self.seq_mu:
            try:
                self.mu.assign(valor_mu)
                print("VOLTA - Resolvendo valor de mu {}".format(float(self.mu)))
                print("VOLTA - Resolvendo valor de kmax {}".format(float(self.alphabar)))
                if linha[9].upper() == 'DA' and valor_mu == self.seq_mu[-1]: set_working_tape(Tape()) ;print("LIMPOU A FITA")
                if self.anotar: solver2.solve()
                else: solver2.solve(annotate=False)
            except:
                print("ENTROU NO SEGUNDO EXCEPT")
                prm2['newton_solver']['maximum_iterations'] = 500
                prm2['newton_solver']['relaxation_parameter'] = 0.3
                prm2['newton_solver']['absolute_tolerance'] = 1E-6
                if linha[9].upper() == 'DA' and valor_mu == self.seq_mu[-1]: set_working_tape(Tape()) ;print("LIMPOU A FITA")
                if self.anotar: solver2.solve()
                else: solver2.solve(annotate=False)

        (u2, p2) = self.w2.split()
        u2.rename("velocidade", "conforme_tempo")
        p2.rename("pressao", "conforme_tempo")
        if save_results:
            self.veloc_file2 << u2
            self.press_file2 << p2

        self.w2_old = self.w2
        return self.w2
