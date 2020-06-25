"""
This module creates the adjoint object with ics as Fe
"""
# import dolfin as Fe
from dolfin import *
import BrunoDoc.properties as BProp
import BrunoDoc.forward_problem as FProb
from fenicstools import interpolate_nonmatching_mesh
from BrunoDoc.read_param_file import *
from BrunoDoc.filter_class import filter_obj

if linha[9][:2].upper() == 'DA': from dolfin_adjoint import *

delta = float(linha[2])
gap = float(linha[3])
radius = float(linha[4])
altura = 1.5

Reynolds = 300
v_average = Reynolds / (2*gap)
WF = 0.1
class AP(FProb.FP):
    minimize = True #False
    def __init__(self, hash=''):
        FProb.FP.__init__(self, hash)
        print("Creating the Adjoint Problem")

    def Funcional(self, rho, save_results=True): #, w, w2):
        mesh = rho.function_space().mesh()
        # self.filter_f = filter_obj(mesh, rmin=0.05, beta=0)
        # rho = self.filter_f.Rho_elem(rho)
        # rho.rename("ControlFiltered", "ControlFiltered")
        # self.file_filtrado << rho


        w = self.get_forward_solution(rho, save_results)
        w2 = self.get_forward_solution2(rho, save_results)
        (u, p) = split(w)
        (u2, p2) = split(w)
        # (u, p) = w.split()
        # (u2, p2) = w2.split()

        # Absoluto COM KVV
        funcional1 = 1*(  inner(self.alpha(rho) * u, u) \
            +  0.5 * AP.mu *(
                inner(grad(u) + grad(u).T, grad(u) + grad(u).T)
                ) \
            # - inner(self.alpha(rho), (self.r_n-self.radius)**4) \
            )
        funcional2 = 1* (  inner(self.alpha(rho) * u2, u2) \
            +  0.5 * AP.mu * (
                inner(grad(u2) + grad(u2).T, grad(u2) + grad(u2).T)
                )\
            )

        Fstar = WF/(2*gap)**2 * self.alpha(rho) * inner(u2,u2)/(self.alphabar *v_average)

        visc_term1 = 0.5 * AP.mu * inner(grad(u), grad(u))* dx
        visc_term2 = 0.5 * AP.mu * inner(grad(u2), grad(u2))* dx

        funcional1 *= dx
        funcional2 *= dx
        Fstar *= dx

        return funcional1, funcional2, visc_term1, visc_term2, Fstar, w, w2

    def get_adjoint_solution(self, rho, w):

        # w = self.get_forward_solution(rho)

        bc_hom = self.boundaries_cond()
        # self.bc_hom[0].homogenize() # adjoint has homogeneous BCs #FIXME: REVER A BAGA
        bc_hom[0].homogenize() # adjoint has homogeneous BCs #FIXME: REVER A BAGA
        bc_hom[1].homogenize() # adjoint has homogeneous BCs #FIXME: REVER A BAGA
        bc_hom[2].homogenize() # adjoint has homogeneous BCs #FIXME: REVER A BAGA

        adj = Function(self.W)
        adj_t = TrialFunction(self.W)
        (u_ad_t, p_ad_t) = split(adj_t)
        adj_tst = TestFunction(self.W)
        (v_ad, q_ad) = split(adj_tst)
        (u, p) = split(w)

        F_ad = (
                inner(div(u_ad_t), q_ad) \
                + inner(grad(u_ad_t), grad(v_ad)) \
                - inner(grad(u_ad_t)*u, v_ad) \
                + inner(grad(u).T* u_ad_t, v_ad) \
                + self.alpha(rho) * inner(u_ad_t, v_ad) \
                + inner(grad(p_ad_t), v_ad) \
                )*dx

        '''dJdu = derivative(self.funcional1, w) #FIXME
        '''
        dJdu = 1 *(
                2*self.alpha(rho) * inner(u, v_ad)* dx
                + 0.5*(
                    inner(grad(u), grad(v_ad)) * dx
                    + inner(grad(u).T, grad(v_ad)) * dx
                    + inner(grad(u), grad(v_ad).T) * dx
                    + inner(grad(u).T, grad(v_ad).T) * dx
                    )
                )
                # inner(grad(u), grad(v_ad)) * dx + 2*self.alpha(rho) * inner(u, v_ad)* dx
        solve( F_ad == dJdu , adj, bc_hom)

        # rho = self.density_filter(rho)

        adj_u, adj_p = split(adj)
        adj_u_f = adj_u

        dmo = TestFunction( rho.function_space() )

        dJdm = (-1.*self.alphadash(rho)*inner(u,adj_u) + self.alphadash(rho)*inner(u,u))*dmo*dx
        adjfinal_1_resp = assemble(dJdm)

        return adjfinal_1_resp

    def get_adjoint_solution2(self, rho, w2):

        # w2 = self.get_forward_solution2(rho)

        bc_hom2 = self.boundaries_cond2()
        bc_hom2[0].homogenize() # adjoint has homogeneous BCs
        bc_hom2[1].homogenize() # adjoint has homogeneous BCs
        bc_hom2[2].homogenize() # adjoint has homogeneous BCs
        # self.bc_hom2[1].homogenize() # adjoint has homogeneous BCs
        adj2 = Function(self.W2)
        adj_t2 = TrialFunction(self.W2)
        (u_ad_t2, p_ad_t2) = split(adj_t2)
        adj_tst2 = TestFunction(self.W2)
        (v_ad2, q_ad2) = split(adj_tst2)
        (u2, p2) = split(w2)

        F_ad2 = (
                inner(div(u_ad_t2), q_ad2) \
                + inner(grad(u_ad_t2), grad(v_ad2)) \
                - inner(grad(u_ad_t2)*u2, v_ad2) \
                + inner(grad(u2).T* u_ad_t2, v_ad2) \
                + self.alpha(rho) * inner(u_ad_t2, v_ad2) \
                + inner(grad(p_ad_t2), v_ad2) \
                )*dx
        dJdu2 = (
                2*self.alpha(rho) * inner(u2, v_ad2)* dx
                + 0.5*(
                    inner(grad(u2), grad(v_ad2)) * dx
                    + inner(grad(u2).T, grad(v_ad2)) * dx
                    + inner(grad(u2), grad(v_ad2).T) * dx
                    + inner(grad(u2).T, grad(v_ad2).T) * dx
                    )
                )
                # inner(grad(u2), grad(v_ad2)) * dx + 2*self.alpha(rho) * inner(u2, v_ad2)* dx
        solve( F_ad2 == dJdu2 , adj2, bc_hom2)

        # rho = self.density_filter(rho)

        adj_u2, adj_p2 = split(adj2)
        adj_u_r = adj_u2

        dmo = TestFunction( rho.function_space() )

        # dJdm2 = (-1.*self.alphaJdash(rho)*inner(u2,adj_u2) + self.alphaJdash(rho)*inner(u2,u2))*dmo*dx
        dJdm2 = (-1.*self.alphadash(rho)*inner(u2,adj_u2) + self.alphadash(rho)*inner(u2,u2))*dmo*dx
        adjfinal_2_resp = assemble(dJdm2)

        return adjfinal_2_resp


    def get_adjoint_solution3(self, rho, w2):

        # w2 = self.get_forward_solution2(rho, save_results=False)

        bc_hom2 = self.boundaries_cond2()
        bc_hom2[0].homogenize() # adjoint has homogeneous BCs
        bc_hom2[1].homogenize() # adjoint has homogeneous BCs
        bc_hom2[2].homogenize() # adjoint has homogeneous BCs
        # self.bc_hom2[1].homogenize() # adjoint has homogeneous BCs
        adj2 = Function(self.W2)
        adj_t2 = TrialFunction(self.W2)
        (u_ad_t2, p_ad_t2) = split(adj_t2)
        adj_tst2 = TestFunction(self.W2)
        (v_ad2, q_ad2) = split(adj_tst2)
        (u2, p2) = split(w2)

        F_ad2 = (
                inner(div(u_ad_t2), q_ad2) \
                + inner(grad(u_ad_t2), grad(v_ad2)) \
                - inner(grad(u_ad_t2)*u2, v_ad2) \
                + inner(grad(u2).T* u_ad_t2, v_ad2) \
                + self.alpha(rho) * inner(u_ad_t2, v_ad2) \
                + inner(grad(p_ad_t2), v_ad2) \
                )*dx
        dJdu3 = (
                2 * WF/(2*gap)**2 * self.alpha(rho) * inner(u2,v_ad2)/(self.alphabar *v_average)*dx
                )
        solve( F_ad2 == dJdu3 , adj2, bc_hom2)

        # rho = self.density_filter(rho)

        adj_u2, adj_p2 = split(adj2)

        dmo = TestFunction( rho.function_space() )

        adjfinal_2_resp = adj_u2
        dJdm3 = (-1.*self.alphadash(rho)*inner(u2,adj_u2) + WF/(2*gap)**2 * self.alphadash(rho) * inner(u2,u2)/(self.alphabar *v_average))*dmo*dx
        adjfinal_3_resp = assemble(dJdm3)

        return adjfinal_3_resp


