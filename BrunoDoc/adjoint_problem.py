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
altura = 2.0

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
        (u, p) = split(w)
        # (u, p) = w.split()
        # (u2, p2) = w2.split()

        # Absoluto COM KVV
        '''self.funcional1 = 1*(  inner(self.alpha(rho) * u, u) \
            +  0.5 * AP.mu *(
                inner(grad(u) + grad(u).T, grad(u) + grad(u).T)
                # (u[1].dx(0)-u[0].dx(1))**2
                # inner(curl(u), curl(u))
                ) \
            )
        '''

        facet_marker = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        facet_marker.set_all(0)
        self.direita.mark(facet_marker,2)
        File("bru.pvd") << facet_marker
        dss = Measure("ds",domain=self.mesh,subdomain_data=facet_marker)
        ud = as_vector((102/2, 0))
        self.funcional1 = (
                inner(curl(u), curl(u))*dx
                # + inner((u - ud),(u - ud))*dss(2)
                )

        # self.funcional1 *= dx

        # return funcional1, funcional2, visc_term1, visc_term2, Fstar, w, w2
        return self.funcional1, w

    def get_adjoint_solution(self, rho, w):

        # w = self.get_forward_solution(rho)

        bc_hom = self.boundaries_cond()
        bc_hom[0].homogenize() # adjoint has homogeneous BCs #FIXME: REVER A BAGA

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

        # dJdu = derivative(self.funcional1, w) #FIXME
        facet_marker = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        facet_marker.set_all(0)
        self.direita.mark(facet_marker,2)
        dss = Measure("ds",domain=self.mesh,subdomain_data=facet_marker)
        ud = as_vector((102/2, 0))
        dJdu = (
                2. * inner(curl(u), curl(v_ad))*dx
                #+ 2. * inner(v_ad, u - ud) * dss(2)
                )
        '''
        dJdu = 1 *(
                # 2*self.alpha(rho) * inner(u, v_ad)* dx
                + 0.5*(
                    inner(grad(u), grad(v_ad)) * dx
                    + inner(grad(u).T, grad(v_ad)) * dx
                    + inner(grad(u), grad(v_ad).T) * dx
                    + inner(grad(u).T, grad(v_ad).T) * dx
                    )
                )'''
                # inner(grad(u), grad(v_ad)) * dx + 2*self.alpha(rho) * inner(u, v_ad)* dx
        solve( F_ad == dJdu , adj, bc_hom)

        # rho = self.density_filter(rho)

        adj_u, adj_p = split(adj)
        adj_u_f = adj_u

        dmo = TestFunction( rho.function_space() )

        # dJdm = (-1.*self.alphadash(rho)*inner(u,adj_u) + self.alphadash(rho)*inner(u,u))*dmo*dx
        dJdm = -1.*self.alphadash(rho)*inner(u,adj_u) *dmo*dx
        adjfinal_1_resp = assemble(dJdm)

        return adjfinal_1_resp

