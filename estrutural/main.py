import numpy as np
from dolfin import *
from mshr import *
from dolfin_adjoint import *

from .filter_class import filter_obj


# turn off redundant output in parallel
parameters["std_out_all_processes"] = False
#######-----------------------------------------------------------------#######
###############################################################################


def estrutural(rho_fenics):
    ###############################################################################
    #######------------------------INITIAL PARAMETERS-----------------------#######
    # nelx   = 50                             # Elements on length
    # nely   = 25                             # Elements on height
    Height = 1.5                            # Beam height        - mm
    Width  = 2.0                            # Beam width         - mm
    Eo     = 10.0#*1e3                           # Young's Modulus    - MPa
    nu     = 0.3                            # Poisson's ratio
    u0     = Constant((0.0, 0.0))           # Clamp condition
    b      = Constant((0.0, -1e-3))        # Force               - N
    g      = 9.810                          # Gravity acel        - m/s²
    q      = Constant(1)                    # power used in the SIMP
    eps    = Constant(1.0e-3)               # epsilon used in the SIMP
    rmin   = 1.2                            # radius of influecnce Proj method
    beta   = 1.0                            # Exp coeficient Proj method

    rho = rho_fenics.copy(deepcopy=True)
    mesh = Mesh(rho.function_space().mesh())
    A = VectorFunctionSpace(mesh, "CG", 1)
    B = FunctionSpace(mesh, "DG", 0)
    rho = Function(B)
    rho.vector()[:] = 1 - np.array(rho_fenics.vector())
    set_working_tape(Tape())

    def sigma(v):
        mu     = E(rho)/(2.0*(1.0+nu))               # Lamé parameter
        lmbda  = E(rho)*nu/((1.0+nu)*(1.0-2.0*nu))   # Lamé parameter
        I = Identity(v.geometric_dimension())
        return (2*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*I)

    class LeftEdge(SubDomain):
          def inside(self, x, on_boundary):
                return abs(x[0] - 2.0) < DOLFIN_EPS #and x[1] >= 0.5
                # return on_boundary and abs(x[0] - 2.0) < DOLFIN_EPS #and x[1] >= 0.5

    class RightEdge(SubDomain):
          def inside(self, x, on_boundary):
                return abs(x[0])  < DOLFIN_EPS #and x[1]>= 0.5
                # return on_boundary and abs(x[0])  < DOLFIN_EPS #and x[1]>= 0.5

    left_edge = LeftEdge()
    right_edge = RightEdge()

    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(0)
    right_edge.mark(sub_domains, 1)
    left_edge.mark(sub_domains, 2)
    File("apagar.pvd") << sub_domains

    ds = Measure("ds")[sub_domains]

    #plot(sub_domains)

    bc = DirichletBC(A, u0, left_edge)
    bc_adj = DirichletBC(A, u0, left_edge)

    ###############################################################################
    #######-----------------------SIMP METHOD-------------------------------#######
    # Define Solid Isotropic Material with Penalisation SIMP rule
    def E(rho):
        """Solid isotropic material with penalisation (SIMP)."""
        return eps + (1 - eps) * Eo*(rho**q)
    def E_dash(rho):
        """Solid isotropic material with penalisation (SIMP)."""
        dot_materialModel = ( q * (1 - eps) * Eo * rho **(q-1))
        return dot_materialModel
    #######-----------------------------------------------------------------#######
    ###############################################################################


    ###############################################################################
    #######---------------------FORWARD PDE PROBLEM-------------------------#######
    def forward (rho, annotate=True):
        """Solve the forward problem for a given material distribution c(x)."""
        u = TrialFunction(A)
        w = TestFunction(A)
        F = inner(sigma(u), grad(w))*dx - dot(b,w)*ds(1)
        a = lhs(F)
        L = rhs(F)
        u = Function(A)
        solve(a == L, u, bc)# , annotate=annotate)
        return u, F

    def solve_adjoint(rho, u):
        """
        Function that gives the adjoint solution in order to compute later
        the gradient for the optimization
        """
        bc_adj.homogenize() #adjoint has homogeneous BCs. It means(DirichletBC(U, Constant(0), 'on_boundary')
        adj = Function(A)
        adj_tst = TestFunction(A)
        #Let's start with the main terms of the adjoint equation
        adjEquation = E(rho)*inner(sigma(adj), grad(adj_tst))*dx - \
                        inner(b, adj_tst)*ds(1)
        a = lhs(adjEquation)
        L = rhs(adjEquation)
        solve(a == L, adj, bc_adj) #, annotate=False)
        #Now is the point to implement the filtering terms
        return adj
    #######-----------------------------------------------------------------#######
    ###############################################################################

    u, F = forward(rho)
    File("apagar_u.pvd") << u
    J = assemble(dot(b,u)*ds(1))
    m = Control(rho)

    # filter_fun = filter_obj(mesh, rmin=rmin, beta=beta)
    # ds_vars_filtered = self.filter_obj.Rho_elem(ds_vars)

    # lam = solve_adjoint(rho, u)
    '''adF = adjoint(derivative(F, u))
    dJ = derivative(J, u, TestFunction(w.function_space()))
    solve(action(adF, adj) - dJ == 0, adj, bc)

    dmo = TestFunction( rho.function_space())
    L = - adj * dmo * dx
    dfval = assemble(L)'''
    Jhat = ReducedFunctional(J, m)
    dfval = Jhat.derivative()
    dfval = - np.array(dfval.vector())

    return J, dfval

