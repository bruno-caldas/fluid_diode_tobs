"""
Class that inputs the properties
"""
import BrunoDoc.newdir as Nw
from dolfin import *
from mshr import *
from mpi4py import MPI as paralellism
from BrunoDoc.read_param_file import *

# import malha.malha as ma
import malha.malha_quad as ma

if linha[9][:2].upper() == 'DA': from dolfin_adjoint import *

delta = float(linha[2])
gap = float(linha[3])
radius = float(linha[4])

class esquerda_def(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0]<-0.999
class direita_def(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0]>delta+0.999

class InflowOutflow(UserExpression):
    """
    Boundary Conditions Class. It affects forward_problem and adjoint_problem
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Reynolds = 300
        v_average = Reynolds / (2*gap)
        self.gbar = (3/2) * v_average
        self.length = gap

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0.0
        values[1] = 0.0
        t = x[1]
        if near(x[0], -1.0) or near(x[0], delta + 1.0) or near(x[1], 0):
            values[0] = abs(self.gbar*( (1 - t/self.length)*(1+t/self.length) ) )#Entrada do Canal
    def value_shape(self):
        return (2,)

class InflowOutflow2(InflowOutflow):
    """
    Boundary Conditions Class. It affects forward_problem and adjoint_problem
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0.0
        values[1] = 0.0
        t = x[1]
        if near(x[0], -1.0) or near(x[0], delta + 1.0) or near(x[1], 0):
            values[0] = - abs(self.gbar*( (1 - t/self.length)*(1 + t/self.length) ) )#Entrada do Canal

class Distribution(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval_cell(self, values, x, ufc_cell):
        if x[1] > 1.5 + radius - 1.5*(1-0.75): #and x[1] < 1.5 + radius - 1.5*(1-0.9):
            values[0] = 0.0
        else:
            values[0] = 1.0

class PP:
    """
    Properties that can be changed
    """
    #rho_eq = 2.696373e-6 #Exemplo do Breda
    rho_eq = 1.
    mu = Constant(1.0)
    # seq_mu = [10, 5, 3, 2, 1.5, 1.3, 1.2]#, 1.15, 1.1, float(mu)]
    seq_mu = [float(mu)]
    alphaunderbar = 0.
    alphabar = 2.5 * mu * 1e6 * 1e-6    # kg/ (m**3 *s)
    alphaJbar = 2.5 * mu * 1e6 * 1e-6    # kg/ (m**3 *s)
    q = Constant(1.0e5) # q value that controls difficulty/discrete-valuedness of solution
    # q = Constant(0.001) # aqui da ruim (quadriculado)
    r_min = 0.05

    delta = delta # 2 - 2*0.25
    altura = 1.5# 0.35
    N = 30# 40
    # radius = 30 # Exemplo do Breda
    radius = radius
    config = 0
    diodicidade = False

    def __init__(self, hash=''):
        # self.workdir = Nw.erase_old_results('results', hash)
        self.comm = paralellism.COMM_WORLD
        self.rank = self.comm.Get_rank()
        if self.rank == 0:
            self.workdir = Nw.erase_old_results('results', hash)
        else:
            self.workdir = None
        self.workdir = self.comm.bcast(self.workdir, root=0)

    def sim(self):
        self.mesh = self.mesh_fun(mesh_file=None)
        self.subdominios()
        self.functionsp()
        self.functionsp2()
        self.rho = interpolate(Distribution(), self.A)
        File("apagar.pvd") << self.rho
        if linha[9].upper() == 'DA': self.rho.rename("ControlVisualisation", "")
        else: self.rho.rename("control", "")
        self.file_domain << self.rho

    def alpha(self, rho):
        """Inverse permeability as a function of rho"""
        model = self.alphabar + (self.alphaunderbar - self.alphabar) *\
            rho * (1 + self.q) / (rho + self.q)
        return model

    def alphadash(self, rho):
        """Derivative of alpha(rho) in respect to rho"""
        model = (self.alphaunderbar - self.alphabar) * (1 * (1 + self.q) / (rho + self.q)\
            - rho * (1 + self.q)/((rho + self.q)*(rho + self.q)))
        model = (self.alphaunderbar - self.alphabar) * (
                    (1+self.q)/(rho+self.q) - rho*(1+self.q)/(rho+self.q)**2
                    )
        return model

    def mesh_fun(self, mesh_file=None):
        """Defines the mesh"""
        self.esquerda = esquerda_def()
        self.direita = direita_def()
        delta = self.delta
        if mesh_file is None and (self.forward_problem == '2D' or self.forward_problem == '2DSwirl'):
            if self.config == 1:
                vertices = Polygon([Point(-1.0, 0. + self.radius),
                            Point(delta+1.0, 0. + self.radius),
                            Point(delta+1.0, gap + self.radius),
                            Point(delta, gap + self.radius),
                            Point(delta, self.altura + self.radius),
                            Point(0., self.altura + self.radius),
                            Point(0., gap + self.radius),
                            Point(-1.0 , gap + self.radius)])
            else:
                raise Exception('Defina uma configuracao')

            if linha[9].upper()[:2] == 'DA': mesh = Mesh(generate_mesh(vertices, self.N))
            # else: mesh = generate_mesh(vertices, self.N)
            # else: mesh = ma.valve(delta, self.N)
            else: mesh = ma.generate_quad_mesh(int(self.N/2))
            (self.z_n, self.r_n) = SpatialCoordinate(mesh)
            self.nr = 1
            self.nz = 0
            self.full_geo = vertices
        elif self.forward_problem == '3D':
            cilindro1 = Cylinder(Point(-0.1, 0, 0), Point(1.1, 0, 0), 0.1, 0.1)
            cilindro2 = Cylinder(Point(0, 0, 0), Point(1.0, 0, 0), 0.35, 0.35)
            dominio = cilindro1 +cilindro2
            generator = CSGCGALMeshGenerator3D()
            generator.parameters["edge_size"] = 0.025 *100
            generator.parameters["facet_angle"] = 25.0 *100
            generator.parameters["facet_size"] = 0.05 *100
            generator.parameters["cell_size"] = 0.01 * 100
            # generator.parameters["cell_radius_edge_ratio"] = 0.01 * 100


            # Invoke the mesh generator
            mesh = generator.generate(CSGCGALDomain3D(dominio))
            # import matplotlib.pyplot as plt
            # plt.show()
        else:
            #It should be implemented the read of a mesh file
            raise NotImplemented()

        return mesh

    def subdominios(self):
        self.sub_domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        self.dxx = Measure('dx', domain=self.mesh, subdomain_data=self.sub_domains)
        self.sub_domains.set_all(2)
        CompiledSubDomain("x[0]>"+str(0)+" && "+"x[0]<"+str(self.delta) ).mark(self.sub_domains, 2)

        '''CompiledSubDomain("x[0]<="+str(self.delta)+ "&& ( \
                x[1] < " +str(self.radius + gap) + "|| \
                x[0] < 0 \
                )").mark(self.sub_domains, 1)'''
        # CompiledSubDomain("x[0]<="+str(0 + gap*3) +" || x[1] >="+str(self.altura+self.radius - gap)).mark(self.sub_domains,1)
        CompiledSubDomain("x[0]<="+str(0 + gap*0) +" || x[1] >="+str(self.altura+self.radius)).mark(self.sub_domains,1)

        CompiledSubDomain("x[0]>="+str(self.delta - gap*0) + " || x[1]>"+str(10*self.altura+self.radius-self.delta/2) ).mark(self.sub_domains, 3)
        File("subdominios.pvd") << self.sub_domains


    def functionsp(self):
        self.A = FunctionSpace(self.mesh, "DG", 0)
        if self.forward_problem == '2D':
            self.U_h = VectorElement("CG", self.mesh.ufl_cell(), 2, dim=2)
        elif self.forward_problem == '2DSwirl':
            self.U_h = VectorElement("CG", self.mesh.ufl_cell(), 2, dim=3)
        elif self.forward_problem == '3D':
            self.U_h = VectorElement("CG", self.mesh.ufl_cell(), 2, dim=3)

        self.P_h = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.W = FunctionSpace(self.mesh, self.U_h*self.P_h)
        self.U_plano = VectorFunctionSpace(self.mesh, "CG", 2, dim=2)
        self.U_perp = VectorFunctionSpace(self.mesh, "CG", 2, dim=3)
        self.U = VectorFunctionSpace(self.mesh, "CG", 2, dim=2)
        # self.u0 = Function(self.U)
        self.w = Function(self.W)
        self.wt = TrialFunction(self.W)

        return

    def functionsp2(self):
        if self.forward_problem == '2D':
            self.U_h2 = VectorElement("CG", self.mesh.ufl_cell(), 2, dim=2)
        elif self.forward_problem == '2DSwirl':
            self.U_h2 = VectorElement("CG", self.mesh.ufl_cell(), 2, dim=3)
        elif self.forward_problem == '3D':
            self.U_h2 = VectorElement("CG", self.mesh.ufl_cell(), 2, dim=3)

        self.P_h2 = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.W2 = FunctionSpace(self.mesh, self.U_h2*self.P_h2)
        self.U2_plano = VectorFunctionSpace(self.mesh, "CG", 2, dim=2)
        self.U2_perp = VectorFunctionSpace(self.mesh, "CG", 2, dim=3)
        self.U2 = VectorFunctionSpace(self.mesh, "CG", 2, dim=2)
        self.u02 = Function(self.U2)
        self.w2 = Function(self.W2)
        self.wt2 = TrialFunction(self.W2)
        return

    def boundaries_cond(self):
        """
        In the Fenics 2018 (Python 3) version, we must use self.G
        and not local definition
        """
        if self.forward_problem == '2D': self.G = InflowOutflow(degree=2)

        boundary = [
                DirichletBC(self.W.sub(0).sub(1), Constant(0), "x[1]<="+str(self.radius+1e-10) +" && !(x[0] <= "+str(-.99)+")"),
                DirichletBC(self.W.sub(0), self.G, "on_boundary && !(x[1]<="+str(self.radius+1e-10)+ "&& x[0]>="+str(-.9) +")"),
                DirichletBC(self.W.sub(1), 0.0, "on_boundary && x[0] > "+str(self.delta+0.99)),
                ]

        return boundary

    def boundaries_cond2(self):
        """
        In the Fenics 2018 (Python 3) version, we must use self.G
        and not local definition
        """
        if self.forward_problem == '2D': self.G2 = InflowOutflow2(degree=2)

        boundary2 = [
                DirichletBC(self.W2.sub(0).sub(1), Constant(0), "x[1]<="+str(self.radius+1e-10) +" && !(x[0] >= "+str(self.delta +.99)+")"),
                DirichletBC(self.W2.sub(0), self.G2, "on_boundary && !(x[1]<="+str(self.radius+1e-10) + "&& x[0]<="+str(self.delta+.9) +")"),
                DirichletBC(self.W2.sub(1), 0.0, "on_boundary && x[0] < "+str(-0.99)),
                ]

        return boundary2

