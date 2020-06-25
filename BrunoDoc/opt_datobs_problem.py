"""
This module creates the Optimization Object
"""
from dolfin import *
from dolfin_adjoint import *
import BrunoDoc.adjoint_problem as AProb
import BrunoDoc.parameters as Par
import numpy as np
import BrunoDoc.tobs as tobs


class OP(AProb.AP):
    """
    This Class supplies a python object that has all Ipopt needs,
    such as ObjFun, Gradientes and Constraints
    """
    iterations = 5
    subiterations = 5
    first_iter = True
    first_iter_deriv = True
    opt_mode = "CONT"
    def __init__(self, rho_initial=None, hash=''):
        AProb.AP.__init__(self, hash)
        self.objfun_rf = None
        self.iter_fobj = 0
        self.iter_dobj = 0
        self.cst_U = []
        self.cst_L = []
        self.cst_num = 0
        self.file_out = File(self.workdir + "/control.pvd")
        self.state_file = File(self.workdir + "/state.pvd")
        self.state_file2 = File(self.workdir + "/state2.pvd")
        self.file_mesh_adapted = File(self.workdir + "/mesh_adapt.pvd")
        self.rho_initial = rho_initial

    def __vf_fun_var_assem__(self):
        self.rho_tst   = TestFunction(self.rho.function_space())
        self.vol_xi  = assemble(self.rho_tst * Constant(1) * dx)
        self.vol_sum = self.vol_xi.sum()
        self.malhavol_vol_xi  = assemble(self.rho_tst * Constant(1) * self.dxx(1))#x(1))#/(base*altura)
        self.malhavol_vol_sum = self.malhavol_vol_xi.sum()

    def __vf_fun_var_assem2__(self):
        rho_tst2   = TestFunction(self.rho.function_space())
        self.malhavol_vol_xi2  = assemble(rho_tst2 * Constant(1) * self.dxx(2))
        self.malhavol_vol_sum2 = self.malhavol_vol_xi2.sum()

    def __vf_fun_var_assem3__(self):
        rho_tst3   = TestFunction(self.rho.function_space())
        self.malhavol_vol_xi3  = assemble(rho_tst3 * Constant(1) * self.dxx(3))
        self.malhavol_vol_sum3 = self.malhavol_vol_xi3.sum()

    def obj_fun(self, rho, user_data=None):
        print(" \n **********************************" )
        print(" Objective Function Evaluation" )
        self.rho.vector().set_local(rho)
        self.state = self.w
        self.state2 = self.w2
        # self.funcional1, self.funcional2, visc_term = self.Funcional(self.rho, self.state, self.state2)
        self.funcional1, self.funcional2, visc_term = self.Funcional(self.rho, self.w, self.w2)
        fval = assemble(self.funcional1) #/ (assemble(self.funcional2 ) + 1e-9) #divisao aqui
        print(" fval: {}" .format(fval) )
        print(" ********************************** \n " )
        self.iter_fobj += 1
        self.file_obj = open(self.workdir + "/fun_obj.txt", "a+")

        facet_marker = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        facet_marker.set_all(0)
        self.esquerda.mark(facet_marker,1)
        self.direita.mark(facet_marker,2)
        dp = Measure("ds",domain=self.mesh,subdomain_data=facet_marker)
        (u,p) = split(self.w)
        borda_e = assemble(p*dp(1))
        borda_d = assemble(p*dp(2))
        print("Esquerda pressao: {}".format(borda_e))
        print("Direita pressao: {}".format(borda_d))

        if self.first_iter:
            self.file_obj.write("FunObj" +"\t"+ "Visc Term" +"\t"+ "Ef" +"\t"+ "Pesq" +"\t"+ "Pdir"+ "\n")
            self.first_iter = False
            self.file_obj.write(str(fval) +"\t"+ str(assemble(visc_term)) +"\t"+ str(assemble(self.funcional2)) +"\t"+ str(borda_e) +"\t"+ str(borda_d) + "\n")
        else:
            self.file_obj.write(str(fval) +"\t"+ str(assemble(visc_term)) +"\t"+ str(assemble(self.funcional2)) +"\t"+ str(borda_e) +"\t"+ str(borda_d) + "\n")
        self.file_obj.close()
        #colando aqui
        return fval, borda_e

    def obj_dfun(self, xi, user_data=None):
        print(" \n **********************************" )
        print(" \n Objective Function Gradient Evaluation \n" )
        self.rho.vector().set_local(xi)

        set_working_tape(Tape())
        w = self.get_forward_solution(self.rho)
        (self.z_n, self.r_n) = SpatialCoordinate(self.mesh)
        self.subdominios()
        self.functionsp()
        self.rho = interpolate(self.rho, self.A)
        self.w = interpolate(w, self.W)
        # ds_vars = self.density_filter(ds_vars)

        self.funcional1, self.funcional2, visc_term = self.Funcional(self.rho, w, self.w2)
        self.funcional1 = assemble(self.funcional1)
        control = Control(self.rho)
        self.reduced_f = ReducedFunctional(self.funcional1, control)

        #Derivada da funcao objetivo
        L = self.reduced_f.derivative().vector()

        self.iter_dobj += 1
        return np.array(L)

    def add_volf_constraint(self, upp, lwr):
        self.cst_U.append(upp)
        self.cst_L.append(lwr)
        self.cst_num += 1

    def add_volf_constraint2(self, upp, lwr):
        self.cst_U.append(upp)
        self.cst_L.append(lwr)

        self.cst_num += 1

        self.cst_U = np.array(self.cst_U) #Eu q pus isto
        self.cst_L = np.array(self.cst_L) #Eu q pus isto
    def add_volf_constraint3(self, upp, lwr):

        self.cst_num += 1

        self.cst_U = np.append([self.cst_U],[upp]) #Eu q pus isto
        self.cst_L = np.append([self.cst_L],[lwr]) #Eu q pus isto

    def volfrac_fun(self, xi):


        #volume_val = float( self.vol_xi.inner( self.rho.vector() ) )
        volume_val = float( self.malhavol_vol_xi.inner( self.rho.vector() ) )

        #return volume_val/self.vol_sum
        return volume_val/self.malhavol_vol_sum

    def volfrac_fun2(self, xi):

        volume_val2 = float( self.malhavol_vol_xi2.inner( self.rho.vector() ) )

        return volume_val2/self.malhavol_vol_sum2

    def volfrac_fun3(self, xi):
        volume_val3 = float( self.malhavol_vol_xi3.inner( self.rho.vector() ) )

        return volume_val3/self.malhavol_vol_sum3

    def volfrac_dfun(self, xi=None, user_data=None):
        resultado = self.malhavol_vol_xi/self.malhavol_vol_sum
        return resultado

    def volfrac_dfun2(self, xi=None, user_data=None):
        resultado = self.malhavol_vol_xi2/self.malhavol_vol_sum2
        return resultado
    def volfrac_dfun3(self, xi=None, user_data=None):
        resultado = self.malhavol_vol_xi3/self.malhavol_vol_sum3
        return resultado

    def flag_jacobian(self):
        rows = []
        for i in range(self.cst_num):
            rows += [i] * self.nvars

        cols = list(range(self.nvars)) * self.cst_num

        #return (np.array(rows, dtype=np.int), np.array(cols, dtype=np.int))
        return (np.array(rows), np.array(cols))

    def cst_fval(self, xi, user_data=None):
        if self.cst_num==1:
            cst_val = np.array(self.volfrac_fun(xi), dtype=np.float).T
        if self.cst_num==2:
            cst_val = np.array([self.volfrac_fun(xi), self.volfrac_fun2(xi)], dtype=np.float ).T
        if self.cst_num==3:
            cst_val = np.array([self.volfrac_fun(xi), self.volfrac_fun2(xi), self.volfrac_fun3(xi)], dtype=np.float ).T

        return cst_val

    def jacobian(self, xi, flag=False, user_data=None):
        print(" \n Constraint Gradient Evaluation \n" )
        if flag:
            dfval = self.flag_jacobian()
        else:
            if self.cst_num==1:
                print( "CST Value1: ", self.volfrac_fun(xi) )
                dfval = self.volfrac_dfun()
            if self.cst_num==2:
                print( "CST Value1: ", self.volfrac_fun(xi) )
                print( "CST Value2: ", self.volfrac_fun2(xi) )
                dfval = np.array([ self.volfrac_dfun(), self.volfrac_dfun2() ])
            if self.cst_num==3:
                print( "CST Value1: ", self.volfrac_fun(xi) )
                print( "CST Value2: ", self.volfrac_fun2(xi) )
                print( "CST Value3: ", self.volfrac_fun3(xi) )
                dfval = np.array([ self.volfrac_dfun(), self.volfrac_dfun2(), self.volfrac_dfun3() ])

        return dfval
    """def jacobian(self, xi):
        jacobiano = (-self.malhavol_vol_xi, -self.malhavol_vol_xi2)
        return jacobiano"""

    def sim(self, max_iter=1000):
        """
        Function that returns a pyipopt object with OptObj atributes
        """
        super().sim()
        if self.rho_initial != None:
            self.rho = Function(self.A, self.rho_initial.vector(), name="Control")
        print("Optimization Beginning")
        self.state = Function(self.W)
        self.state2 = Function(self.W2)
        if self.cst_num >= 1: self.__vf_fun_var_assem__()
        if self.cst_num >= 2: self.__vf_fun_var_assem2__()
        if self.cst_num >= 3: self.__vf_fun_var_assem3__()
        self.nvars = len(self.rho.vector())

        # Number of Design Variables
        nvar = self.nvars
        # Upper and lower bounds
        x_L = np.ones(nvar) * 0. #bounds[0]
        x_U = np.ones(nvar) * 1. #bounds[1]
        # Number of non-zeros gradients
        constraints_nnz = nvar*self.cst_num
        acst_L = np.array(self.cst_L)
        acst_U = np.array(self.cst_U)
        OptObj = tobs.create(nvar,               # number of the design variables
                            x_L,                       # lower bounds of the design variables
                            x_U,                       # upper bounds of the design variables
                            self.cst_num,            # number of constraints
                            acst_L,                    # lower bounds on constraints,
                            acst_U,                    # upper bounds on constraints,
                            constraints_nnz,           # number of nonzeros in the constraint Jacobian
                            0,                         # number of nonzeros in the Hessian
                            self.obj_fun,            # objective function
                            self.obj_dfun,           # gradient of the objective function
                            self.cst_fval,           # constraint function
                            self.jacobian )          # gradient of the constraint function

        #Parameters

        def cb_post(rho_opt, rho_notfiltered, mesh_adapt):
            self.file_mesh_adapted << mesh_adapt
            self.rho = rho_opt

        self.file_mesh_adapted << self.rho.function_space().mesh()
        self.rho.full_geo = self.full_geo
        OptObj.solve(self.rho, minimize=self.minimize, filter_fun=self.density_filter, call_back=cb_post)
        return self.rho

