"""
This module creates the Optimization Object
"""
from dolfin import *
import BrunoDoc.adjoint_problem as AProb
import BrunoDoc.parameters as Par
import numpy
import pyipopt


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
        self.rho_initial = rho_initial

    def __check_ds_vars__(self, xi):
        """
        Method which checks the design variables
        """
        chk_var = False
        try: #If self.xi_array has yet not been defined
            xi_eval = self.xi_array - xi
            xi_nrm  = numpy.linalg.norm(xi_eval)
            if xi_nrm > 1e-16:
                self.xi_array = numpy.copy(xi)
                chk_var = True
        except AttributeError as error:
            self.xi_array = numpy.copy(xi)
            chk_var = True
        if chk_var is True:
            self.rho.vector()[:] = xi
        else:
            print(" *** Recycling the design variables...")
        ds_vars = self.rho
        return ds_vars

        '''    def __vf_fun_var_assem__(self):
        rho_tst   = TestFunction(self.rho.function_space())
        self.vol_xi  = assemble(rho_tst * Constant(1) * dx)
        self.vol_sum = self.vol_xi.sum()'''

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
        ds_vars = self.__check_ds_vars__(rho)
        self.state = self.w
        self.state2 = self.w2
        self.funcional1, self.funcional2, visc_term = self.Funcional(self.rho, self.state, self.state2)
        fval = assemble(self.funcional1) #/ (assemble(self.funcional2 ) + 1e-9) #divisao aqui
        print(" fval: {}" .format(fval) )
        print(" ********************************** \n " )
        if self.file_out is not None:
            self.rho.rename("control", "label")
            self.file_out << self.rho
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
        return fval

    def obj_dfun(self, xi, user_data=None):
        print(" \n **********************************" )
        print(" \n Objective Function Gradient Evaluation \n" )
        ds_vars  = self.__check_ds_vars__(xi)
        self.funcional1, self.funcional2, visc_term = self.Funcional(ds_vars, self.state, self.state2)

        lam = self.get_adjoint_solution(ds_vars)
        # lam2 = self.get_adjoint_solution2(ds_vars)

        self.seq_mu = [self.seq_mu[-1]] # Matar a continuidade para posteriores simulacoes
        dmo = TestFunction( self.rho.function_space() )

        # L = assemble(dmo*lam/(assemble(self.funcional2) + 1e-9) * dx - dmo*assemble(self.funcional1) * lam2**-2 * dx ) #seria menos aqui?
        L = assemble(lam *dmo*dx)

        self.file_deriv = open(self.workdir + "/derivada_cont.txt", "a+")
        if self.first_iter_deriv:
            self.file_deriv.write("Derivada" +"\t"+ "delta_f" + "\n")
            self.first_iter_deriv = False
        else:
            self.file_deriv.write(str(numpy.array(L).sum()) +"\t"+ str(assemble(self.funcional1)) + "\t" + str(assemble(self.rho * dx)) + "\n")

        self.file_deriv.close()


        self.iter_dobj += 1
        return numpy.array(L)

    def add_volf_constraint(self, upp, lwr):
        self.cst_U.append(upp)
        self.cst_L.append(lwr)
        self.cst_num += 1

    def add_volf_constraint2(self, upp, lwr):
        self.cst_U.append(upp)
        self.cst_L.append(lwr)

        self.cst_num += 1

        self.cst_U = numpy.array(self.cst_U) #Eu q pus isto
        self.cst_L = numpy.array(self.cst_L) #Eu q pus isto
    def add_volf_constraint3(self, upp, lwr):

        self.cst_num += 1

        self.cst_U = numpy.append([self.cst_U],[upp]) #Eu q pus isto
        self.cst_L = numpy.append([self.cst_L],[lwr]) #Eu q pus isto

    def volfrac_fun(self, xi):

        self.__check_ds_vars__(xi)
        #self.__check_ds_vars__(self.malhavol.vector()[:])

        #volume_val = float( self.vol_xi.inner( self.rho.vector() ) )
        volume_val = float( self.malhavol_vol_xi.inner( self.rho.vector() ) )

        #return volume_val/self.vol_sum
        return volume_val/self.malhavol_vol_sum

    def volfrac_fun2(self, xi):

        #self.__check_ds_vars__(xi)
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

        #return (numpy.array(rows, dtype=numpy.int), numpy.array(cols, dtype=numpy.int))
        return (numpy.array(rows), numpy.array(cols))

    def cst_fval(self, xi, user_data=None):
        if self.cst_num==1:
            cst_val = numpy.array(self.volfrac_fun(xi), dtype=numpy.float).T
        if self.cst_num==2:
            cst_val = numpy.array([self.volfrac_fun(xi), self.volfrac_fun2(xi)], dtype=numpy.float ).T
        if self.cst_num==3:
            cst_val = numpy.array([self.volfrac_fun(xi), self.volfrac_fun2(xi), self.volfrac_fun3(xi)], dtype=numpy.float ).T

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
                dfval = numpy.array([ self.volfrac_dfun(), self.volfrac_dfun2() ])
            if self.cst_num==3:
                print( "CST Value1: ", self.volfrac_fun(xi) )
                print( "CST Value2: ", self.volfrac_fun2(xi) )
                print( "CST Value3: ", self.volfrac_fun3(xi) )
                dfval = numpy.array([ self.volfrac_dfun(), self.volfrac_dfun2(), self.volfrac_dfun3() ])

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
        x_L = numpy.ones(nvar) * 0. #bounds[0]
        x_U = numpy.ones(nvar) * 1. #bounds[1]
        # Number of non-zeros gradients
        constraints_nnz = nvar*self.cst_num
        acst_L = numpy.array(self.cst_L)
        acst_U = numpy.array(self.cst_U)
        PyIpOptObj = pyipopt.create(nvar,               # number of the design variables
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
        PyIpOptObj.num_option('obj_scaling_factor', 1.0) #MAXIMIZE OR MINIMIZE
        PyIpOptObj.num_option('acceptable_tol', 1.0e-10)
        PyIpOptObj.num_option('eta_phi', 1e-12)                 # eta_phi: Relaxation factor in the Armijo condition.
        PyIpOptObj.num_option('theta_max_fact', 30000)	        # Determines upper bound for constraint violation in the filter.
        PyIpOptObj.int_option('max_soc', 20)
        PyIpOptObj.int_option('max_iter', max_iter)
        PyIpOptObj.int_option('watchdog_shortened_iter_trigger', 20)
        PyIpOptObj.int_option('accept_after_max_steps', 5)
        pyipopt.set_loglevel(1)                                 # turn off annoying pyipopt logging
        PyIpOptObj.int_option('print_level', 6)                 # very useful IPOPT output

        x0 = numpy.copy(self.rho.vector())
        self.rho.vector()[:], zl, zu, constraint_multipliers, obj, status = PyIpOptObj.solve(x0)
        return self.rho

