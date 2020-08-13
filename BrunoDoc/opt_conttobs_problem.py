"""
This module creates the Optimization Object
"""
from dolfin import *
import BrunoDoc.adjoint_problem as AProb
import BrunoDoc.parameters as Par
import numpy
import BrunoDoc.tobs as tobs
import math
from estrutural import main as estrut

from BrunoDoc.read_param_file import *

delta = float(linha[2])
gap = float(linha[3])
radius = float(linha[4])
altura = 1.5

class OP(AProb.AP):
    """
    This Class supplies a python object that has all Ipopt needs,
    such as ObjFun, Gradientes and Constraints
    """
    iteration = 0
    subiterations = 5
    first_iter = True
    first_iter_deriv = True
    opt_mode = "CONT"
    filter_f = None
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
        self.file_domain = File(self.workdir + "/domain.pvd")
        self.rho_initial = rho_initial
        self.file_sen = File(self.workdir + '/sensibility.pvd')
        self.file_filtrado = File(self.workdir + '/controlFiltrered.pvd')

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

    @staticmethod
    def estrutural(rho):
        j, jd = estrut.estrutural(rho.copy(deepcopy=True))
        return j, jd

    def obj_fun(self, rho, user_data=None):
        print(" \n **********************************" )
        print(" Objective Function Evaluation" )
        ds_vars = self.__check_ds_vars__(rho)
        funcional1, funcional2, visc_term1, visc_term2, Fstar, w, w2 = self.Funcional(ds_vars)
        if assemble(funcional2) == 0: fval = 0
        fval = assemble(funcional1) / assemble(funcional2) + assemble(Fstar)
        print(" fval: {}" .format(fval) )
        print(" ********************************** \n " )
        self.iter_fobj += 1
        self.file_obj = open(self.workdir + "/fun_obj.txt", "a+")

        facet_marker = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        facet_marker.set_all(0)
        self.esquerda.mark(facet_marker,1)
        self.direita.mark(facet_marker,2)
        dp = Measure("ds",domain=self.mesh,subdomain_data=facet_marker)
        (u,p) = split(w)
        (u2,p2) = split(w2)
        borda_e = assemble(p*dp(1))/gap
        borda_e0 = assemble(p*dp(2))/gap
        borda_d0 = assemble(p2*dp(1))/gap
        borda_d = assemble(p2*dp(2))/gap
        borda_e = borda_e - borda_e0
        borda_d = borda_d - borda_d0
        print("Esquerda pressao: {}".format(borda_e))
        print("Direita pressao: {}".format(borda_d))

        if self.first_iter:
            self.file_obj.write("FunObj" +"\t"+ "Volume" + "\t"+ "Fstar" +"\t"+ "Pida" +"\t"+ "Pvolta"+ "\n")
            self.first_iter = False
        self.file_obj.write(str(fval) +"\t"+ str(assemble(self.rho*self.dxx(2))/3*100)+ "\t"+ str(assemble(Fstar)) +"\t"+ str(borda_e) +"\t"+ str(borda_d) + "\n")
        self.file_obj.close()
        #colando aqui
        if math.isnan(fval): fval = 0

        j, jd = self.estrutural(ds_vars)
        fval += j
        return fval# , borda_e

    def obj_dfun(self, xi, user_data=None):
        '''if self.iter_dobj == 0:
            error = self.test_obj_dfun(xi)
            for err in error:
                if err > .10: print("Warning: Error too big in the functional gradient!")'''

        print(" \n **********************************" )
        print(" \n Objective Function Gradient Evaluation \n" )
        ds_vars  = self.__check_ds_vars__(xi)
        ds_vars.rename("Control", "Control")

        mesh = ds_vars.function_space().mesh()

        funcional1, funcional2, visc_term1, visc_term2, Fstar, w, w2 = self.Funcional(ds_vars) #, self.state, self.state2)
        w3 = w2.copy(deepcopy=True) #FIXME arrumar aqui

        lam1 = self.get_adjoint_solution(ds_vars, w)
        lam2 = self.get_adjoint_solution2(ds_vars, w2)
        lam3 = self.get_adjoint_solution3(ds_vars, w3)

        fval1 = assemble(funcional1)
        fval2 = assemble(funcional2)
        L1 = numpy.array(lam1)
        L2 = numpy.array(lam2)
        L3 = numpy.array(lam3)

        self.file_analise = open(self.workdir + "/analise.txt", "a+")
        self.file_analise.write(str(fval1) + '\t' + str(fval2) + '\t' + str(assemble(Fstar)) + '\t' \
                + str(assemble(ds_vars*dx)) + '\t'+ str(L1.sum()) + '\t' + str(L2.sum()) +'\t' \
                + str(L3.sum()) + '\n')

        if fval1 ==0: L = 0 * L1
        elif fval2 ==0: L = 0 * L2
        # else: L = 2 * L1/fval2 - 1 * L2/fval2
        else:
            L = L1/fval2 - 1 *fval1/fval2**2 * L2 + L3
        L = L1/fval2 - 1 *fval1/fval2**2 * L2 + L3
        # L = L / (delta*altura + 2*0.5*1)

        sensibility = ds_vars.copy(deepcopy=True)
        sensibility.vector().set_local(L)

        for cell in cells(mesh):
            if cell.midpoint().x() > delta or cell.midpoint().x() < 0:# or cell.midpoint().y() < gap:
                sensibility.vector()[cell.index()] = L.min()

        sensibility.rename("Sensitivity", "Sensitivity")
        self.file_sen << sensibility
        L = numpy.array(sensibility.vector())
        # L /= numpy.absolute(L).max()

        self.file_deriv = open(self.workdir + "/derivada_cont.txt", "a+")
        if self.first_iter_deriv:
            self.file_deriv.write("Derivada" +"\t"+ "Volume" + "\n")
            self.first_iter_deriv = False
        else:
            self.file_deriv.write(str(L.sum()) +"\t" + str(assemble(self.rho * dx)) + "\n")

        self.file_deriv.close()

        self.iter_dobj += 1

        j, jd = self.estrutural(ds_vars)
        L += jd
        return L

    def test_obj_dfun(self, xi, user_data=None):
        ds_vars = self.__check_ds_vars__(xi)

        mesh = ds_vars.function_space().mesh()

        ds_vars.rename("ControlFiltered", "ControlFiltered")

        random_cells = numpy.random.randint(10, size=(1600))
        # random_cells = [*range(1600)]
        self.file_analise = open(self.workdir + "/analise.txt", "a+")
        self.file_analise.write(' \t Point Number\tFinite Diference\tSensibility\tError\n')

        ds_vars.vector()[:] = 1
        funcional1, funcional2, visc_term1, visc_term2, Fstar, w, w2 = self.Funcional(ds_vars, save_results=False)
        w3 = w2.copy(deepcopy=True)
        lam1 = self.get_adjoint_solution(ds_vars, w)
        lam2 = self.get_adjoint_solution2(ds_vars, w2)
        lam3 = self.get_adjoint_solution3(ds_vars, w3)
        fval1 = assemble(funcional1)
        fval2 = assemble(funcional2)
        fval3 = assemble(Fstar)
        fval_0 = fval1/fval2 + fval3

        L = lam1/fval2 - 1 *fval1/fval2**2 * lam2 + lam3

        '''a = TrialFunction(ds_vars.function_space())
        b = TestFunction(ds_vars.function_space())
        M = assemble(inner(a, b)*dx)
        dL2 = Function(ds_vars.function_space(), name="deriv")
        solve(M, dL2.vector(), L)
        L = numpy.array(dL2.vector()) * .03333333**2'''
        L = numpy.array(L)

        error_cells = []
        sens = ds_vars.copy(deepcopy=True)
        sens.vector()[:] = 0
        delta_ds_vars = 1e-7
        file_teste = File(self.workdir + "/sens_error.pvd")
        for rcell in random_cells:
            ds_vars.vector()[rcell] = 1 - delta_ds_vars
            funcional1, funcional2, visc_term1, visc_term2, Fstar, w, w2 = self.Funcional(ds_vars, save_results=False)
            fval1 = assemble(funcional1)
            fval2 = assemble(funcional2)
            fval3 = assemble(Fstar)
            fval_1 = fval1/fval2 + fval3

            finite_diference = (fval_0 - fval_1) / delta_ds_vars
            error = (finite_diference-L[rcell])/L[rcell]
            self.file_analise.write('Point ID:\t' +str(rcell) + '\t' + \
                    str(finite_diference) + '\t' + str(L[rcell]) + '\t' + str(error) + '\n')
            ds_vars.vector()[:] = 1
            error_cells.append(error)
            sens.vector()[rcell] = finite_diference
            file_teste << sens

        self.file_analise.close()

        return error_cells

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

        volume_val = numpy.array(assemble(self.rho_tst * self.rho* self.dxx(1))).sum()
        # volume_val = float( self.malhavol_vol_xi.inner( self.rho.vector() ) )

        #return volume_val/self.vol_sum
        return volume_val/self.malhavol_vol_sum

    def volfrac_fun2(self, xi):

        #self.__check_ds_vars__(xi)
        # volume_val2 = float( self.malhavol_vol_xi2.inner( self.rho.vector() ) )
        volume_val2 = numpy.array(assemble(self.rho_tst * self.rho* self.dxx(2))).sum()

        return volume_val2/self.malhavol_vol_sum2

    def volfrac_fun3(self, xi):
        # volume_val3 = float( self.malhavol_vol_xi3.inner( self.rho.vector() ) )
        volume_val3 = numpy.array(assemble(self.rho_tst * self.rho* self.dxx(3))).sum()

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
            cst_val = numpy.array([self.volfrac_fun(xi), self.volfrac_fun2(xi), self.volfrac_fun2(xi),self.volfrac_fun3(xi)], dtype=numpy.float ).T

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
                dfval = numpy.array([ self.volfrac_dfun(), self.volfrac_dfun2(), self.volfrac_dfun2(), self.volfrac_dfun3() ])

        return dfval

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

        def cb_post(rho_opt, rho_notfiltered, mesh_adapt, domain, iteration):
            # self.file_mesh_adapted << mesh_adapt
            self.rho = rho_opt
            # self.file_domain << domain
            rho_opt.rename("controlNotFiltered", "controlNotFiltered")
            self.file_out << rho_opt
            self.iteration = iteration
            # self.rotating_parts = domain
            '''if iteration == 30:
                self.alphabar.assign(2.5e4)'''
            '''if iteration == 20:
                self.alphabar.assign(3.33e5)
            if iteration == 30:
                self.alphabar.assign(3.33e6)
            if iteration == 40:
                self.alphabar.assign(3.33e7)'''

        self.file_mesh_adapted << self.rho.function_space().mesh()
        self.rho.full_geo = self.full_geo
        OptObj.solve(self.rho, minimize=self.minimize, filter_fun=None, call_back=cb_post)
        return self.rho

