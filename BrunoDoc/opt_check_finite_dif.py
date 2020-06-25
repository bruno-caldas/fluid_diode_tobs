"""
This module creates the Optimization Object
"""
from dolfin import *
import BrunoDoc.adjoint_problem as AProb
import BrunoDoc.parameters as Par
import numpy

class Distribution(UserExpression):
    def __init__(self, inc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inc = inc

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0
        if near(x[0],0) and near(x[1], 40.5):
            values[0] += self.inc


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
        self.evolucao_rho = File(self.workdir + '/evolucao_rho.pvd')
        self.file_out = File(self.workdir + "/control.pvd")
        self.state_file = File(self.workdir + "/state.pvd")
        self.state_file2 = File(self.workdir + "/state2.pvd")
        self.rho_initial = rho_initial
        self.deriv_visual = File(self.workdir + "/deriv_visual.pvd")

    def obj_fun(self, rho, user_data=None):
        print(" \n **********************************" )
        print(" Objective Function Evaluation" )
        self.state = self.w
        self.state2 = self.w2
        self.funcional1, self.funcional2 = self.Funcional(self.rho, self.state, self.state2)
        fval = assemble(self.funcional1) #- assemble(self.funcional2 ) #divisao aqui
        print(" fval: {}" .format(fval) )
        print(" ********************************** \n " )
        return fval

    def obj_dfun(self, xi, user_data=None):
        print(" \n **********************************" )
        print(" \n Objective Function Gradient Evaluation \n" )
        self.funcional1, self.funcional2 = self.Funcional(self.rho, self.state, self.state2)
        lam = self.get_adjoint_solution(self.rho)
        #lam2 = self.get_adjoint_solution2(self.rho)

        dmo = TestFunction( self.rho.function_space() )
        L = assemble(lam *dmo*dx)

        import pdb; pdb.set_trace()
        return numpy.array(L)


    def sim(self, max_iter=1000):
        """
        Function that returns a pyipopt object with OptObj atributes
        """
        super().sim()
        if self.rho_initial != None:
            self.rho = Function(self.A, self.rho_initial.vector(), name="Control")
        print("Gradient Checking")
        self.state = Function(self.W)
        self.state2 = Function(self.W2)
        for i in range(10):
            print("Valor i: {}" .format(i))
            f0 = self.obj_fun(self.rho)
            self.rho = interpolate(Distribution(i), self.A)
            self.w = self.get_forward_solution(self.rho)
            f1 = self.obj_fun(self.rho)
            df = self.obj_dfun(self.rho)
            self.rho.rename("control", "")
            self.evolucao_rho << self.rho

        return self.rho

