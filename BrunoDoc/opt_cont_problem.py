"""
This module creates the Optimization Object
"""
from dolfin import *
import BrunoDoc.opt_conttobs_problem as Optobs
import BrunoDoc.properties as Prop
import BrunoDoc.parameters as Par
import numpy
import pyipopt


class OP(Optobs.OP):
    """
    """
    def obj_dfun(self, rho, user_data=None):
        ds_vars = self.__check_ds_vars__(rho)
        ds_vars.rename("control", "control")
        self.file_out << ds_vars
        dfval = super().obj_dfun(rho)
        if self.iteration > 30:
            q.assign(0.1)
        return dfval

    def sim(self, max_iter=240):
        """
        Function that returns a pyipopt object with OptObj atributes
        """
        # super().sim()
        Prop.PP.sim(self)
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

