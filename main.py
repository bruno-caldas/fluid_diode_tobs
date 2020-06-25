"""
Main File to simulate Bruno Caldas PhD Work
"""
import re, sys
from BrunoDoc.read_param_file import *
import BrunoDoc.optimization_problem as OProb
# import BrunoDoc.optimization_problem_dolfinadj as OProb
#import pdb; pdb.set_trace() # Use this line to interrupt wherever you want
import BrunoDoc.opt_check_finite_dif as Test
from dolfin import *

if __name__ == "__main__":
    # TEST = Test.OP(hash='testando_df')
    # TEST.forward_problem = '2DSwirl'
    # TEST.omega = 2000/60*2*3.1415
    # TEST.sim()
    # exit()

    OP = OProb.optimization_problem(linha[9])
    OPTIM_1 = OP(rho_initial=None, hash='opt_config1_'+str(linha_num))
    OPTIM_1.config = 1
    OPTIM_1.forward_problem = '2D' # 2D or 2DSwirl
    OPTIM_1.omega = float(linha[1])/60*2*3.1415
    OPTIM_1.diodicidade = True
    V_U1 = 1.0#*base*altura PORCENTAGEM DE FLUIDO SEMPRE
    V_L1 = 0.85#*base*altura
    V_U2 = float(linha[5])#*base*altura
    V_L2 = float(linha[6])#*base*altura
    V_U3 = 1.0#*base*altura
    V_L3 = 0.85#*base*altura
    OPTIM_1.r_min = float(linha[10])
    # OPTIM_1.alphabar = 2.5 * OPTIM_1.mu * 1e-6 * float(linha[7])    # kg/ (m**3 *s)
    # OPTIM_1.alphaJbar = 2.5 * OPTIM_1.mu * 1e-6 * float(linha[8])    # kg/ (m**3 *s)
    OPTIM_1.alphabar = Constant(float(linha[7]))    # kg/ (m**3 *s)
    # OPTIM_1.alphaJbar = Constant(float(linha[8]))    # kg/ (m**3 *s)
    OPTIM_1.add_volf_constraint(V_U1,V_L1)
    OPTIM_1.add_volf_constraint2(V_U2,V_L2)
    OPTIM_1.add_volf_constraint3(V_U3,V_L3)
    OPTIM_1.subiterations = 5
    OPTIM_1.iterations = 400
    RHO_INTERM = OPTIM_1.sim(max_iter=500)
