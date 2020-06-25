"""
This module creates the Optimization Object
"""
import BrunoDoc.adjoint_problem as AProb

def optimization_problem(opt_mode):
    if opt_mode.upper() == "CONT":
        import BrunoDoc.opt_cont_problem as OProb
        return OProb.OP
    elif opt_mode.upper() == "DA":
        import BrunoDoc.opt_da_problem as OProbDA
        return OProbDA.OP
    elif opt_mode.upper() == "DA.TOBS":
        import BrunoDoc.opt_datobs_problem as OProbDATobs
        return OProbDATobs.OP
    elif opt_mode.upper() == "CONT.TOBS":
        import BrunoDoc.opt_conttobs_problem as OProbContTobs
        return OProbContTobs.OP
    else:
        raise Except("No correct opt_mode was defined!")
