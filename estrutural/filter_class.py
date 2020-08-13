# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:06 2015

@author: francisco
"""
###############################################################################
#######------------TOPOLOGY OPTIMIZATION FOR LINEAR STRUCTURES----------#######
###############################################################################


###############################################################################
#######--------------------IMPORTING LIBRARIES--------------------------#######
import numpy as np
from dolfin import *


class filter_obj(object):
    mi_e = 0
    def __init__(self, mesh, rmin=0, beta=0):
        self.mesh = mesh
        self.rmin = rmin
        self.beta = beta
        self.calculate_weight()

    def calculate_weight(self):
        pos_cel = []
        for cell in cells(self.mesh):
            pos_cel.append([cell.midpoint().x(),cell.midpoint().y()])
        pos_cel = np.array(pos_cel)

        self.Viz     = []
        dist    = []
        d_viz   = []
        #j_index = np.zeros(mesh.num_cells())
        for cell in cells(self.mesh):

            #Calculate the distance ce from node i
            d_viz = np.sqrt( np.power( cell.midpoint().x() - pos_cel[:,0] ,2) + np.power( cell.midpoint().y() - pos_cel[:,1] ,2) )

            # find the set of cells are inside radius
            j_index = np.where((d_viz <= self.rmin) & (d_viz >= 0))[0]

            # store the neighbours cells from cell i
            self.Viz.append(j_index)

            # store the neighbours distance cell from cell i
            dist.append(d_viz[j_index])
        #######-----------------------------------------------------------------#######
        ###############################################################################


        ###############################################################################
        #######-----PROJ METHOD >> CALCULATE WEIGHT FOR EACH NEIGBOURING NODE---#######
        self.weight = []
        for cell in cells(self.mesh):
            # Calculate the elements weight
            self.weight.append((self.rmin - dist[cell.index()])/self.rmin)

        #######-----------------------------------------------------------------#######
        ###############################################################################


        ###############################################################################
        #######-----PROJ METHOD >> CALCULATE REVISED ELEMENT DENSITY------------#######

    def Rho_elem (self, c):
        frho  = Function(c.function_space())
        vetore = c.vector()
        rho_e = np.zeros(self.mesh.num_cells())
        self.mi_e = []
        for cell in cells(self.mesh):
            ind = cell.index()
            we = self.weight[ind]
            ro = vetore[self.Viz[ind]]
            #Calculate the revised linear element density
            # rho_e[ind] = np.dot(ro, we) / np.sum(we)

            #Calculate the revised nonlinear element density
            #mi_e = 0.0
            mi_e = np.dot(ro, we) / np.sum(we)
            self.mi_e.append(mi_e)
            rho_e[ind] = 1 - exp(-self.beta*mi_e) + mi_e*exp(-self.beta)
            # rho_e[ind] = exp(-self.beta*mi_e) - mi_e*exp(-self.beta)

        rho_e[np.where(rho_e > 1)[0]] = 1
        rho_e[np.where(rho_e < 0)[0]] = 0

        # frho.vector()[:] = 1 - rho_e
        frho.vector()[:] = rho_e
        return frho
        #######-----------------------------------------------------------------#######
        ###############################################################################


        ###############################################################################
        #######-----PROJ METHOD >> CALCULATE REVISED SENSITIVITY----------------#######

    def Sens_elem (self, Dfdxy):
        fsens  = Function(Dfdxy.function_space())
        vetore = Dfdxy.vector()
        sens = np.zeros(self.mesh.num_cells())
        for cell in cells(self.mesh):
            ind = cell.index()
            we = self.weight[ind]
            dfdxo = vetore[self.Viz[ind]]

            #Calculate the revised element density
            # sens[ind] = np.dot(dfdxo, we) / np.sum(we)

            if self.mi_e != 0:
                dmi_e = we / np.sum(we)
                dgama_dd = self.beta * dmi_e * np.exp(-self.beta*self.mi_e[ind]) + dmi_e * np.exp(-self.beta)
                sens[ind] = np.dot(dfdxo, dgama_dd)
                # sens[ind] = exp(-self.beta*mi_e) + mi_e*exp(-self.beta)
            else:
                sens[ind] = vetore[ind]

        fsens.vector()[:] = sens
        # fsens.vector()[:] = 1 - sens
        # plot(fsens, title='Element Sensitivity',key="Sensitivity")
        return fsens
    #######-----------------------------------------------------------------#######
    ###############################################################################

