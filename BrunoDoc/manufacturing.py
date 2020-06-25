# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:06 2015

@author: bcaldas
"""
###############################################################################
#######------------TOPOLOGY OPTIMIZATION FOR LINEAR STRUCTURES----------#######
###############################################################################


###############################################################################
#######--------------------IMPORTING LIBRARIES--------------------------#######
import numpy as np
from dolfin import *


class casting_obj(object):
    mi_e = 0
    q = 20
    def __init__(self, mesh, rmin=0):
        self.mesh = mesh
        self.rmin = rmin
        self.calculate_weight()

    def calculate_weight(self):
        pos_cel = []
        for cell in cells(self.mesh):
            pos_cel.append([cell.midpoint().x(),cell.midpoint().y()])
        pos_cel = np.array(pos_cel)

        self.Viz     = []
        dist    = []
        d_x   = []
        d_y   = []
        #j_index = np.zeros(mesh.num_cells())
        for cell in cells(self.mesh):

            #Calculate the distance ce from node i
            # d_viz = np.sqrt( np.power( cell.midpoint().x() - pos_cel[:,0] ,2) + np.power( cell.midpoint().y() - pos_cel[:,1] ,2) )
            d_x = np.abs( cell.midpoint().x() - pos_cel[:,0])
            # d_y = np.abs( cell.midpoint().y() - pos_cel[:,1])
            d_y = cell.midpoint().y() - pos_cel[:,1]

            # find the set of cells are inside radius
            # j_index = np.where((d_y <= self.rmin) & (d_x <= 0.001) )[0]
            # j_index = np.where((d_y >= 0) & (d_x <= 0.001) )[0]
            j_index = np.where(d_x <= 0.25)[0]

            # store the neighbours cells from cell i
            self.Viz.append(j_index)

            # store the neighbours distance cell from cell i
            dist.append(d_y[j_index])
        #######-----------------------------------------------------------------#######
        ###############################################################################


        ###############################################################################
        #######-----PROJ METHOD >> CALCULATE WEIGHT FOR EACH NEIGBOURING NODE---#######
        self.weight = []
        for cell in cells(self.mesh):
            # Calculate the elements weight
            # self.weight.append((self.rmin - dist[cell.index()])/self.rmin)
            x = np.array( [ 1 if num>=0 else 0 for num in dist[cell.index()] ] )
            self.weight.append(x)
            # self.weight.append(np.ones(dist[cell.index()].shape))

        #######-----------------------------------------------------------------#######
        ###############################################################################


        ###############################################################################
        #######-----PROJ METHOD >> CALCULATE REVISED ELEMENT DENSITY------------#######

    def Rho_elem (self, c):
        frho  = Function(c.function_space())
        self.vetore = c.vector()
        rho_e = np.zeros(self.mesh.num_cells())
        self.mi_e = []
        for cell in cells(self.mesh):
            ind = cell.index()
            we = self.weight[ind]
            ro = self.vetore[self.Viz[ind]]
            #Calculate the revised linear element density
            # rho_e[ind] = np.dot(ro, we) #/ np.sum(we)
            temp = np.array([element**500 for element in 1-ro])#.sum()**(1/500)
            temp = temp[np.where(we > 0.5)]
            rho_e[ind] = temp.sum()**(1/500)
            # rho_e[ind] = np.dot(ro, we) #/ np.sum(we)

            # if ind>200: import pdb; pdb.set_trace()
            #Calculate the revised nonlinear element density
            #mi_e = 0.0

        rho_e = rho_e
        rho_e[np.where(rho_e > 1)[0]] = 1
        rho_e[np.where(rho_e < 0)[0]] = 0

        frho.vector()[:] = 1 - rho_e
        self.rho = rho_e
        # frho.vector()[:] = rho_e
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
            ro = self.vetore[self.Viz[ind]]

            #Calculate the revised element density
            # sens[ind] = np.dot(dfdxo, we) # / np.sum(we)
            # drho_dd = (1-ro) * np.array([element**500 for element in 1-ro]).sum()**(1/500-1)
            temp = np.array([element**500 for element in 1-ro])
            temp = temp[np.where(we > 0.5)].sum()
            temp **= (1/500-1)
            temp = (1-ro)**(500-1) * temp
            temp[np.isnan(temp)] = 0
            # temp[np.where(temp<0)] = 0
            # temp[np.where(temp<1)] = 1
            temp[np.isinf(temp)] = 1
            sens[ind] = np.dot(dfdxo, temp)


        fsens.vector()[:] = sens
        # plot(fsens, title='Element Sensitivity',key="Sensitivity")
        return fsens
    #######-----------------------------------------------------------------#######
    ###############################################################################

