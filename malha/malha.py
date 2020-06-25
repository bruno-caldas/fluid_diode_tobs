from dolfin import *
from mshr import *

import numpy as np

def rectangle(x, y, nx=None, ny=None, diagonal='right'):
    """
    Return a mesh over a domain with the shape of a rectangle.
    ============  =================================================
    Name          Description
    ============  =================================================
    x, y          interval [,] or coordinate arrays
    nx, ny        integers reflecting the division of intervals
    diagonal      string specifying the direction of diagonals
                  ('left', 'right', 'left/right', 'right/left',
                  'crossed')
    ============  =================================================
    In the x and y directions one can either specify an interval
    to be uniformly partitioned, with a given number of
    divisions (nx or ny), or one can specify a coordinate array
    for non-uniformly partitioned structured meshes.
    Examples::
        # Unit square
        mesh = box(x=[0, 1], y=[0, 1], nx=10, ny=12)
        # Specified graded mesh in y direction
        y = [0, 0.1, 0.2, 0.5, 1, 2]  # implies nx=len(x)
        mesh = box(x=[0, 3], y=y, nx=12)
    """

    for arg in x, y:
        if not isinstance(arg, (list,tuple,np.ndarray)):
            raise TypeError('box: x, y, z must be list, tuple or numpy '\
                            'array, not %s' % type(arg))
    if len(x) == 2:
        if nx is None:
            raise ValueError('box: interval in x %s, no nx set' % x)
        x = np.linspace(x[0], x[1], nx+1)
    else:
        nx = len(x)-1
    if len(y) == 2:
        if nx is None:
            raise ValueError('box: interval in y %s, no ny set' % y)
        y = np.linspace(y[0], y[1], ny+1)
    else:
        ny = len(y)-1

    valid_diagonals = 'left', 'right', 'left/right', 'right/left', 'crossed'
    if not diagonal in valid_diagonals:
        raise ValueError('rectangle: wrong value of diagonal="%s", not in %s' \
                         % (diagonal, ', '.join(valid_diagonals)))

    editor = MeshEditor()
    mesh = Mesh()
    tdim = gdim = 2
    editor.open(mesh, 'triangle', tdim, gdim)

    if diagonal == 'crossed':
        editor.init_vertices((nx+1)*(ny+1) + nx*ny)
        editor.init_cells(4*nx*ny)
    else:
        editor.init_vertices((nx+1)*(ny+1))
        editor.init_cells(2*nx*ny)

    vertex = 0
    for iy in range(ny+1):
        for ix in range(nx+1):
            editor.add_vertex(vertex, Point(x[ix], y[iy]))
            vertex += 1
    if diagonal == 'crossed':
        for iy in range(ny):
            for ix in range(nx):
                x_mid = 0.5*(x[ix+1] + x[ix])
                y_mid = 0.5*(y[iy+1] + y[iy])
                editor.add_vertex(vertex, Point(x_mid, y_mid))
                vertex += 1

    cell = 0
    if diagonal == 'crossed':
        for iy in range(ny):
            for ix in range(nx):
                v0 = iy*(nx+1) + ix
                v1 = v0 + 1
                v2 = v0 + (nx+1)
                v3 = v1 + (nx+1)
                vmid = (nx+1)*(ny+1) + iy*nx + ix

                # Note that v0 < v1 < v2 < v3 < vmid.
                editor.add_cell(cell, v0, v1, vmid);  cell += 1
                editor.add_cell(cell, v0, v2, vmid);  cell += 1
                editor.add_cell(cell, v1, v3, vmid);  cell += 1
                editor.add_cell(cell, v2, v3, vmid);  cell += 1

    else:
        local_diagonal = diagonal
        # Set up alternating diagonal
        for iy in range(ny):
            if diagonal == "right/left":
                if iy % 2 == 0:
                    local_diagonal = "right"
                else:
                    local_diagonal = "left"

            if diagonal == "left/right":
                if iy % 2 == 0:
                    local_diagonal = "left"
                else:
                    local_diagonal = "right"
            for ix in range(nx):
                v0 = iy*(nx + 1) + ix
                v1 = v0 + 1
                v2 = v0 + nx+1
                v3 = v1 + nx+1

                if local_diagonal == "left":
                    editor.add_cell(cell, [v0, v1, v2]);  cell += 1
                    editor.add_cell(cell, [v1, v2, v3]);  cell += 1
                    if diagonal == "right/left" or diagonal == "left/right":
                        local_diagonal = "right"
                else:
                    editor.add_cell(cell, [v0, v1, v3]);  cell += 1
                    editor.add_cell(cell, [v0, v2, v3]);  cell += 1
                    if diagonal == "right/left" or diagonal == "left/right":
                        local_diagonal = "left"

    editor.close()
    mesh.structured_mesh = (x, y)
    return mesh


def valve(delta, n=None, diagonal='right'):
    """
    Return a mesh over a domain with the shape of a rectangle.
    """

    if n is None:
        raise ValueError('box:  no n set')
    x_inlet = np.linspace(-1, 0, 2*n+1) #Discretization of inlet/outlet
    y_inlet = np.linspace(0, 0.5, n+1) #Discretization of inlet/outlet
    x= np.linspace(0, delta, 4*n+1) #Discretization of domain
    y= np.linspace(0, 1.5, 3*n+1) #Discretization of domain
    x_outlet = np.linspace(delta, delta+1, 2*n+1) #Discretization of inlet/outlet
    y_outlet = np.linspace(0, 0.5, n+1) #Discretization of inlet/outlet

    editor = MeshEditor()
    mesh = Mesh()
    tdim = gdim = 2
    editor.open(mesh, 'triangle', tdim, gdim)

    nvertices = (
            len(x_inlet) * len(y_inlet) +
            len(x_outlet) * len(y_outlet) +
            len(x) * len(y)
            )
    nvertices -= len(y_inlet) + len(y_outlet)# (10+1) + (10+1)
    ncells = 2*(
            (len(x_inlet)-1) * (len(y_inlet)-1) +
            (len(x_outlet)-1) * (len(y_outlet)-1) +
            (len(x)-1) * (len(y)-1)
            )
    editor.init_vertices(nvertices)
    editor.init_cells(ncells)

    vertex = 0
    cont = 0
    corresponding = {}
    design_points = None
    for iy in y_inlet:
        for ix in x_inlet:
            editor.add_vertex(vertex, Point(ix, iy))
            vertex += 1
            if design_points is None:
                design_points = np.array([ix, iy])
            else:
                design_points = np.vstack((design_points, np.array([ix, iy]) ))
            cont += 1
    inlet_limit = cont
    print("inlet limit {}".format(inlet_limit))

    for iy in y:
        for ix in x:
            search = np.where((design_points == (ix, iy)).all(axis=1))[0]
            if search.size == 0:
                editor.add_vertex(vertex, Point(ix, iy))
                corresponding[cont] = vertex
                vertex += 1
            else:
                corresponding[cont] = search[0]

            design_points = np.vstack((design_points, np.array([ix, iy]) ))
            cont += 1
    design_limit = cont
    print("design limit {}".format(design_limit))

    for iy in y_outlet:
        for ix in x_outlet:
            search = np.where((design_points == (ix, iy)).all(axis=1))[0]
            if search.size == 0:
                editor.add_vertex(vertex, Point(ix, iy))
                corresponding[cont] = vertex
                vertex += 1
            else:
                corresponding[cont] = corresponding[search[0]]

            cont += 1
    outlet_limit = cont
    print("outlet limit {}".format(outlet_limit))

    cell = 0
    local_diagonal = diagonal
    # Set up alternating diagonal
    for iy in range(len(y_inlet)-1):
        if diagonal == "right/left":
            if iy % 2 == 0:
                local_diagonal = "right"
            else:
                local_diagonal = "left"

        if diagonal == "left/right":
            if iy % 2 == 0:
                local_diagonal = "left"
            else:
                local_diagonal = "right"
        for ix in range(len(x_inlet)-1):
            nx = len(x_inlet)-1

            v0 = iy*(nx + 1) + ix
            v1 = v0 + 1
            v2 = v0 + nx+1
            v3 = v1 + nx+1

            if local_diagonal == "left":
                editor.add_cell(cell, [v0, v1, v2]);  cell += 1
                editor.add_cell(cell, [v1, v2, v3]);  cell += 1
                if diagonal == "right/left" or diagonal == "left/right":
                    local_diagonal = "right"
            else:
                editor.add_cell(cell, [v0, v1, v3]);  cell += 1
                editor.add_cell(cell, [v0, v2, v3]);  cell += 1
                if diagonal == "right/left" or diagonal == "left/right":
                    local_diagonal = "left"
    local_diagonal = diagonal
    # Set up alternating diagonal
    for iy in range(len(y)-1):
        if diagonal == "right/left":
            if iy % 2 == 0:
                local_diagonal = "right"
            else:
                local_diagonal = "left"

        if diagonal == "left/right":
            if iy % 2 == 0:
                local_diagonal = "left"
            else:
                local_diagonal = "right"
        for ix in range(len(x)-1):
            nx = len(x)-1

            v0 = iy*(nx + 1) + ix + inlet_limit
            v1 = v0 + 1
            v2 = v0 + nx+1
            v3 = v1 + nx+1

            if v0 in corresponding: v0 = corresponding[v0]
            if v1 in corresponding: v1 = corresponding[v1]
            if v2 in corresponding: v2 = corresponding[v2]
            if v3 in corresponding: v3 = corresponding[v3]

            if local_diagonal == "left":
                editor.add_cell(cell, [v0, v1, v2]);  cell += 1
                editor.add_cell(cell, [v1, v2, v3]);  cell += 1
                if diagonal == "right/left" or diagonal == "left/right":
                    local_diagonal = "right"
            else:
                editor.add_cell(cell, [v0, v1, v3]);  cell += 1
                editor.add_cell(cell, [v0, v2, v3]);  cell += 1
                if diagonal == "right/left" or diagonal == "left/right":
                    local_diagonal = "left"
    local_diagonal = diagonal
    # Set up alternating diagonal
    for iy in range(len(y_outlet)-1):
        if diagonal == "right/left":
            if iy % 2 == 0:
                local_diagonal = "right"
            else:
                local_diagonal = "left"

        if diagonal == "left/right":
            if iy % 2 == 0:
                local_diagonal = "left"
            else:
                local_diagonal = "right"
        for ix in range(len(x_outlet)-1):
            nx = len(x_outlet)-1
            ny = len(y_outlet)-1

            v0 = iy*(nx + 1) + ix + design_limit
            v1 = v0 + 1
            v2 = v0 + nx+1
            v3 = v1 + nx+1

            if v0 in corresponding: v0 = corresponding[v0]
            if v1 in corresponding: v1 = corresponding[v1]
            if v2 in corresponding: v2 = corresponding[v2]
            if v3 in corresponding: v3 = corresponding[v3]

            if local_diagonal == "left":
                editor.add_cell(cell, [v0, v1, v2]);  cell += 1
                editor.add_cell(cell, [v1, v2, v3]);  cell += 1
                if diagonal == "right/left" or diagonal == "left/right":
                    local_diagonal = "right"
            else:
                editor.add_cell(cell, [v0, v1, v3]);  cell += 1
                editor.add_cell(cell, [v0, v2, v3]);  cell += 1
                if diagonal == "right/left" or diagonal == "left/right":
                    local_diagonal = "left"

    editor.close()
    mesh.structured_mesh = (x, y)
    return mesh

if __name__ == '__main__':
    p1 = Point(0,0)
    p2 = Point(2,1)

    # malha = Rectangle(p1, p2)
    # malha = generate_mesh(malha, 20)

    # malha = rectangle([p1.x(), p2.x()], [p1.y(), p2.y()], 40, 20)

    malha = valve(1.5, 10)

    File('malha.pvd') << malha

