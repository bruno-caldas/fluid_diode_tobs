from dolfin import *
from mshr import *
import numpy as np

def poprow(my_array,pr):
    """ row popping in numpy arrays
    Input: my_array - NumPy array, pr: row index to pop out
    Output: [new_array,popped_row] """
    i = pr
    pop = my_array[i]
    new_array = np.vstack((my_array[:i],my_array[i+1:]))
    return [new_array,pop]

def generate_quad_mesh(N, radius=0, delta=2):
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 'quadrilateral', 2,2)

    ptos_x = np.linspace(-1, delta+1, int((delta+2)*2*N+1))
    ptos_y = np.linspace(0+radius, 1.5+radius, 3*N + 1)
    out_1x, out_1y = np.meshgrid(
                        np.linspace(-1, 0, 2*N, endpoint=False),
                        np.linspace(1.5+radius, 0.5+radius, 2*N, endpoint=False)
                        )
    out_2x, out_2y = np.meshgrid(
                        np.linspace(delta+1, delta, 2*N, endpoint=False),
                        np.linspace(1.5+radius, 0.5+radius, 2*N, endpoint=False)
                        )
    out_1 = np.round(np.array([out_1x.flatten(), out_1y.flatten()]).T, 10)
    out_2 = np.round(np.array([out_2x.flatten(), out_2y.flatten()]).T, 10)

    pontos = []
    for py in ptos_y:
        for px in ptos_x:
            pontos.append([px, py])
    pontos = np.round(np.array(pontos), 10)

    for elem in out_1:
        point_out = np.where((pontos == elem).all(axis=1))[0][0]
        pontos = poprow(pontos, point_out)[0]
    for elem in out_2:
        point_out = np.where((pontos == elem).all(axis=1))[0][0]
        pontos = poprow(pontos, point_out)[0]

    nvertices = len(pontos)
    ncells = int(delta+2)*2*N * 3*N - len(out_1x)*len(out_1y) - len(out_2x)*len(out_2y)
    editor.init_vertices(nvertices)
    editor.init_cells(ncells)

    for i in range(nvertices):
        print("vertice {}, ponto ({},{})".format(i, pontos[i][0], pontos[i][1]))
        editor.add_vertex(i, pontos[i])

    index = 0
    cell_size = round(0.5/N, 10)
    for i in pontos:
        elem= np.where((np.round(pontos,9) == np.round((i[0], i[1]), 9)).all(axis=1))[0]
        elem_right = np.where((np.round(pontos,9) == np.round((i[0] + cell_size, i[1]), 9)).all(axis=1))[0]
        elem_above = np.where((np.round(pontos,9) == np.round((i[0], i[1] + cell_size), 9)).all(axis=1))[0]
        elem_diag_sup = np.where((np.round(pontos,9) == np.round((i[0] + cell_size, i[1] + cell_size), 9)).all(axis=1))[0]
        if 0 in [elem.size, elem_right.size, elem_above.size, elem_diag_sup.size]:
            continue
        else:
            elem = elem[0]
            elem_right = elem_right[0]
            elem_above = elem_above[0]
            elem_diag_sup = elem_diag_sup[0]
        if i[1] == 3.0: continue
        editor.add_cell(index, [
                                elem,
                                elem_right,
                                elem_above,
                                elem_diag_sup
                                ])
        print("celula {}, v ({},{},{},{})".format(index,
                                                    elem,
                                                    elem_right,
                                                    elem_above,
                                                    elem_diag_sup))
        index += 1

    editor.close()
    return mesh

if __name__ == '__main__':
    N = 5
    mesh = generate_quad_mesh(N)
    File("apagar.pvd") << mesh

    A = FunctionSpace(mesh, "DG", 0)

