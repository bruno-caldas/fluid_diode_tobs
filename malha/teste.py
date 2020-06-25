from dolfin import *
from mshr import *
p1 = Point(0,0)
p2 = Point(2,1)
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, 'quadrilateral', 2,2)

editor.init_vertices(6)
editor.init_cells(2)

editor.add_vertex(0, Point(0,0))
editor.add_vertex(1, Point(1,0))
editor.add_vertex(2, Point(0,1))
editor.add_vertex(3, Point(1,1))

editor.add_vertex(4, Point(2,0))
editor.add_vertex(5, Point(2,1))
#editor.add_vertex(6, Point(1,1))
#editor.add_vertex(7, Point(2,1))


# editor.add_vertex(8, Point(0,1))
# editor.add_vertex(9, Point(1,1))
# editor.add_vertex(10, Point(0,2))
# editor.add_vertex(11, Point(1,2))

#editor.add_vertex(8, Point(1,1))
#editor.add_vertex(9, Point(2,1))
#editor.add_vertex(10, Point(1,2))
#editor.add_vertex(11, Point(2,2))

editor.add_cell(0,[0,1,2,3])
editor.add_cell(1,[1,4,3,5])
# editor.add_cell(2,[8,9,10,11])

editor.close()

File("apagar.pvd") << mesh

A = FunctionSpace(mesh, "DG", 0)

