from dolfin import *
from mshr import *
p1 = Point(0,0)
p2 = Point(2,1)
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, 'quadrilateral', 2,2)
editor.init_vertices(4)
editor.init_cells(1)
editor.add_vertex(0, Point(0,0))
editor.add_vertex(1, Point(2,0))
editor.add_vertex(2, Point(0,1))
editor.add_vertex(3, Point(2,1))
editor.add_cell(0,[0,1,2,3])
editor.close(order=False)

File("apagar.pvd") << mesh

