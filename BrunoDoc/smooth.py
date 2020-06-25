from dolfin import *
from dolfin_adjoint import *
from mshr import *
import numpy as np
import os

from polylidar import extractPlanesAndPolygons, extractPolygons, Delaunator
import matplotlib.pyplot as plt
from polylidarutil import (generate_test_points, plot_points, plot_triangles, get_estimated_lmax, plot_triangle_meshes, get_triangles_from_he, get_plane_triangles, plot_polygons)
# import shapely.geometry as shape
from shapely import geometry

n = 0
m = 0
N = 30
lmax = 0.30 #0.10

def generate_point_list(rho):
    mesh = rho.function_space().mesh()
    point_cloud = []
    for celula in cells(mesh):
        xy = celula.get_vertex_coordinates()
        xg = (xy[0] + xy[2] + xy[4])/3.
        yg = (xy[1] + xy[3] + xy[5])/3.
        if rho(xg,yg) < 0.5:
            point_cloud.append([xg, yg])
    return np.array(point_cloud)

def get_point(p_index, points):
    if points.shape[1] > 2:
        return [points[p_index, 0], points[p_index, 1], points[p_index, 2]]
    else:
        return [points[p_index, 0], points[p_index, 1]]

def generate_polygon(rho, geo='2D', accept_holes=False):
    point_cloud = generate_point_list(rho)
    if geo == '2D':
        # delaunay, planes, polygons = extractPlanesAndPolygons(point_cloud, xyThresh=0.0, alpha=0.0, lmax=0.05, minTriangles=5) # 15
        try:
            delaunay, planes, polygons = extractPlanesAndPolygons(point_cloud, xyThresh=0.0, alpha=0.0, lmax=lmax, minTriangles=5)
        except RuntimeError:
            return None, None

        fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
        plot_points(point_cloud, ax)
        plot_polygons(polygons, delaunay, point_cloud, ax)
        shell_coords = []
        hole_coords = []
        for poly in polygons:
            # shell_coords = [get_point(p_index, point_cloud) for p_index in poly.shell]
            shell_coords.append([get_point(p_index, point_cloud) for p_index in poly.shell])
            try:
                poly.holes[0]
            except:
                hole_coords.append([])
                accept_holes=False
            if accept_holes:
                hole_coords.append([get_point(p_index, point_cloud) for p_index in poly.holes[0]])
        # shape = shape.Polygon([ [item[0], item[1]] for item in shell_coords])
        # shell_coords = shell_coords[0::2] # FIXME: Talvez necessario

        shapes = []
        num = 0
        rotating = None
        for shell in shell_coords:
            shell.reverse()
            polig = []
            for item in shell:
                polig.append(Point(item[0], item[1]))
                if item[1] < 40.1: rotating = num
            num += 1
            shapes.append(Polygon(polig))

        # hole_coords = hole_coords[0::2]
        i = 0
        for hole in hole_coords:
            if not hole == []:
                shape_hole = Polygon([Point(item[0], item[1]) for item in hole])
                shapes[i] = shapes[i] - shape_hole
            i += 1

        new_geo = rho.full_geo
        '''for shape in shapes:
            new_geo = new_geo - shape'''
        new_mesh = generate_mesh(new_geo, N)
        rho.new_mesh = new_mesh

        if rotating is not None:
            rotating_countour = np.array([[item.x(), item.y()] for item in shapes[rotating].vertices()])

    else:
        raise NotImplementedError()

    class BorderRotating(SubDomain):
        def inside(self, x, on_boundary):
            global n
            global m
            line = geometry.LineString(rotating_countour)
            point = geometry.Point(x[0], x[1])
            # if line.contains(point):
            if point.distance(line) <1e-2 or x[1]<=40.00001:
                n += 1
                return True and on_boundary
                # return True and on_boundary
            else:
                m += 1
                return False and on_boundary
                # return False and on_boundary
    class BorderFixed(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1]>40.000001
    print()
    print("n = {n}, m = {m}".format(n=n,m=m))
    domain = MeshFunction("size_t", new_mesh, 1)
    domain.set_all(0)
    edge2 = BorderFixed()
    edge2.mark(domain, 2)
    if rotating is not None:
        edge1 = BorderRotating()
        edge1.mark(domain, 1)

    return new_mesh, domain

def generate_polygon_refined(rho, geo="2D", accept_holes=False):
    class Border(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    new_mesh = generate_polygon(rho, geo, accept_holes)
    regions = MeshFunction("bool", new_mesh, False)
    regions.set_all(0)
    region_to_refine = Border()
    region_to_refine.mark(regions, True)

    new_mesh_refined = refine(new_mesh, regions)

    domain = MeshFunction("size_t", new_mesh, 1)
    domain.set_all(0)
    edge = Border()
    edge.mark(domain, 1)

    return new_mesh_refined, domain

