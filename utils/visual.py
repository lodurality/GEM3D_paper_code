import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def get_o3d_graph(points, edges, color=[0,0,1]):
    colors = [color for i in range(len(edges))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def get_o3d_cloud(pts, color=[1,0,0]):
    pts_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if len(color) != len(pts):
        colors = [color for i in range(len(pts))]
    else:
        assert len(color) == len(pts), 'Length of color should be same as length of points'

        if not isinstance(color[0], (int, float)):
            colors = color
        else:
            colors = [color for i in range(len(pts))]

    pts_o3d.colors = o3d.utility.Vector3dVector(colors)
    return pts_o3d


def get_o3d_mesh(trimesh_mesh, color=[1, 0, 0]):
    cur_mesh_o3 = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
                                            triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces))

    cur_mesh_o3.compute_vertex_normals()
    cur_mesh_o3.paint_uniform_color(color)

    return cur_mesh_o3


def draw(*items):
    o3d.visualization.draw_geometries(items)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def get_image(o3d_mesh, rotation=(0, 0, 0), resolution=(256, 256), camera_position=4.2):
    img_width, img_height = resolution
    render = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)

    # Pick a background colour (default is light gray)
    render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA

    # Create the mesh geometry.
    # (We use arrows instead of a sphere, to show the rotation more clearly.)
    mesh = o3d_mesh

    # Create a copy of the above mesh and rotate it around the origin.
    # (If the original mesh is not going to be used, we can just rotate it directly without making a copy.)
    # R = mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    # R = mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
    R = mesh.get_rotation_matrix_from_xyz(rotation)
    # R = mesh.get_rotation_matrix_from_xyz((0, 0, 0))
    mesh_r = deepcopy(mesh)
    mesh_r.rotate(R, center=(0, 0, 0))

    # Show the original coordinate axes for comparison.
    # X is red, Y is green and Z is blue.
    # render.scene.show_axes(True)

    # Define a simple unlit Material.
    # (The base color does not replace the arrows' own colors.)
    mtl = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"
    mtl.shader = "unlitLine"
    mtl.line_width = 0.5  # note that this is sca

    # Add the arrow mesh to the scene.
    # (These are thicker than the main axis arrows, but the same length.)
    render.scene.add_geometry("rotated_model", mesh_r, mtl)

    # Since the arrow material is unlit, it is not necessary to change the scene lighting.
    # render.scene.scene.enable_sun_light(False)
    # render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

    # Optionally set the camera field of view (to zoom in a bit)
    vertical_field_of_view = 15.0  # between 5 and 90 degrees
    aspect_ratio = img_width / img_height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 10.0
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    render.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

    # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
    center = [0, 0, 0]  # look_at target
    eye = [0, 0, camera_position]  # camera position
    up = [0, 1, 0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)

    # Read the image into a variable
    img_o3d = render.render_to_image()
    arr = np.asarray(img_o3d) / 256
    bin_arr = (arr < 0.5)
    bin_arr = bin_arr.mean(axis=-1)

    return bin_arr > 0