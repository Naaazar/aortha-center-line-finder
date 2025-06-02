import pyvista as pv
import numpy as np
from skimage.morphology import skeletonize
import vtk
from scipy.spatial import cKDTree
import networkx as nx
from scipy.interpolate import splprep, splev

# === 1. Wczytaj model aorty (.vtp)
mesh = pv.read(r"your path here")

# === 2. Ustaw rozdzielczość voxelizacji
voxel_size = 0.1  # mm
bounds = mesh.bounds
x_dim = int((bounds[1] - bounds[0]) / voxel_size)
y_dim = int((bounds[3] - bounds[2]) / voxel_size)
z_dim = int((bounds[5] - bounds[4]) / voxel_size)

# === 3. Stwórz funkcję odległości od powierzchni
implicit_distance = vtk.vtkImplicitPolyDataDistance()
implicit_distance.SetInput(mesh)

# === 4. Stwórz pustą siatkę voxelową
image = vtk.vtkImageData()
image.SetDimensions(x_dim, y_dim, z_dim)
image.SetSpacing(voxel_size, voxel_size, voxel_size)
image.SetOrigin(bounds[0], bounds[2], bounds[4])
image.AllocateScalars(vtk.VTK_FLOAT, 1)

# === 5. Wypełnij voxel wartościami odległości
for z in range(z_dim):
    for y in range(y_dim):
        for x in range(x_dim):
            pt = [
                bounds[0] + x * voxel_size,
                bounds[2] + y * voxel_size,
                bounds[4] + z * voxel_size
            ]
            d = implicit_distance.EvaluateFunction(pt)
            image.SetScalarComponentFromFloat(x, y, z, 0, d)

# === 6. Konwersja do PyVista UniformGrid
volume = pv.wrap(image)

# === 7. Binaryzacja i skeletonizacja
mask = volume.point_data["ImageScalars"] < 0
binary_volume = mask.reshape(volume.dimensions[::-1])
skeleton = skeletonize(binary_volume)

# === 8. Współrzędne skeletonu
svox = np.argwhere(skeleton)
coords_xyz = svox[:, ::-1]
spacing = np.array(volume.spacing)
origin = np.array(volume.origin)
coords_mm = coords_xyz * spacing + origin

# === 9. Odległość od ściany
dist_calc = vtk.vtkImplicitPolyDataDistance()
dist_calc.SetInput(mesh)
distances = [abs(dist_calc.EvaluateFunction(pt)) for pt in coords_mm]

# === 10. PolyLine z punktów
def points_to_polyline(points):
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt)

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(len(points))
    for i in range(len(points)):
        polyline.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(cells)

    return pv.wrap(polydata)

# === 11. Graf
def build_graph(points, distances, radius=voxel_size*2.0):
    tree = cKDTree(points)
    G = nx.Graph()
    for i, pt in enumerate(points):
        G.add_node(i, pos=pt, distance=distances[i])
    pairs = tree.query_pairs(radius)
    for i, j in pairs:
        dist = np.linalg.norm(points[i] - points[j])
        G.add_edge(i, j, weight=dist)
    extended_pairs = tree.query_pairs(radius * 1.5)
    for i, j in extended_pairs:
        if not G.has_edge(i, j) and abs(distances[i] - distances[j]) < max(distances[i], distances[j]) * 0.5:
            dist = np.linalg.norm(points[i] - points[j])
            G.add_edge(i, j, weight=dist)
    return G

# === 12–16. Centerline i scoring
def evaluate_path_quality(G, start_node, visited, lookahead):
    current = start_node
    local_visited = visited.copy()
    local_visited.add(current)
    distances_sum = G.nodes[current]['distance']
    count = 1
    for _ in range(lookahead - 1):
        neighbors = [n for n in G.neighbors(current) if n not in local_visited]
        if not neighbors:
            break
        next_node = max(neighbors, key=lambda x: G.nodes[x]['distance'])
        distances_sum += G.nodes[next_node]['distance']
        count += 1
        local_visited.add(next_node)
        current = next_node
    return distances_sum / count if count > 0 else 0

def find_smart_centerline(G, lookahead=5):
    best_start_node = min(G.nodes, key=lambda n: G.nodes[n]['pos'][2])
    path = [best_start_node]
    visited = {best_start_node}
    current = best_start_node
    forward_vec = None

    while True:
        neighbors = [n for n in G.neighbors(current) if n not in visited]
        if not neighbors:
            break
        best_score = -np.inf
        best_candidate = None
        for candidate in neighbors:
            candidate_vec = G.nodes[candidate]['pos'] - G.nodes[current]['pos']
            candidate_norm = np.linalg.norm(candidate_vec)
            if candidate_norm == 0:
                continue
            cos_angle = 1.0 if forward_vec is None else np.dot(forward_vec, candidate_vec) / (np.linalg.norm(forward_vec) * candidate_norm)
            if forward_vec is not None and cos_angle < 0.3:
                continue
            quality = evaluate_path_quality(G, candidate, visited, lookahead)
            wall_bonus = G.nodes[candidate]['distance']
            score = quality + 0.5 * wall_bonus + 0.3 * cos_angle
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_vec = candidate_vec
        if best_candidate is not None:
            path.append(best_candidate)
            visited.add(best_candidate)
            current = best_candidate
            forward_vec = best_vec
        else:
            break
    return path

# === 17. Analiza gałęzi
def trace_branch(G, start_node, main_path_set, max_length=50):
    path = [start_node]
    visited = main_path_set.copy()
    visited.add(start_node)
    current = start_node
    for _ in range(max_length):
        neighbors = [n for n in G.neighbors(current) if n not in visited]
        if not neighbors:
            break
        next_node = max(neighbors, key=lambda x: G.nodes[x]['distance'])
        path.append(next_node)
        visited.add(next_node)
        current = next_node
        if G.degree(current) == 1 or (G.degree(current) > 2 and len(path) > 5):
            break
    return path

def classify_branch(G, branch_path):
    diameters = [G.nodes[node]['distance'] * 2 for node in branch_path]
    avg_diameter = np.mean(diameters)
    max_diameter = np.max(diameters)
    length = len(branch_path)
    if avg_diameter > 4.0 and length > 10:
        return "Major Artery"
    elif avg_diameter > 2.0 and length > 5:
        return "Secondary Artery"
    elif avg_diameter > 1.0:
        return "Small Vessel"
    else:
        return "Capillary"

def analyze_branches(G, main_path):
    main_path_set = set(main_path)
    branches = []
    for node in main_path:
        neighbors = list(G.neighbors(node))
        branch_neighbors = [n for n in neighbors if n not in main_path_set]
        for branch_start in branch_neighbors:
            branch_path = trace_branch(G, branch_start, main_path_set)
            if len(branch_path) >= 3:
                classification = classify_branch(G, branch_path)
                branches.append({
                    'start_node': node,
                    'branch_start': branch_start,
                    'path': branch_path,
                    'length': len(branch_path),
                    'avg_diameter': np.mean([G.nodes[n]['distance'] * 2 for n in branch_path]),
                    'classification': classification
                })
    return branches

# === 18. Główne przetwarzanie
G = build_graph(coords_mm, distances)
largest_cc = max(nx.connected_components(G), key=len)
G_sub = G.subgraph(largest_cc).copy()
print("Szukanie centerline z ulepszoną detekcją gałęzi arteryjnych...")
path_indices = find_smart_centerline(G_sub)
ordered_coords = coords_mm[list(path_indices)]

# === Wygładzenie linii za pomocą B-spline
def smooth_path(points, smoothing=0.001, num_points=500):
    tck, _ = splprep(points.T, s=smoothing)
    u_fine = np.linspace(0, 1, num_points)
    smoothed = splev(u_fine, tck)
    return np.vstack(smoothed).T

smoothed_coords = smooth_path(ordered_coords)

# === 19. Wizualizacja
centerline_polyline = points_to_polyline(smoothed_coords)
skeleton_cloud = pv.PolyData(coords_mm)
skeleton_cloud["distance_to_wall"] = distances

plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightblue", opacity=0.2, label="Aorta")
plotter.add_mesh(skeleton_cloud, scalars="distance_to_wall", cmap="plasma", point_size=2, render_points_as_spheres=True, label="Skeleton")
plotter.add_mesh(centerline_polyline, color="darkred", line_width=6, label="Smoothed Centerline")

plotter.add_title("Smoothed Aortic Centerline")
plotter.add_legend()
plotter.show()
