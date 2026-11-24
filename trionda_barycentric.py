import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.path import Path

def chiral_edge(p0, p1, n_pts=80, amp=0.22, harmonic=0.04):
    """
    Deforms a straight edge (p0 -> p1) into a C2-symmetric S-curve.
    Any odd function of (t - 0.5) guarantees exact interlocking when rotated 180 deg.
    """
    t = np.linspace(0.0, 1.0, n_pts)
    vec = p1 - p0
    length = np.linalg.norm(vec)
    tangent = vec / length
    normal = np.array([-tangent[1], tangent[0]])

    # Antisymmetric wave around t = 0.5 (zero at endpoints and midpoint)
    offset = length * (amp * np.sin(2 * np.pi * t) + harmonic * np.sin(4 * np.pi * t))
    return p0 + np.outer(t, vec) + np.outer(offset, normal)

def build_flat_panel(amp=0.22, harmonic=0.04, n_edge=80, grid_res=75):
    """
    Generates a 2D chiral boomerang panel and triangulates its interior.
    """
    # Base equilateral triangle vertices (unit circumradius)
    angles = np.deg2rad([90, 210, 330])
    V = np.column_stack([np.cos(angles), np.sin(angles)])

    # Construct closed boundary from 3 identically deformed edges
    edges = [
        chiral_edge(V[i], V[(i + 1) % 3], n_pts=n_edge, amp=amp, harmonic=harmonic)[:-1]
        for i in range(3)
    ]
    boundary = np.vstack(edges)
    poly_path = Path(boundary)

    # Sample bounding box + boundary points for a clean interior mesh
    pad = 0.35
    xs = np.linspace(boundary[:, 0].min() - pad, boundary[:, 0].max() + pad, grid_res)
    ys = np.linspace(boundary[:, 1].min() - pad, boundary[:, 1].max() + pad, grid_res)
    gx, gy = np.meshgrid(xs, ys)
    candidates = np.column_stack([gx.ravel(), gy.ravel()])

    inside = poly_path.contains_points(candidates)
    pts_2d = np.vstack([boundary, candidates[inside]])

    # Delaunay triangulation pruned to triangles whose centroids lie inside the boomerang
    tri = Delaunay(pts_2d)
    centroids = pts_2d[tri.simplices].mean(axis=1)
    valid_simplices = tri.simplices[poly_path.contains_points(centroids)]

    return V, boundary, pts_2d, valid_simplices

def barycentric_weights(pts, V):
    """
    Computes barycentric coordinates (w0, w1, w2) of 2D points relative to triangle V.
    """
    T = np.column_stack([V[0] - V[2], V[1] - V[2]])
    inv_T = np.linalg.inv(T)
    w01 = (pts - V[2]) @ inv_T.T
    w2 = 1.0 - w01[:, 0] - w01[:, 1]
    return np.column_stack([w01, w2])

def fold_to_sphere(weights, tet_verts):
    """
    Maps barycentric weights onto a 3D tetrahedral face and projects radially onto S^2.
    """
    pts_3d = weights @ tet_verts
    norms = np.linalg.norm(pts_3d, axis=1, keepdims=True)
    return pts_3d / norms

def main():
    # Regular tetrahedron vertices on the unit sphere
    tet_vertices = np.array([
        [ 1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
    ]) / np.sqrt(3.0)

    # The 4 oriented faces of the tetrahedron (consistent outward winding)
    tet_faces = [
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ]

    # Build the master 2D panel
    V_2d, boundary_2d, pts_2d, simplices = build_flat_panel(amp=0.21, harmonic=0.035)
    w_pts = barycentric_weights(pts_2d, V_2d)
    w_bnd = barycentric_weights(boundary_2d, V_2d)

    # --- Visualization ---
    fig = plt.figure(figsize=(13, 6), facecolor="#111317")

    # Left: 2D Flat Boomerang Panel + Reference Tetrahedron Triangle
    ax2d = fig.add_subplot(121, facecolor="#111317")
    ax2d.fill(boundary_2d[:, 0], boundary_2d[:, 1], color="#2a9d8f", alpha=0.85, label="Chiral Boomerang Panel")
    ax2d.plot(np.append(boundary_2d[:, 0], boundary_2d[0, 0]),
              np.append(boundary_2d[:, 1], boundary_2d[0, 1]),
              color="#ffffff", lw=2.0)

    # Overlay dashed reference equilateral triangle
    ref_tri = np.vstack([V_2d, V_2d[0]])
    ax2d.plot(ref_tri[:, 0], ref_tri[:, 1], "--", color="#f4a261", lw=1.5, label="Base Tetrahedral Face")
    ax2d.scatter(V_2d[:, 0], V_2d[:, 1], color="#e76f51", zorder=5, s=45)

    ax2d.set_aspect("equal")
    ax2d.set_title("Flat 2D Die-Cut Panel (Chiral $C_3$ Symmetry)", color="white", fontsize=12, pad=12)
    ax2d.legend(loc="lower right", frameon=True, facecolor="#1c1f26", edgecolor="none", labelcolor="white")
    ax2d.axis("off")

    # Right: 3D Folded Sphere (4 Interlocking Panels)
    ax3d = fig.add_subplot(122, projection="3d", facecolor="#111317")
    palette = ["#2a9d8f", "#e9c46a", "#e76f51", "#4ea8de"]

    for idx, face in enumerate(tet_faces):
        face_verts = tet_vertices[face]
        pts_3d = fold_to_sphere(w_pts, face_verts)
        bnd_3d = fold_to_sphere(w_bnd, face_verts)
        bnd_closed = np.vstack([bnd_3d, bnd_3d[0]])

        ax3d.plot_trisurf(
            pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
            triangles=simplices,
            color=palette[idx],
            edgecolor="none",
            alpha=0.95,
            shade=True
        )
        # Render the deep seam groove
        ax3d.plot(bnd_closed[:, 0], bnd_closed[:, 1], bnd_closed[:, 2],
                  color="#111317", lw=2.2, zorder=10)

    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_xlim([-0.75, 0.75])
    ax3d.set_ylim([-0.75, 0.75])
    ax3d.set_zlim([-0.75, 0.75])
    ax3d.view_init(elev=22, azim=38)
    ax3d.set_title("Folded 3D Sphere (12 Rotational Symmetries)", color="white", fontsize=12, pad=12)
    ax3d.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
