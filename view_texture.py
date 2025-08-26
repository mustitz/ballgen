#!/usr/bin/env python3
"""
Soccer Ball Texture Viewer

Displays equirectangular textures on a 3D sphere using PyVista
with proper UV mapping to avoid texture duplication.
"""

import argparse
import pyvista as pv
import numpy as np
import sys

DEFAULT_DETALIZATION = 256

def main(texture_file, detalization=DEFAULT_DETALIZATION):
    print(f"Loading texture: {texture_file}")
    print(f"Sphere detalization: {detalization}")

    # Create sphere with slight overlap to avoid seams
    sphere = pv.Sphere(
        radius=1,
        theta_resolution=detalization,
        phi_resolution=detalization,
        start_theta=270.001,  # Slight overlap to avoid seams
        end_theta=270
    )

    # Manually calculate UV coordinates to avoid duplication
    sphere.active_texture_coordinates = np.zeros((sphere.points.shape[0], 2))

    for i in range(sphere.points.shape[0]):
        sphere.active_texture_coordinates[i] = [
            0.5 + np.arctan2(-sphere.points[i, 0], sphere.points[i, 1]) / (2 * np.pi),
            0.5 + np.arcsin(sphere.points[i, 2]) / np.pi,
        ]

    # Load texture
    try:
        tex = pv.read_texture(texture_file)
    except:
        print(f"Error: Could not load texture file '{texture_file}'")
        sys.exit(1)

    # Create plotter
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, texture=tex)
    plotter.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="View soccer ball texture on 3D sphere")
    parser.add_argument('texture_file', help='Texture file to display')
    parser.add_argument('-d', '--detalization', type=int, default=DEFAULT_DETALIZATION,
                       help=f'Sphere detail level (default: {DEFAULT_DETALIZATION})')
    args = parser.parse_args()

    main(args.texture_file, args.detalization)
