#!/usr/bin/env python3
"""
Soccer Ball Sprites Generator

Generates NxN 3D sphere sprite renders with different rotation configurations.
"""

import argparse
import pyvista as pv
import numpy as np
import sys

from pathlib import Path
from PIL import Image

DEFAULT_SIZE = "128x128"
DEFAULT_COUNT = 5
DEFAULT_DETALIZATION = 256

def main(texture_file, count, size, output_prefix="sprite", detalization=DEFAULT_DETALIZATION):
    """Generate NxN sphere sprite renders with texture rotation."""
    print(f"Loading texture: {texture_file}")
    print(f"Generating {count}x{count} sprite map")
    print(f"Size: {size}")
    print(f"Detalization: {detalization}")
    print(f"Output prefix: {output_prefix}")

    # Parse size argument
    if 'x' in size:
        width, height = map(int, size.split('x'))
    else:
        width = int(size)
        height = width

    # Create output directory structure from prefix path
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Load texture
    try:
        tex = pv.read_texture(texture_file)
    except:
        print(f"Error: Could not load texture file '{texture_file}'")
        sys.exit(1)

    # Generate NxN sprite variations with different rotation combinations
    roll_step = 360 / count
    elevation_step = 360 / count

    for i in range(count):
        for j in range(count):
            roll = j * roll_step
            elevation = i * elevation_step

            sprite_name = f"{output_prefix}_{i+1:02d}_{j+1:02d}.png"
            print(f"  Generating: {sprite_name} (roll: {roll:.1f}°, elevation: {elevation:.1f}°)")

            # Create sphere with texture
            sphere = pv.Sphere(
                radius=1,
                theta_resolution=detalization,
                phi_resolution=detalization,
                start_theta=270.001,
                end_theta=270
            )

            # Manual UV coordinates to avoid duplication
            sphere.active_texture_coordinates = np.zeros((sphere.points.shape[0], 2))
            for k in range(sphere.points.shape[0]):
                sphere.active_texture_coordinates[k] = [
                    0.5 + np.arctan2(-sphere.points[k, 0], sphere.points[k, 1]) / (2 * np.pi),
                    0.5 + np.arcsin(sphere.points[k, 2]) / np.pi,
                ]

            # Create plotter with transparent background and aggressive anti-aliasing
            plotter = pv.Plotter(off_screen=True, window_size=(width * 2, height * 2))  # Render at 2x resolution

            # Enable multi-sample anti-aliasing with maximum samples
            plotter.enable_anti_aliasing('msaa', multi_samples=8)

            # Enable SSAA (Super-sampling) for even better quality
            plotter.enable_anti_aliasing('ssaa')

            # Add sphere with texture and smooth shading
            plotter.add_mesh(sphere, texture=tex, smooth_shading=True)

            # Enable depth peeling for better transparency (more layers)
            plotter.enable_depth_peeling(number_of_peels=10)

            # Set camera rotation
            plotter.camera.roll = -roll
            plotter.camera.elevation = -elevation
            plotter.camera.zoom(1.2)  # Slightly zoom in

            # Render at high resolution then downscale for better quality
            temp_name = f"{output_prefix}_{i+1:02d}_{j+1:02d}.temp.png"
            plotter.screenshot(temp_name, transparent_background=True)
            plotter.close()

            # Downscale from 2x to target resolution with high-quality resampling
            img = Image.open(temp_name)
            img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
            img_resized.save(sprite_name)
            Path(temp_name).unlink()

    # Create composite images with all sprites
    print("Creating composite images...")

    # Create big image for all sprites (count x count grid)
    composite_width = width * count
    composite_height = height * count

    # Transparent background composite
    composite_transparent = Image.new('RGBA', (composite_width, composite_height), (0, 0, 0, 0))

    # Green background composite
    composite_green = Image.new('RGB', (composite_width, composite_height), (0, 128, 0))

    # Place each sprite in the grid
    for i in range(count):
        for j in range(count):
            sprite_file = f"{output_prefix}_{i+1:02d}_{j+1:02d}.png"
            sprite_img = Image.open(sprite_file)

            x_pos = j * width
            y_pos = i * height

            # Add to transparent composite
            composite_transparent.paste(sprite_img, (x_pos, y_pos), sprite_img)

            # Add to green composite (convert RGBA to RGB for green background)
            sprite_rgb = Image.new('RGB', sprite_img.size, (0, 128, 0))
            sprite_rgb.paste(sprite_img, mask=sprite_img.split()[-1])  # Use alpha as mask
            composite_green.paste(sprite_rgb, (x_pos, y_pos))

    # Save composite images
    composite_transparent.save(f"{output_prefix}.all.png")
    composite_green.save(f"{output_prefix}.green.png")

    print(f"Created: {output_prefix}.all.png (transparent)")
    print(f"Created: {output_prefix}.green.png (green background)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 3D sphere sprite renders")
    parser.add_argument('texture_file', help='Texture file to apply to sphere')
    parser.add_argument('count', type=int, help='Number of sprites to generate (NxN)')
    parser.add_argument('-s', '--size', default=DEFAULT_SIZE,
                       help=f'Sprite size as WIDTH or WIDTHxHEIGHT (default: {DEFAULT_SIZE})')
    parser.add_argument('-o', '--output-prefix', default="sprite",
                       help='Output filename prefix (default: sprite)')
    parser.add_argument('-d', '--detalization', type=int, default=DEFAULT_DETALIZATION,
                       help=f'Sphere detail level (default: {DEFAULT_DETALIZATION})')
    args = parser.parse_args()

    main(args.texture_file, args.count, args.size, args.output_prefix, args.detalization)
