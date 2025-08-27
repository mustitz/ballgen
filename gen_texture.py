#!/usr/bin/env python3
"""
Soccer Ball Texture Generator

Generates equirectangular textures of a truncated icosahedron (soccer ball)
with proper great circle interpolation and polar distortion compensation.
"""

import argparse
import math
import numpy as np
from PIL import Image, ImageDraw, ImageColor

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 512
EDGE_THICK = 2
DEFAULT_FILE_NAME = "soccer_ball_texture.png"
DEFAULT_BG_COLOR = "white"
DEFAULT_PENTAGON_COLOR = "black"
INTERPOLATION_POINTS = 1000

DEFAULT_LAT_ROTATION = 15   # degrees - latitude rotation
DEFAULT_LON_ROTATION = 0    # degrees - longitude rotation to avoid lines near poles


# Truncated icosahedron geometry

C0 = (1 + math.sqrt(5)) / 4
C1 = (1 + math.sqrt(5)) / 2
C2 = (5 + math.sqrt(5)) / 4
C3 = (2 + math.sqrt(5)) / 2
C4 = 3 * (1 + math.sqrt(5)) / 4

VERTICES = [
    ( 0.5,  0.0,   C4), ( 0.5,  0.0,  -C4), (-0.5,  0.0,   C4), (-0.5,  0.0,  -C4), (  C4,  0.5,  0.0), (  C4, -0.5,  0.0), ( -C4,  0.5,  0.0),
    ( -C4, -0.5,  0.0), ( 0.0,   C4,  0.5), ( 0.0,   C4, -0.5), ( 0.0,  -C4,  0.5), ( 0.0,  -C4, -0.5), ( 1.0,   C0,   C3), ( 1.0,   C0,  -C3),
    ( 1.0,  -C0,   C3), ( 1.0,  -C0,  -C3), (-1.0,   C0,   C3), (-1.0,   C0,  -C3), (-1.0,  -C0,   C3), (-1.0,  -C0,  -C3), (  C3,  1.0,   C0),
    (  C3,  1.0,  -C0), (  C3, -1.0,   C0), (  C3, -1.0,  -C0), ( -C3,  1.0,   C0), ( -C3,  1.0,  -C0), ( -C3, -1.0,   C0), ( -C3, -1.0,  -C0),
    (  C0,   C3,  1.0), (  C0,   C3, -1.0), (  C0,  -C3,  1.0), (  C0,  -C3, -1.0), ( -C0,   C3,  1.0), ( -C0,   C3, -1.0), ( -C0,  -C3,  1.0),
    ( -C0,  -C3, -1.0), ( 0.5,   C1,   C2), ( 0.5,   C1,  -C2), ( 0.5,  -C1,   C2), ( 0.5,  -C1,  -C2), (-0.5,   C1,   C2), (-0.5,   C1,  -C2),
    (-0.5,  -C1,   C2), (-0.5,  -C1,  -C2), (  C2,  0.5,   C1), (  C2,  0.5,  -C1), (  C2, -0.5,   C1), (  C2, -0.5,  -C1), ( -C2,  0.5,   C1),
    ( -C2,  0.5,  -C1), ( -C2, -0.5,   C1), ( -C2, -0.5,  -C1), (  C1,   C2,  0.5), (  C1,   C2, -0.5), (  C1,  -C2,  0.5), (  C1,  -C2, -0.5),
    ( -C1,   C2,  0.5), ( -C1,   C2, -0.5), ( -C1,  -C2,  0.5), ( -C1,  -C2, -0.5),
]

FACES = [
    [  0,  2, 18, 42, 38, 14 ], [  0,  2, 18, 42, 38, 14 ], [  1,  3, 17, 41, 37, 13 ],
    [  2,  0, 12, 36, 40, 16 ], [  3,  1, 15, 39, 43, 19 ], [  4,  5, 23, 47, 45, 21 ],
    [  5,  4, 20, 44, 46, 22 ], [  6,  7, 26, 50, 48, 24 ], [  7,  6, 25, 49, 51, 27 ],
    [  8,  9, 33, 57, 56, 32 ], [  9,  8, 28, 52, 53, 29 ], [ 10, 11, 31, 55, 54, 30 ],
    [ 11, 10, 34, 58, 59, 35 ], [ 12, 44, 20, 52, 28, 36 ], [ 13, 37, 29, 53, 21, 45 ],
    [ 14, 38, 30, 54, 22, 46 ], [ 15, 47, 23, 55, 31, 39 ], [ 16, 40, 32, 56, 24, 48 ],
    [ 17, 49, 25, 57, 33, 41 ], [ 18, 50, 26, 58, 34, 42 ], [ 19, 43, 35, 59, 27, 51 ],

    [  0, 14, 46, 44, 12 ], [  1, 13, 45, 47, 15 ], [  2, 16, 48, 50, 18 ],
    [  3, 19, 51, 49, 17 ], [  4, 21, 53, 52, 20 ], [  5, 22, 54, 55, 23 ],
    [  6, 24, 56, 57, 25 ], [  7, 27, 59, 58, 26 ], [  8, 32, 40, 36, 28 ],
    [  9, 29, 37, 41, 33 ], [ 10, 30, 38, 42, 34 ], [ 11, 35, 43, 39, 31 ],
]


def xyz_to_lonlat(x, y, z):
    # Normalize to unit sphere
    r = math.sqrt(x*x + y*y + z*z)
    x, y, z = x/r, y/r, z/r

    # Convert to longitude and latitude
    lon = math.atan2(y, x)
    lat = math.asin(z)

    # Convert to degrees
    lon_deg = math.degrees(lon)
    lat_deg = math.degrees(lat)

    return lon_deg, lat_deg


def lonlat_to_xy(lon_deg, lat_deg, width, height):
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    u = int((lon_rad + math.pi) / (2*math.pi) * width)
    v = int((math.pi/2 - lat_rad) / math.pi * height)
    u = u % width
    v = max(0, min(height-1, v))
    return u, v

def main(file_name=None, edge_thickness=EDGE_THICK, width=DEFAULT_WIDTH,
         height=DEFAULT_HEIGHT, bg_color=DEFAULT_BG_COLOR,
         pentagon_color=DEFAULT_PENTAGON_COLOR, edge_color=None,
         interpolation_points=INTERPOLATION_POINTS, lat_rotation=DEFAULT_LAT_ROTATION,
         lon_rotation=DEFAULT_LON_ROTATION):

    # Convert color names to RGB tuples
    bg_rgb = ImageColor.getrgb(bg_color)
    pentagon_rgb = ImageColor.getrgb(pentagon_color)
    edge_rgb = ImageColor.getrgb(edge_color if edge_color else pentagon_color)

    # Create background
    img = Image.new("RGB", (width, height), bg_rgb)
    draw = ImageDraw.Draw(img)

    # Force background fill
    draw.rectangle([(0, 0), (width-1, height-1)], fill=bg_rgb)

    # Convert vertices to longitude/latitude array with 3D rotation
    vertex_lonlat = []
    lat_rot_rad = math.radians(lat_rotation)  # X-axis rotation (latitude)
    lon_rot_rad = math.radians(lon_rotation)  # Y-axis rotation (longitude)

    for x, y, z in VERTICES:
        # Apply rotation matrix around X-axis (latitude rotation)
        x1 = x
        y1 = y * math.cos(lat_rot_rad) - z * math.sin(lat_rot_rad)
        z1 = y * math.sin(lat_rot_rad) + z * math.cos(lat_rot_rad)

        # Apply rotation matrix around Y-axis (longitude rotation)
        x_rot = x1 * math.cos(lon_rot_rad) + z1 * math.sin(lon_rot_rad)
        y_rot = y1
        z_rot = -x1 * math.sin(lon_rot_rad) + z1 * math.cos(lon_rot_rad)

        # Convert rotated coordinates to lon/lat
        lon, lat = xyz_to_lonlat(x_rot, y_rot, z_rot)
        vertex_lonlat.append((lon, lat))

    # Connect vertices using FACES
    for face in FACES:
        face_list = list(face)  # Convert set to list
        face_list.append(face_list[0])  # Close the face

        # Draw lines connecting vertices in the face
        for i in range(len(face_list) - 1):
            start_idx = face_list[i]
            end_idx = face_list[i + 1]

            points = []
            start_lon, start_lat = vertex_lonlat[start_idx]
            end_lon, end_lat = vertex_lonlat[end_idx]

            # Convert to radians
            lat1 = math.radians(start_lat)
            lon1 = math.radians(start_lon)
            lat2 = math.radians(end_lat)
            lon2 = math.radians(end_lon)

            # Calculate the great circle distance using haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

            for j in range(interpolation_points):
                f = j / (interpolation_points - 1)  # Interpolation parameter 0 to 1

                # Spherical linear interpolation (SLERP)
                a = math.sin((1-f) * c) / math.sin(c)
                b = math.sin(f * c) / math.sin(c)

                # Convert to Cartesian coordinates
                x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(lat2) * math.cos(lon2)
                y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(lat2) * math.sin(lon2)
                z = a * math.sin(lat1) + b * math.sin(lat2)

                # Convert back to spherical coordinates
                lat = math.atan2(z, math.sqrt(x*x + y*y))
                lon = math.atan2(y, x)

                # Convert back to degrees
                lat_deg = math.degrees(lat)
                lon_deg = math.degrees(lon)
                points.append((lon_deg, lat_deg))

            for lon, lat in points:
                px, py = lonlat_to_xy(lon, lat, width, height)
                # Latitude-adjusted radius based on latitude (bigger near poles)
                lat_factor = 1.0 / max(0.001, math.cos(math.radians(abs(lat))))  # bigger at poles
                lat_adjusted_radius = int(edge_thickness * lat_factor)

                # Draw circle at each point
                draw.ellipse((px-lat_adjusted_radius, py-edge_thickness, px+lat_adjusted_radius, py+edge_thickness), fill=edge_rgb)

    # Flood fill pentagons with black
    for face in FACES:
        face_list = list(face)

        # HACK: Fixed margin for edge detection - not a solid solution for all pentagon splits
        # Changin rotation may cause flood fill corruption
        # TODO: fix it
        margin = int(0.08 * width)

        if len(face_list) == 5:  # Pentagon
            # Calculate center point handling longitude wrap-around
            u_coords = []
            v_coords = []
            for vertex_idx in face_list:
                lon, lat = vertex_lonlat[vertex_idx]
                u, v = lonlat_to_xy(lon, lat, width, height)
                u_coords.append(u)
                v_coords.append(v)

            # Check for wrap-around in u coordinates (longitude boundary)
            min_u, max_u = min(u_coords), max(u_coords)
            if max_u - min_u > width / 2:
                # Wrap-around detected, adjust coordinates
                adjusted_u = []
                for u in u_coords:
                    if u < width / 2:
                        adjusted_u.append(u + width)  # Move to right side
                    else:
                        adjusted_u.append(u)
                center_u = int(sum(adjusted_u) / 5) % width
            else:
                center_u = int(sum(u_coords) / 5)

            center_v = int(sum(v_coords) / 5)

            # Flood fill from center
            ImageDraw.floodfill(img, (center_u, center_v), pentagon_rgb)

            # If center is near left edge, also flood fill from corresponding right edge
            if center_u < margin:
                mirror_u = width - 1
                ImageDraw.floodfill(img, (mirror_u, center_v), pentagon_rgb)
            # If center is near right edge, also flood fill from corresponding left edge
            elif center_u > width - margin:
                mirror_u = 1
                ImageDraw.floodfill(img, (mirror_u, center_v), pentagon_rgb)

    if file_name is None:
        return img

    # Save the image
    img.save(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate soccer ball texture")
    parser.add_argument('-o', '--output', default=DEFAULT_FILE_NAME,
                       help=f'Output filename (default: {DEFAULT_FILE_NAME})')
    parser.add_argument('-t', '--thickness', type=int, default=EDGE_THICK,
                       help=f'Edge thickness in pixels (default: {EDGE_THICK})')
    parser.add_argument('-s', '--size', default=f'{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}',
                       help=f'Texture size as WIDTH or WIDTHxHEIGHT (default: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT})')
    parser.add_argument('--bg-color', default=DEFAULT_BG_COLOR,
                       help=f'Background color (default: {DEFAULT_BG_COLOR})')
    parser.add_argument('--pentagon-color', default=DEFAULT_PENTAGON_COLOR,
                       help=f'Pentagon color (default: {DEFAULT_PENTAGON_COLOR})')
    parser.add_argument('--edge-color', default=None,
                       help='Edge color (default: same as pentagon color)')
    parser.add_argument('-i', '--interpolation', type=int, default=INTERPOLATION_POINTS,
                       help=f'Interpolation points for smooth edges (default: {INTERPOLATION_POINTS})')
    parser.add_argument('--lat-rotation', type=int, default=DEFAULT_LAT_ROTATION,
                       help=f'Latitude rotation in degrees (default: {DEFAULT_LAT_ROTATION})')
    parser.add_argument('--lon-rotation', type=int, default=DEFAULT_LON_ROTATION,
                       help=f'Longitude rotation in degrees (default: {DEFAULT_LON_ROTATION})')
    args = parser.parse_args()

    # Parse size argument
    if 'x' in args.size:
        width, height = map(int, args.size.split('x'))
    else:
        width = int(args.size)
        height = width // 2

    main(args.output, args.thickness, width, height, args.bg_color, args.pentagon_color, args.edge_color, args.interpolation, args.lat_rotation, args.lon_rotation)
