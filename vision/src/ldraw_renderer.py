"""
LDraw 3D renderer for generating multi-view training images.
Uses software rasterization with numpy for portability.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging

from ldraw_parser import LDrawParser, LDrawPart, Triangle, Quad, POPULAR_PARTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """Configuration for rendering."""
    width: int = 224
    height: int = 224
    background_color: Tuple[int, int, int] = (255, 255, 255)
    ambient_light: float = 0.3
    diffuse_light: float = 0.7
    light_direction: np.ndarray = None

    def __post_init__(self):
        if self.light_direction is None:
            # Default light from upper-left-front
            self.light_direction = np.array([0.5, -0.8, 0.5])
            self.light_direction /= np.linalg.norm(self.light_direction)


class LDrawRenderer:
    """Renders LDraw parts from multiple viewpoints."""

    def __init__(self, ldraw_path: Path, config: RenderConfig = None):
        """
        Initialize renderer.

        Args:
            ldraw_path: Path to LDraw library
            config: Render configuration
        """
        self.parser = LDrawParser(ldraw_path)
        self.config = config or RenderConfig()

    def _rotation_matrix(self, elevation: float, azimuth: float, roll: float = 0) -> np.ndarray:
        """
        Create rotation matrix from elevation and azimuth angles.

        Args:
            elevation: Angle from XZ plane (degrees), positive = looking down
            azimuth: Angle around Y axis (degrees)
            roll: Roll angle (degrees)

        Returns:
            4x4 rotation matrix
        """
        # Convert to radians
        el = np.radians(elevation)
        az = np.radians(azimuth)
        ro = np.radians(roll)

        # Rotation around X (elevation)
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(el), -np.sin(el), 0],
            [0, np.sin(el), np.cos(el), 0],
            [0, 0, 0, 1]
        ])

        # Rotation around Y (azimuth)
        Ry = np.array([
            [np.cos(az), 0, np.sin(az), 0],
            [0, 1, 0, 0],
            [-np.sin(az), 0, np.cos(az), 0],
            [0, 0, 0, 1]
        ])

        # Rotation around Z (roll)
        Rz = np.array([
            [np.cos(ro), -np.sin(ro), 0, 0],
            [np.sin(ro), np.cos(ro), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        return Rz @ Rx @ Ry

    def _project_vertex(self, vertex: np.ndarray, scale: float,
                        offset_x: float, offset_y: float) -> Tuple[int, int]:
        """Project 3D vertex to 2D screen coordinates (orthographic)."""
        # LDraw: Y is up, we want screen Y to increase downward
        x = int(vertex[0] * scale + offset_x)
        y = int(-vertex[1] * scale + offset_y)  # Flip Y
        return x, y

    def _compute_normal(self, vertices: np.ndarray) -> np.ndarray:
        """Compute face normal from vertices."""
        if len(vertices) >= 3:
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                return normal / norm
        return np.array([0, 0, 1])

    def _shade_color(self, base_color: Tuple[int, int, int],
                     normal: np.ndarray) -> Tuple[int, int, int]:
        """Apply simple diffuse shading to a color."""
        light_dir = self.config.light_direction
        dot = max(0, -np.dot(normal, light_dir))

        intensity = self.config.ambient_light + self.config.diffuse_light * dot
        intensity = min(1.0, intensity)

        return tuple(int(c * intensity) for c in base_color)

    def _transform_part(self, part: LDrawPart, transform: np.ndarray) -> Tuple[List, List]:
        """Transform all geometry in a part."""
        transformed_tris = []
        transformed_quads = []

        rot_matrix = transform[:3, :3]

        for tri in part.triangles:
            verts = np.hstack([tri.vertices, np.ones((3, 1))])
            transformed = (transform @ verts.T).T[:, :3]
            normal = self._compute_normal(transformed)
            # Transform normal
            normal = rot_matrix @ normal
            transformed_tris.append((transformed, tri.color, normal))

        for quad in part.quads:
            verts = np.hstack([quad.vertices, np.ones((4, 1))])
            transformed = (transform @ verts.T).T[:, :3]
            normal = self._compute_normal(transformed)
            normal = rot_matrix @ normal
            transformed_quads.append((transformed, quad.color, normal))

        return transformed_tris, transformed_quads

    def render(self, part: LDrawPart, color: int = 4,
               elevation: float = 30, azimuth: float = 45,
               roll: float = 0) -> np.ndarray:
        """
        Render a part from a specific viewpoint.

        Args:
            part: Parsed LDraw part
            color: LDraw color code for the part
            elevation: Camera elevation angle (degrees)
            azimuth: Camera azimuth angle (degrees)
            roll: Camera roll angle (degrees)

        Returns:
            Rendered image as numpy array (BGR)
        """
        w, h = self.config.width, self.config.height

        # Create image with background
        image = np.full((h, w, 3), self.config.background_color, dtype=np.uint8)
        z_buffer = np.full((h, w), -np.inf)

        # Get part bounds for scaling
        min_bounds, max_bounds = self.parser.get_part_bounds(part)
        center = (min_bounds + max_bounds) / 2
        size = max_bounds - min_bounds
        max_dim = max(size) if max(size) > 0 else 1

        # Create view transform
        # First center the part
        center_transform = np.eye(4)
        center_transform[:3, 3] = -center

        # Then rotate
        rotation = self._rotation_matrix(elevation, azimuth, roll)

        # Combine transforms
        transform = rotation @ center_transform

        # Calculate scale to fit in image with margin
        margin = 0.15
        scale = min(w, h) * (1 - margin) / max_dim

        # Offset to center in image
        offset_x = w / 2
        offset_y = h / 2

        # Transform geometry
        transformed_tris, transformed_quads = self._transform_part(part, transform)

        # Collect all faces for depth sorting
        all_faces = []

        for verts, face_color, normal in transformed_tris:
            # Backface culling
            if normal[2] < 0:
                continue
            avg_z = verts[:, 2].mean()
            all_faces.append(('tri', verts, face_color, normal, avg_z))

        for verts, face_color, normal in transformed_quads:
            if normal[2] < 0:
                continue
            avg_z = verts[:, 2].mean()
            all_faces.append(('quad', verts, face_color, normal, avg_z))

        # Sort by depth (painter's algorithm)
        all_faces.sort(key=lambda x: x[4])

        # Render faces
        for face_type, verts, face_color, normal, _ in all_faces:
            # Get actual RGB color
            if face_color == 16:
                face_color = color
            rgb = self.parser.get_color_rgb(face_color, color)
            shaded = self._shade_color(rgb, normal)

            # Convert to BGR for OpenCV
            bgr = (shaded[2], shaded[1], shaded[0])

            # Project vertices
            if face_type == 'tri':
                pts = np.array([
                    self._project_vertex(verts[0], scale, offset_x, offset_y),
                    self._project_vertex(verts[1], scale, offset_x, offset_y),
                    self._project_vertex(verts[2], scale, offset_x, offset_y),
                ], dtype=np.int32)
            else:
                pts = np.array([
                    self._project_vertex(verts[0], scale, offset_x, offset_y),
                    self._project_vertex(verts[1], scale, offset_x, offset_y),
                    self._project_vertex(verts[2], scale, offset_x, offset_y),
                    self._project_vertex(verts[3], scale, offset_x, offset_y),
                ], dtype=np.int32)

            # Draw filled polygon
            cv2.fillPoly(image, [pts], bgr)

        return image

    def render_multiview(self, part: LDrawPart, color: int = 4,
                         num_views: int = 8,
                         elevation_range: Tuple[float, float] = (15, 75),
                         include_random: bool = True) -> List[np.ndarray]:
        """
        Render a part from multiple viewpoints.

        Args:
            part: Parsed LDraw part
            color: LDraw color code
            num_views: Number of views to generate
            elevation_range: Min/max elevation angles
            include_random: Include random viewpoints

        Returns:
            List of rendered images
        """
        images = []

        if include_random:
            # Generate random viewpoints
            for _ in range(num_views):
                elevation = np.random.uniform(*elevation_range)
                azimuth = np.random.uniform(0, 360)
                roll = np.random.uniform(-15, 15)
                img = self.render(part, color, elevation, azimuth, roll)
                images.append(img)
        else:
            # Generate evenly spaced viewpoints
            azimuths = np.linspace(0, 360, num_views, endpoint=False)
            elevation = np.mean(elevation_range)

            for az in azimuths:
                img = self.render(part, color, elevation, az, 0)
                images.append(img)

        return images


def generate_training_images(ldraw_path: Path, output_path: Path,
                             parts: List[str] = None,
                             colors: List[int] = None,
                             views_per_part: int = 16):
    """
    Generate multi-view training images for parts.

    Args:
        ldraw_path: Path to LDraw library
        output_path: Where to save images
        parts: List of part filenames (default: POPULAR_PARTS)
        colors: List of color codes to use
        views_per_part: Number of views per part/color combination
    """
    if parts is None:
        parts = POPULAR_PARTS[:100]  # Top 100 parts

    if colors is None:
        # Common colors: Red, Blue, Yellow, Green, Black, White, Orange, etc.
        colors = [4, 1, 14, 2, 0, 15, 25, 70, 71, 72]

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    renderer = LDrawRenderer(ldraw_path)
    total_images = 0

    for part_name in parts:
        logger.info(f"Processing {part_name}...")

        try:
            part = renderer.parser.parse_part(part_name)

            if len(part.triangles) == 0 and len(part.quads) == 0:
                logger.warning(f"No geometry found for {part_name}")
                continue

            part_id = Path(part_name).stem

            for color in colors:
                # Generate multiview images
                images = renderer.render_multiview(
                    part, color,
                    num_views=views_per_part,
                    elevation_range=(20, 70)
                )

                # Save images
                for i, img in enumerate(images):
                    filename = f"{part_id}_c{color}_v{i:02d}.jpg"
                    filepath = output_path / filename
                    cv2.imwrite(str(filepath), img)
                    total_images += 1

        except Exception as e:
            logger.error(f"Error processing {part_name}: {e}")

    logger.info(f"Generated {total_images} images in {output_path}")


if __name__ == "__main__":
    import sys

    ldraw_path = Path(__file__).resolve().parent.parent / "data" / "ldraw"
    output_path = Path(__file__).resolve().parent.parent / "output" / "ldraw_renders"

    if not ldraw_path.exists():
        print(f"LDraw library not found at {ldraw_path}")
        sys.exit(1)

    # Quick test - render a few views of a 2x4 brick
    print("Testing renderer...")
    renderer = LDrawRenderer(ldraw_path)

    part = renderer.parser.parse_part("3001.dat")
    print(f"Loaded {part.description}: {len(part.triangles)} tris, {len(part.quads)} quads")

    # Render from multiple angles
    output_path.mkdir(parents=True, exist_ok=True)

    colors = [4, 1, 14, 2, 15]  # Red, Blue, Yellow, Green, White
    color_names = ['red', 'blue', 'yellow', 'green', 'white']

    for color, name in zip(colors, color_names):
        for az in [0, 45, 90, 135, 180, 225, 270, 315]:
            for el in [30, 60]:
                img = renderer.render(part, color, elevation=el, azimuth=az)
                filename = f"3001_{name}_el{el}_az{az}.jpg"
                cv2.imwrite(str(output_path / filename), img)

    print(f"Test renders saved to {output_path}")

    # Generate full dataset
    print("\nGenerating training dataset...")
    generate_training_images(
        ldraw_path,
        output_path / "training",
        parts=POPULAR_PARTS[:20],  # Start with top 20 parts
        views_per_part=8
    )
