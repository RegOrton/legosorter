"""
LDraw file format parser.
Parses .dat files and resolves subfile references to build complete 3D geometry.
"""
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LDrawColor:
    """Represents an LDraw color definition."""
    code: int
    name: str
    value: Tuple[int, int, int]  # RGB
    edge: Tuple[int, int, int]   # Edge RGB
    alpha: int = 255


@dataclass
class Triangle:
    """A triangle with 3 vertices and a color."""
    vertices: np.ndarray  # (3, 3) array
    color: int


@dataclass
class Quad:
    """A quad with 4 vertices and a color."""
    vertices: np.ndarray  # (4, 3) array
    color: int


@dataclass
class LDrawPart:
    """Parsed LDraw part with geometry."""
    name: str
    description: str = ""
    triangles: List[Triangle] = field(default_factory=list)
    quads: List[Quad] = field(default_factory=list)


class LDrawParser:
    """Parser for LDraw .dat files."""

    def __init__(self, ldraw_path: Path):
        """
        Initialize parser with path to LDraw library.

        Args:
            ldraw_path: Path to ldraw directory (containing parts/, p/, LDConfig.ldr)
        """
        self.ldraw_path = Path(ldraw_path)
        self.parts_path = self.ldraw_path / "parts"
        self.primitives_path = self.ldraw_path / "p"
        self.subparts_path = self.parts_path / "s"

        self.colors: Dict[int, LDrawColor] = {}
        self.part_cache: Dict[str, LDrawPart] = {}

        self._load_colors()

    def _load_colors(self):
        """Load color definitions from LDConfig.ldr."""
        config_path = self.ldraw_path / "LDConfig.ldr"
        if not config_path.exists():
            logger.warning(f"LDConfig.ldr not found at {config_path}")
            self._add_default_colors()
            return

        # Pattern: 0 !COLOUR Name CODE n VALUE #RRGGBB EDGE #RRGGBB [ALPHA n]
        color_pattern = re.compile(
            r'0\s+!COLOUR\s+(\S+)\s+CODE\s+(\d+)\s+VALUE\s+#([0-9A-Fa-f]{6})\s+EDGE\s+#([0-9A-Fa-f]{6})(?:\s+ALPHA\s+(\d+))?'
        )

        with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = color_pattern.search(line)
                if match:
                    name = match.group(1)
                    code = int(match.group(2))
                    value_hex = match.group(3)
                    edge_hex = match.group(4)
                    alpha = int(match.group(5)) if match.group(5) else 255

                    value_rgb = tuple(int(value_hex[i:i+2], 16) for i in (0, 2, 4))
                    edge_rgb = tuple(int(edge_hex[i:i+2], 16) for i in (0, 2, 4))

                    self.colors[code] = LDrawColor(
                        code=code,
                        name=name,
                        value=value_rgb,
                        edge=edge_rgb,
                        alpha=alpha
                    )

        logger.info(f"Loaded {len(self.colors)} colors")
        self._add_default_colors()

    def _add_default_colors(self):
        """Add default colors if not defined."""
        defaults = {
            16: LDrawColor(16, "Main_Colour", (127, 127, 127), (0, 0, 0)),
            24: LDrawColor(24, "Edge_Colour", (0, 0, 0), (0, 0, 0)),
        }
        for code, color in defaults.items():
            if code not in self.colors:
                self.colors[code] = color

    def get_color_rgb(self, code: int, parent_color: int = 16) -> Tuple[int, int, int]:
        """Get RGB color for a color code."""
        if code == 16:  # Main color - inherit from parent
            code = parent_color
        if code == 24:  # Edge color
            if parent_color in self.colors:
                return self.colors[parent_color].edge
            return (0, 0, 0)

        if code in self.colors:
            return self.colors[code].value
        return (127, 127, 127)  # Default gray

    def _find_file(self, filename: str) -> Optional[Path]:
        """Find a file in the LDraw library."""
        # Normalize path separators
        filename = filename.replace('\\', '/').lower()

        # Check various locations
        search_paths = [
            self.parts_path / filename,
            self.primitives_path / filename,
            self.subparts_path / filename.replace('s/', ''),
            self.ldraw_path / filename,
        ]

        # Also try without directory prefix
        base_name = Path(filename).name
        search_paths.extend([
            self.parts_path / base_name,
            self.primitives_path / base_name,
            self.subparts_path / base_name,
        ])

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _parse_line(self, line: str) -> Tuple[int, List[str]]:
        """Parse a line and return (line_type, tokens)."""
        line = line.strip()
        if not line:
            return -1, []

        tokens = line.split()
        if not tokens:
            return -1, []

        try:
            line_type = int(tokens[0])
            return line_type, tokens
        except ValueError:
            return -1, []

    def parse_part(self, part_name: str, max_depth: int = 10) -> LDrawPart:
        """
        Parse a part file and resolve all subfile references.

        Args:
            part_name: Part filename (e.g., "3001.dat")
            max_depth: Maximum recursion depth for subfile references

        Returns:
            LDrawPart with all geometry
        """
        # Check cache
        cache_key = part_name.lower()
        if cache_key in self.part_cache:
            return self.part_cache[cache_key]

        part = LDrawPart(name=part_name)

        # Find the file
        file_path = self._find_file(part_name)
        if file_path is None:
            logger.warning(f"Part not found: {part_name}")
            return part

        # Parse the file
        identity_matrix = np.eye(4)
        self._parse_file(file_path, part, identity_matrix, 16, max_depth)

        # Cache the result
        self.part_cache[cache_key] = part

        return part

    def _parse_file(self, file_path: Path, part: LDrawPart,
                    transform: np.ndarray, color: int, depth: int):
        """Parse a file and add geometry to part."""
        if depth <= 0:
            return

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_type, tokens = self._parse_line(line)

                    if line_type == 0:
                        # Comment/meta - extract description from first line
                        if not part.description and len(tokens) > 1:
                            part.description = ' '.join(tokens[1:])

                    elif line_type == 1:
                        # Subfile reference
                        self._handle_subfile(tokens, part, transform, color, depth)

                    elif line_type == 3:
                        # Triangle
                        self._handle_triangle(tokens, part, transform, color)

                    elif line_type == 4:
                        # Quad
                        self._handle_quad(tokens, part, transform, color)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

    def _handle_subfile(self, tokens: List[str], part: LDrawPart,
                        parent_transform: np.ndarray, parent_color: int, depth: int):
        """Handle a subfile reference (line type 1)."""
        if len(tokens) < 15:
            return

        try:
            sub_color = int(tokens[1])
            if sub_color == 16:
                sub_color = parent_color

            # Translation
            x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])

            # Rotation matrix (3x3)
            a, b, c = float(tokens[5]), float(tokens[6]), float(tokens[7])
            d, e, f = float(tokens[8]), float(tokens[9]), float(tokens[10])
            g, h, i = float(tokens[11]), float(tokens[12]), float(tokens[13])

            # Build 4x4 transform matrix
            local_transform = np.array([
                [a, b, c, x],
                [d, e, f, y],
                [g, h, i, z],
                [0, 0, 0, 1]
            ])

            # Combine with parent transform
            combined_transform = parent_transform @ local_transform

            # Get subfile name
            subfile = ' '.join(tokens[14:])  # Handle filenames with spaces

            # Find and parse subfile
            subfile_path = self._find_file(subfile)
            if subfile_path:
                self._parse_file(subfile_path, part, combined_transform, sub_color, depth - 1)

        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing subfile reference: {e}")

    def _handle_triangle(self, tokens: List[str], part: LDrawPart,
                         transform: np.ndarray, parent_color: int):
        """Handle a triangle (line type 3)."""
        if len(tokens) < 11:
            return

        try:
            color = int(tokens[1])
            if color == 16:
                color = parent_color

            # Parse vertices
            v1 = np.array([float(tokens[2]), float(tokens[3]), float(tokens[4]), 1])
            v2 = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7]), 1])
            v3 = np.array([float(tokens[8]), float(tokens[9]), float(tokens[10]), 1])

            # Transform vertices
            v1_t = (transform @ v1)[:3]
            v2_t = (transform @ v2)[:3]
            v3_t = (transform @ v3)[:3]

            vertices = np.array([v1_t, v2_t, v3_t])
            part.triangles.append(Triangle(vertices=vertices, color=color))

        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing triangle: {e}")

    def _handle_quad(self, tokens: List[str], part: LDrawPart,
                     transform: np.ndarray, parent_color: int):
        """Handle a quad (line type 4)."""
        if len(tokens) < 14:
            return

        try:
            color = int(tokens[1])
            if color == 16:
                color = parent_color

            # Parse vertices
            v1 = np.array([float(tokens[2]), float(tokens[3]), float(tokens[4]), 1])
            v2 = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7]), 1])
            v3 = np.array([float(tokens[8]), float(tokens[9]), float(tokens[10]), 1])
            v4 = np.array([float(tokens[11]), float(tokens[12]), float(tokens[13]), 1])

            # Transform vertices
            v1_t = (transform @ v1)[:3]
            v2_t = (transform @ v2)[:3]
            v3_t = (transform @ v3)[:3]
            v4_t = (transform @ v4)[:3]

            vertices = np.array([v1_t, v2_t, v3_t, v4_t])
            part.quads.append(Quad(vertices=vertices, color=color))

        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing quad: {e}")

    def get_part_bounds(self, part: LDrawPart) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of a part."""
        all_vertices = []

        for tri in part.triangles:
            all_vertices.extend(tri.vertices)
        for quad in part.quads:
            all_vertices.extend(quad.vertices)

        if not all_vertices:
            return np.array([0, 0, 0]), np.array([1, 1, 1])

        vertices = np.array(all_vertices)
        return vertices.min(axis=0), vertices.max(axis=0)


# List of common/popular parts to start with
POPULAR_PARTS = [
    "3001.dat",   # Brick 2x4
    "3003.dat",   # Brick 2x2
    "3004.dat",   # Brick 1x2
    "3010.dat",   # Brick 1x4
    "3005.dat",   # Brick 1x1
    "3002.dat",   # Brick 2x3
    "3009.dat",   # Brick 1x6
    "3008.dat",   # Brick 1x8
    "3020.dat",   # Plate 2x4
    "3021.dat",   # Plate 2x3
    "3022.dat",   # Plate 2x2
    "3023.dat",   # Plate 1x2
    "3024.dat",   # Plate 1x1
    "3710.dat",   # Plate 1x4
    "3460.dat",   # Plate 1x8
    "3795.dat",   # Plate 2x6
    "3034.dat",   # Plate 2x8
    "3832.dat",   # Plate 2x10
    "3028.dat",   # Plate 6x12
    "3035.dat",   # Plate 4x8
    "3036.dat",   # Plate 6x8
    "3958.dat",   # Plate 6x6
    "3031.dat",   # Plate 4x4
    "3032.dat",   # Plate 4x6
    "3666.dat",   # Plate 1x6
    "3040.dat",   # Slope 45 2x1
    "3037.dat",   # Slope 45 2x4
    "3038.dat",   # Slope 45 2x3
    "3039.dat",   # Slope 45 2x2
    "3044.dat",   # Slope 45 1x3
    "3665.dat",   # Slope 45 2x1 Inverted
    "3660.dat",   # Slope 45 2x2 Inverted
    "3747.dat",   # Slope 45 1x2 Triple
    "3298.dat",   # Slope 33 3x2
    "3300.dat",   # Slope 33 2x2
    "4286.dat",   # Slope 33 3x1
    "3070b.dat",  # Tile 1x1
    "3069b.dat",  # Tile 1x2
    "3068b.dat",  # Tile 2x2
    "2431.dat",   # Tile 1x4
    "6636.dat",   # Tile 1x6
    "4162.dat",   # Tile 1x8
    "87079.dat",  # Tile 2x4
    "6636.dat",   # Tile 1x6
    "3062b.dat",  # Round Brick 1x1
    "3941.dat",   # Round Brick 2x2
    "6143.dat",   # Round Brick 2x2
    "4073.dat",   # Round Plate 1x1
    "4032.dat",   # Round Plate 2x2
    "3700.dat",   # Technic Brick 1x2
    "3701.dat",   # Technic Brick 1x4
    "3702.dat",   # Technic Brick 1x8
    "32316.dat",  # Technic Beam 1x5
    "32524.dat",  # Technic Beam 1x7
    "3703.dat",   # Technic Brick 1x16
    "3705.dat",   # Technic Axle 4
    "3706.dat",   # Technic Axle 6
    "3707.dat",   # Technic Axle 8
    "3708.dat",   # Technic Axle 12
    "6558.dat",   # Technic Pin Long
    "4274.dat",   # Technic Pin 1/2
    "2780.dat",   # Technic Pin with Friction
    "43093.dat",  # Technic Axle Pin
    "32073.dat",  # Technic Axle 5
    "99773.dat",  # Technic Axle 1
    "32062.dat",  # Technic Axle 2 Notched
    "87580.dat",  # Plate 2x2 with 1 Stud
    "3176.dat",   # Plate 3x2 with Hole
    "3623.dat",   # Plate 1x3
    "2420.dat",   # Plate 2x2 Corner
    "2429.dat",   # Hinge Plate 1x4
    "3045.dat",   # Slope 45 2x2 Double Convex
    "3046.dat",   # Slope 45 2x2 Double Concave
    "4589.dat",   # Cone 1x1
    "3942.dat",   # Cone 2x2x2
    "2654.dat",   # Dish 2x2
    "3960.dat",   # Dish 4x4
    "6141.dat",   # Round Plate 1x1
    "14769.dat",  # Tile Round 2x2
    "15573.dat",  # Plate 1x2 Modified
    "11211.dat",  # Brick 1x2 with 2 Studs on Side
    "87087.dat",  # Brick 1x1 with Stud on Side
    "4070.dat",   # Brick 1x1 with Headlight
    "6091.dat",   # Brick 2x1x1&1/3 with Curved Top
    "98283.dat",  # Brick 1x2 Modified with Masonry
    "15068.dat",  # Slope Curved 2x2x2/3
    "11477.dat",  # Slope Curved 2x1
    "50950.dat",  # Slope Curved 3x1
    "93273.dat",  # Slope Curved 4x1
    "85984.dat",  # Slope 30 1x2x2/3
    "54200.dat",  # Slope 30 1x1x2/3
    "60481.dat",  # Slope 65 2x1x2
    "3245.dat",   # Brick 1x2x2
    "2357.dat",   # Brick 2x2 Corner
]


if __name__ == "__main__":
    # Test the parser
    import sys

    ldraw_path = Path(__file__).resolve().parent.parent / "data" / "ldraw"

    if not ldraw_path.exists():
        print(f"LDraw library not found at {ldraw_path}")
        sys.exit(1)

    parser = LDrawParser(ldraw_path)

    # Parse a test part
    test_part = "3001.dat"
    print(f"\nParsing {test_part}...")
    part = parser.parse_part(test_part)

    print(f"Description: {part.description}")
    print(f"Triangles: {len(part.triangles)}")
    print(f"Quads: {len(part.quads)}")

    min_bounds, max_bounds = parser.get_part_bounds(part)
    print(f"Bounds: {min_bounds} to {max_bounds}")
