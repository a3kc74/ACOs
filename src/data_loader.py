"""
DataLoader class for parsing TSPLIB format files and computing distance matrices.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import re


class DataLoader:
    """
    Loads and processes TSPLIB .tsp files.
    
    Supports:
    - EUC_2D (Euclidean distance in 2D)
    - GEO (Geographical coordinates)
    - EXPLICIT (Pre-computed distance matrices)
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the DataLoader with a TSPLIB file.
        
        Args:
            filepath: Path to the .tsp file
        """
        self.filepath = filepath
        self.name: Optional[str] = None
        self.comment: Optional[str] = None
        self.dimension: Optional[int] = None
        self.edge_weight_type: Optional[str] = None
        self.coordinates: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
        
        self._parse_file()
        self._compute_distance_matrix()
    
    def _parse_file(self) -> None:
        """Parse the TSPLIB file and extract metadata and coordinates."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header information
        coord_section_start = -1
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('NAME'):
                self.name = line.split(':', 1)[1].strip()
            elif line.startswith('COMMENT'):
                self.comment = line.split(':', 1)[1].strip()
            elif line.startswith('DIMENSION'):
                self.dimension = int(line.split(':')[1].strip())
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                self.edge_weight_type = line.split(':')[1].strip()
            elif line.startswith('NODE_COORD_SECTION'):
                coord_section_start = i + 1
                break

        if coord_section_start == -1:
            raise ValueError("NODE_COORD_SECTION not found in file")
        
        # Parse coordinates
        coords = []
        for i in range(coord_section_start, len(lines)):
            line = lines[i].strip()
            if line == 'EOF' or line == '':
                break
            
            parts = line.split()
            if len(parts) >= 3:
                # Format: node_id x y
                x, y = float(parts[1]), float(parts[2])
                coords.append([x, y])
        
        self.coordinates = np.array(coords)
        
        if self.dimension is None:
            self.dimension = len(coords)
        
        if len(coords) != self.dimension:
            raise ValueError(
                f"Mismatch: DIMENSION={self.dimension} but found {len(coords)} coordinates"
            )
    
    def _compute_distance_matrix(self) -> None:
        """Compute the distance matrix based on the edge weight type."""
        if self.edge_weight_type == 'EUC_2D':
            self._compute_euclidean_distance()
        elif self.edge_weight_type == 'GEO':
            self._compute_geographical_distance()
        else:
            raise NotImplementedError(
                f"Edge weight type '{self.edge_weight_type}' is not yet supported"
            )
    
    def _compute_euclidean_distance(self) -> None:
        """Compute Euclidean distance matrix for EUC_2D type."""
        n = self.dimension
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean distance rounded to nearest integer (TSPLIB standard)
                dx = self.coordinates[i, 0] - self.coordinates[j, 0]
                dy = self.coordinates[i, 1] - self.coordinates[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                self.distance_matrix[i, j] = int(np.round(dist))
                self.distance_matrix[j, i] = self.distance_matrix[i, j]
    
    def _compute_geographical_distance(self) -> None:
        """Compute geographical distance matrix for GEO type."""
        n = self.dimension
        self.distance_matrix = np.zeros((n, n))
        RRR = 6378.388  # Earth radius in km
        
        # Convert coordinates to radians
        lat_lon = np.zeros_like(self.coordinates)
        for i in range(n):
            # TSPLIB GEO format: degrees with decimal minutes
            x, y = self.coordinates[i]
            deg_x = int(x)
            min_x = x - deg_x
            lat_lon[i, 0] = np.pi * (deg_x + 5.0 * min_x / 3.0) / 180.0
            
            deg_y = int(y)
            min_y = y - deg_y
            lat_lon[i, 1] = np.pi * (deg_y + 5.0 * min_y / 3.0) / 180.0
        
        # Compute distances
        for i in range(n):
            for j in range(i + 1, n):
                q1 = np.cos(lat_lon[i, 1] - lat_lon[j, 1])
                q2 = np.cos(lat_lon[i, 0] - lat_lon[j, 0])
                q3 = np.cos(lat_lon[i, 0] + lat_lon[j, 0])
                dist = RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
                self.distance_matrix[i, j] = int(dist)
                self.distance_matrix[j, i] = self.distance_matrix[i, j]
    
    def get_distance_matrix(self) -> np.ndarray:
        """
        Get the computed distance matrix.
        
        Returns:
            Distance matrix as numpy array
        """
        return self.distance_matrix
    
    def get_coordinates(self) -> np.ndarray:
        """
        Get the node coordinates.
        
        Returns:
            Coordinates array of shape (n, 2)
        """
        return self.coordinates
    
    def get_metadata(self) -> Dict[str, any]:
        """
        Get metadata about the problem instance.
        
        Returns:
            Dictionary containing name, dimension, and edge weight type
        """
        return {
            'name': self.name,
            'comment': self.comment,
            'dimension': self.dimension,
            'edge_weight_type': self.edge_weight_type
        }
    
    def __repr__(self) -> str:
        return (
            f"DataLoader(name='{self.name}', dimension={self.dimension}, "
            f"edge_weight_type='{self.edge_weight_type}')"
        )
