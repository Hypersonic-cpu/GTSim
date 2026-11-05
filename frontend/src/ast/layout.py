"""Physical layout for a sequence of independently tiled matrices."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TileMajorLayout:
    tile_k: int
    tile_n: int
    matrix_k: int

    def validate(self, shape):
        rows, cols = shape
        if (
            min(self.tile_k, self.tile_n, self.matrix_k) <= 0
            or self.matrix_k % self.tile_k
            or rows % self.matrix_k
            or cols % self.tile_n
        ):
            raise ValueError(f"tile-major shape {shape} incompatible with {self}")

    def pages(self, width, row, col, height, columns):
        """Physical memory tile indices intersecting a logical rectangle."""
        k_tiles = self.matrix_k // self.tile_k
        tiles_per_matrix = k_tiles * (width // self.tile_n)
        for tk in range(row // self.tile_k, (row + height - 1) // self.tile_k + 1):
            matrix, local_k = divmod(tk, k_tiles)
            for tn in range(col // self.tile_n, (col + columns - 1) // self.tile_n + 1):
                yield matrix * tiles_per_matrix + tn * k_tiles + local_k
