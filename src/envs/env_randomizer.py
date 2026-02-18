import math
import numpy as np
import matplotlib.pyplot as plt

class PoissonDisc:
    """
    Method used for generating points where no two points are within an
    r distance of eachother

    Implemented from this paper:
    https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
    """
    def __init__(self, grid_length, grid_width, min_radius, max_sample_size):
        """Initialization with initial state"""

        self.cell_size = min_radius / math.sqrt(2)
        self.min_radius = min_radius
        self.sample_size = max_sample_size
        self.grid_width = grid_width
        self.grid_length = grid_length

        self.grid_size_x = int(grid_length / self.cell_size) + 1 
        self.grid_size_y = int(grid_width / self.cell_size) + 1
        self.grid = [[None] * self.grid_size_y for _ in range(self.grid_size_x)]
        self.active_list = []
        self.samples = []
        
        x0 = np.array([np.random.uniform(0, grid_length), np.random.uniform(0, grid_width)])
        gx, gy = self._convert_point_grid(x0)
        self.grid[gx][gy] = x0
        self.active_list.append(x0)
        self.samples.append(x0)

    def _convert_point_grid(self, point: [float, float]) -> tuple[int, int]:
        """Converts point indexes for grid"""

        x = int(point[0] / self.cell_size)
        y = int(point[1] / self.cell_size)
        return x, y
    
    def _check_valid_grid(self, x, y) -> bool:
        """Checks if x and y are within the grid size"""

        return 0 <= x < self.grid_size_x and 0 <= y < self.grid_size_y
    
    def _check_valid_point(self, point) -> bool:
        """Checks point to ensure it is within normal grid"""

        return 0 <= point[0] < self.grid_length and 0 <= point[1] < self.grid_width

    def _check_neighbors(self, point) -> bool:
        """Checks distance from point to all surrounding neighbors"""
        
        px, py = self._convert_point_grid(point)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = px+dx, py+dy
                if not self._check_valid_grid(nx, ny):
                    continue

                neigbor_point = self.grid[nx][ny]
                if neigbor_point is None:
                    continue

                if np.linalg.norm(neigbor_point - point) < self.min_radius:
                    return True
        
        return False

    def sample(self):
        """Creates new points surrounding an existing point
        
        Selects front point and generates up to sample_size
        candidate points uniformly in the annulus [r, 2r] around it.
        """
        
        base_point = self.active_list.pop()

        # idx = np.random.randint(len(self.active_list))
        # base_point = self.active_list[idx]

        for _ in range(self.sample_size):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.sqrt(np.random.uniform(pow(self.min_radius,2), pow(self.min_radius*2,2)))
            
            new_point = base_point + np.array([radius * np.cos(angle), radius * np.sin(angle)])

            if not self._check_valid_point(new_point) or self._check_neighbors(new_point):
                continue

            self.active_list.append(new_point)
            self.samples.append(new_point)
            nx, ny = self._convert_point_grid(new_point)
            self.grid[nx][ny] = new_point

    def generate(self):
        """Creates poisson disc distribution"""
        
        while self.active_list:
            self.sample()

        # return [p for row in self.grid for p in row if p is not None]
        return np.array(self.samples)


if __name__ == "__main__":
    """Run this file for an exampl output"""

    t = PoissonDisc(8,8,1.0,10)
    pd = PoissonDisc(grid_length=9, grid_width=9, min_radius=1.0, max_sample_size=30)
    points = pd.generate()

    print("Generated points:", len(points))
    print("Grid size:", pd.grid_size_x, "x", pd.grid_size_y)

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Poisson-disc samples (8x8, r=1.0)")
    plt.show()