import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.3, n=200):
        """
        Initialize the Möbius strip.

        Parameters:
        R (float): Radius from the center to the strip
        w (float): Width of the strip
        n (int): Number of points in each direction for resolution
        """
        self.R = R
        self.w = w
        self.n = n
        self.u_vals = np.linspace(0, 2 * np.pi, n)
        self.v_vals = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u_vals, self.v_vals)
        self.X, self.Y, self.Z = self._compute_coordinates()

    def _compute_coordinates(self):
        """Compute the 3D coordinates of the Möbius strip surface."""
        u = self.U
        v = self.V
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def surface_area(self):
        """
        Approximate the surface area of the Möbius strip using the mesh.

        Returns:
        float: Approximated surface area
        """
        # Compute partial derivatives for area element
        du = 2 * np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)

        Xu = np.gradient(self.X, du, axis=1)
        Yu = np.gradient(self.Y, du, axis=1)
        Zu = np.gradient(self.Z, du, axis=1)

        Xv = np.gradient(self.X, dv, axis=0)
        Yv = np.gradient(self.Y, dv, axis=0)
        Zv = np.gradient(self.Z, dv, axis=0)

        # Cross product of partial derivatives
        Nx = Yu * Zv - Zu * Yv
        Ny = Zu * Xv - Xu * Zv
        Nz = Xu * Yv - Yu * Xv

        dA = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        surface_area = np.sum(dA) * du * dv
        return surface_area

    def edge_length(self):
        """
        Approximate the edge length by tracing one of the boundaries (v = w/2 or -w/2).

        Returns:
        float: Approximated edge length (single loop)
        """
        u = self.u_vals
        v = self.w / 2
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)

        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)

        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        length = np.sum(ds)
        return length

    def plot(self):
        """Visualize the Möbius strip using matplotlib."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', rstride=1, cstride=1, linewidth=0)
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.3, n=300)
    print(f"Approximated Surface Area: {mobius.surface_area():.4f}")
    print(f"Approximated Edge Length: {mobius.edge_length():.4f}")
    mobius.plot()
