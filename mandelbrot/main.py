from mandelbrot.python_backend import mandelbrot, mandelbrot_threaded
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from types import FunctionType
import argparse

def plot(counts, color_scheme = 'default'):
    """Plot the mandelbrot set according to the colorscheme

    This requires free-threaded python (python>=3.14t)

    Parameters
    ----------
    z: np.ndarray
        The final iterated z values
    counts : np.ndarray
        The number of iterations each z value took before becoming unbounded
    z_k: np.ndarray, optional(default=None)
        The final iterated z values + k iterations, only supplied when doing convergence scheme coloring
    epsilon: float
        The epsilon value used to color when doing the convergence scheme
    color_scheme: {'divergence', 'convergence', 'default'}, optional (default='default')
        The colorscheme to use when making the plot
    """
    cmap = None
    norm = None
    match color_scheme:
        case 'divergence':
            max = counts.max().item()
            # Divergence colormap from https://www.sekinoworld.com/fractal/#intro
            # |z1| > 2, black
            # |z2| > 2, red
            # |z3| > 2, black
            # etc.
            # max iterations, white
            # so, even indices red, odd black
            cmap = matplotlib.colors.ListedColormap(['red' if i % 2 == 0 else 'black' for i in range(max)] + ['white'])
            bounds=list(range(max+1))
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            plt.imshow(counts, cmap=cmap, norm=norm)
        case 'convergence':
            # Convergence colormap from https://www.sekinoworld.com/fractal/#intro
            # |z_{1+k} - z1| < epsilon, color_1
            # |z_{2+k} - z2| < epsilon, color_2
            # |z_{3+k} - z3| < epsilon, color_3
            # ...
            # |z_{m+k} - zm| < epsilon, color_m
            # etc.
            # leave everything uncolored, black
            rgb = np.zeros((*counts.shape, 4), dtype=np.float64)
            blue = np.array([0, 204, 255, 255]) /255
            green = np.array([0, 255, 200, 255]) /255
            black = np.array([0, 0, 0, 255]) /255
            red = np.array([255, 0, 0, 255]) /255
            rgb[counts < 0] = black
            rgb[counts == 0] = blue
            rgb[counts == 1] = red
            # rgb[counts == 2] = green
            plt.imshow(rgb, origin='lower')
        case _:
            color = "#00ad43"
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [color, "black"])
            plt.imshow(counts, cmap=cmap, norm=norm)
    plt.show()


def parse_args():
    """Setup and read command line arguments

    Returns
    ----------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["python_serial", "python_parallel"],
        default="python_serial",
        help="Which method to use to generate the image",
    )
    parser.add_argument("--rows", type=int, default=1000, help="number of rows in the image")
    parser.add_argument("--cols", type=int, default=1000, help="number of columns in the image")
    parser.add_argument("--iterations", type=int, default=100, help="maximum number of iterations")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Convergence scheme epsilon comparison")
    parser.add_argument("--k", type=int, default=None, help="k iterations, for convergence scheme")
    parser.add_argument(
        "--color_scheme",
        choices=["default", "divergence", "convergence"],
        default="default",
        help="Which method to use to generate color the image",
    )
    return parser.parse_args()


def generate_plot(func: FunctionType, rows: int, cols: int, **kwargs):
    """Generate the values for a mandelbrot plot

    Parameters
    ----------
    func: FunctionType
        A callable function which can be used to generate the mandelbrot set
    rows: int
        The number of rows in the image
    cols: int
        The number of columns in the image
    **kwargs: dict
        Other options that are specific to the generation function
    """

    # these are all of the c values we are going to iterate on
    c_values = np.zeros((rows, cols), dtype=np.complex128)

    xs = np.linspace(-2.0, 0.5, cols)
    ys = np.linspace(0, 1.12, rows)

    for i in range(rows):
        for j in range(cols):
            c_values[i][j] = np.complex128(xs[j], ys[i])

    # top half, reflection can give us the bottom half
    top_half = func(c_values, **kwargs)
    bottom_half = np.flip(top_half, 1)
    
    counts = np.stack([top_half, bottom_half[1:]])
    return counts


def main():
    args = parse_args()

    method = None
    match args.method:
        case "python_serial":
            method = mandelbrot
        case "python_parallel":
            method = mandelbrot_threaded

    counts = generate_plot(
        method, 
        args.rows, 
        args.cols, 
        max_iterations=args.iterations, 
        k_iterations=args.k,
        epsilon=args.epsilon
    )
    plot(
        counts=counts,
        color_scheme=args.color_scheme
    )

if __name__ == "__main__":
    main()
