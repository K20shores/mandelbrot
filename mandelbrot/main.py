from mandelbrot.python_backend import mandelbrot, mandelbrot_threaded
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from types import FunctionType
import argparse


def plot(counts, color_scheme="default", save_path=None):
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
    save_path: str, optional (default=None)
        If present, save the image to the path rather than display it
    """
    fig, ax = plt.subplots()
    cmap = None
    norm = None
    match color_scheme:
        case "divergence":
            max = counts.max().item()
            # Divergence colormap from https://www.sekinoworld.com/fractal/#intro
            # |z1| > 2, black
            # |z2| > 2, red
            # |z3| > 2, black
            # etc.
            # max iterations, white
            # so, even indices red, odd black
            cmap = matplotlib.colors.ListedColormap(["red" if i % 2 == 0 else "black" for i in range(max)] + ["white"])
            bounds = list(range(max + 1))
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            ax.imshow(counts, cmap=cmap, norm=norm, origin="lower")
        case "convergence":
            # Convergence colormap from https://www.sekinoworld.com/fractal/#intro
            # |z_{1+k} - z1| < epsilon, color_1
            # |z_{2+k} - z2| < epsilon, color_2
            # |z_{3+k} - z3| < epsilon, color_3
            # ...
            # |z_{m+k} - zm| < epsilon, color_m
            # etc.
            # leave everything uncolored, black
            rgb = np.zeros((*counts.shape, 4), dtype=np.float64)
            blue = np.array([0, 204, 255, 255]) / 255
            black = np.array([0, 0, 0, 255]) / 255
            rgb[counts < 0] = black
            rgb[counts == 0] = blue
            cmap = plt.cm.get_cmap("tab20")
            for k in range(1, counts.max() + 1):
                rgb[counts == k] = cmap(k / counts.max())
            ax.imshow(rgb, origin="lower")
        case _:
            color = "#00ad43"
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [color, "black"])
            ax.imshow(counts, cmap=cmap, norm=norm)
    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
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
    parser.add_argument("--xmin", type=float, default=-2.0, help="The minimum x of the canvas")
    parser.add_argument("--xmax", type=float, default=0.5, help="The maximum x of the canvas")
    parser.add_argument("--ymin", type=float, default=-1.12, help="The minimum y of the canvas")
    parser.add_argument("--ymax", type=float, default=1.12, help="The maximum y of the canvas")
    parser.add_argument("--save", type=str, help="File path to save the image")
    args = parser.parse_args()
    if args.color_scheme == "convergence" and args.k is None:
        parser.error("--k is required when --color_scheme is 'convergence'")
    return args


def generate_plot(func: FunctionType, rows: int, cols: int, xbounds: tuple, ybounds: tuple, **kwargs):
    """Generate the values for a mandelbrot plot

    Parameters
    ----------
    func: FunctionType
        A callable function which can be used to generate the mandelbrot set
    rows: int
        The number of rows in the image
    cols: int
        The number of columns in the image
    xbounds: tuple
        The min, max of the x range of the canvas
    ybounds: tuple
        The min, max of the y range of the canvas
    **kwargs: dict
        Other options that are specific to the generation function
    """
    xs = np.linspace(xbounds[0], xbounds[1], cols)
    ys = np.linspace(ybounds[0], ybounds[1], rows)
    if ybounds[0] < 0 and 0 < ybounds[1]:
        positive = ys >= 0
        n_pos = len(ys[positive])
        n_neg = rows - n_pos
        if n_pos > n_neg:
            X, Y = np.meshgrid(xs, ys[positive])
            half = func(X + 1j * Y, **kwargs)
            counts = np.vstack((np.flip(half[:n_neg], 0), half))
        else:
            X, Y = np.meshgrid(xs, ys[~positive])
            half = func(X + 1j * Y, **kwargs)
            counts = np.vstack((half, np.flip(half[-n_pos:], 0)))

    else:
        X, Y = np.meshgrid(xs, ys)
        counts = func(X + 1j * Y, **kwargs)

    assert counts.shape == (rows, cols), f"{counts.shape} != ({rows}, {cols})"
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
        (args.xmin, args.xmax),
        (args.ymin, args.ymax),
        max_iterations=args.iterations,
        k_iterations=args.k,
        epsilon=args.epsilon,
    )
    plot(counts=counts, color_scheme=args.color_scheme, save_path=args.save)


if __name__ == "__main__":
    main()
