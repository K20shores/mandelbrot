from mandelbrot.python_backend import mandelbrot, mandelbrot_threaded
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from types import FunctionType
import argparse


def plot(counts):
    color = "#00ad43"
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [color, "white"])
    plt.imshow(np.real(counts), cmap="inferno_r")
    plt.show()


def parse_args():
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
    ys = np.linspace(-1.12, 1.12, rows)

    for i in range(rows):
        for j in range(cols):
            c_values[i][j] = np.complex128(xs[j], ys[i])

    return func(c_values, **kwargs)


def main():
    args = parse_args()

    method = None
    match args.method:
        case "python_serial":
            method = mandelbrot
        case "python_parallel":
            method = mandelbrot_threaded

    values = generate_plot(method, args.rows, args.cols, max_iterations=args.iterations)
    plot(values)


if __name__ == "__main__":
    main()
