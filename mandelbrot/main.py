from mandelbrot.python_backend import mandelbrot, mandelbrot_threaded
import numpy as np
from PIL import Image


def main():
    rows = 1000
    cols = 1000

    # these are all of the c values we are going to iterate on
    c_vals = np.zeros((rows, cols), dtype=np.complex128)

    xs = np.linspace(-2.0, 0.5, cols)
    ys = np.linspace(-1.12, 1.12, rows)

    for i in range(rows):
        for j in range(cols):
            c_vals[i][j] = np.complex128(xs[j], ys[i])

    # counts = mandelbrot(c_vals)
    counts = mandelbrot_threaded(c_vals)

    img = Image.fromarray(counts)
    img.show()


if __name__ == "__main__":
    main()
[(0, 41), (41, 82), (82, 123), (123, 164), (164, 205), (205, 246), (246, 287), (287, 328), (328, 369), (369, 410), (410, 451), (451, 492), (492, 533), (533, 574), (574, 615), (615, 656), (656, 697), (697, 738), (738, 779), (779, 820), (820, 861), (861, 902), (902, 943), (943, 984), (984, 1025)]
