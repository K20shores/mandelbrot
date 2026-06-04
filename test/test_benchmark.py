from mandelbrot.python_backend import mandelbrot, mandelbrot_threaded
import numpy as np
import pytest


@pytest.fixture
def c_values():
    rows = 1000
    cols = 1000
    c_vals = np.zeros((rows, cols), dtype=np.complex128)

    xs = np.linspace(-2.0, 0.5, cols)
    ys = np.linspace(-1.12, 1.12, rows)

    for i in range(rows):
        for j in range(cols):
            c_vals[i][j] = np.complex128(xs[j], ys[i])

    return c_vals


def test_python_serial_mandelbrot_benchmark(benchmark, c_values):
    benchmark(mandelbrot, c_values)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_workers", [1, 2, 4, 8, 12, 24])
def test_python_parallel_mandelbrot_benchmark(benchmark, c_values, n_workers):
    benchmark(mandelbrot_threaded, c_values=c_values, n_workers=n_workers)
