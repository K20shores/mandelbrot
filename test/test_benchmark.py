from mandelbrot.python_backend import mandelbrot, mandelbrot_threaded
import numpy as np
import pytest
import sys


@pytest.fixture(params=[100, 150, 200])
def c_values(request):
    rows = request.param
    cols = request.param
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
@pytest.mark.parametrize("n_workers", [2, 4, 6, 8])
@pytest.mark.skipif(sys._is_gil_enabled(), reason="Free-threaded python is required")
def test_python_parallel_mandelbrot_benchmark(benchmark, c_values, n_workers):
    benchmark(mandelbrot_threaded, c_values=c_values, n_workers=n_workers)
