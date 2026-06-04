from mandelbrot.python_backend import mandelbrot, mandelbrot_threaded
import numpy as np
import pytest
import sys


@pytest.fixture
def c_values():
    rows = 100
    cols = 100
    c_vals = np.zeros((rows, cols), dtype=np.complex128)

    xs = np.linspace(-2.0, 0.5, cols)
    ys = np.linspace(-1.12, 1.12, rows)

    for i in range(rows):
        for j in range(cols):
            c_vals[i][j] = np.complex128(xs[j], ys[i])

    return c_vals


def test_python_serial_mandelbrot():
    c = np.zeros((1, 1), dtype=np.complex128)
    c[0][0] = np.complex128(1, 0)
    assert mandelbrot(c) == 3


@pytest.mark.skipif(sys._is_gil_enabled(), reason="Free-threaded python is required")
def test_python_parallel_mandelbrot():
    c = np.zeros((1, 1), dtype=np.complex128)
    c[0][0] = np.complex128(1, 0)
    assert mandelbrot_threaded(c, n_workers=1) == 3


@pytest.mark.skipif(sys._is_gil_enabled(), reason="Free-threaded python is required")
def test_compare_serial_parallel(c_values):
    serial = mandelbrot(c_values)
    parallel = mandelbrot_threaded(c_values)
    rows = c_values.shape[0]
    assert serial.shape == parallel.shape
    for row in range(rows):
        assert np.all(serial[row] == parallel[row]), f"Row {row} doesn't match"
