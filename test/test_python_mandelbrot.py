from mandelbrot.python_backend import mandelbrot, mandelbrot_threaded
import numpy as np
import pytest
import sys


@pytest.fixture
def c_values():
    rows = 100
    cols = 100

    xs = np.linspace(-2.0, 0.5, cols)
    ys = np.linspace(-1.12, 1.12, rows)
    X, Y = np.meshgrid(xs, ys)
    return X + 1j * Y


def test_python_serial_mandelbrot():
    c = np.zeros((1, 1), dtype=np.complex128)
    c[0][0] = np.complex128(1, 0)
    assert mandelbrot(c) == 3


def test_python_serial_mandelbrot_convergence(c_values):
    counts = mandelbrot(c_values, k_iterations=4)
    assert counts.min() == -1
    assert counts.max() == 4
    assert np.any(counts == 0)
    counts = mandelbrot(c_values, k_iterations=2)
    assert counts.min() == -1
    assert counts.max() == 2
    assert np.any(counts == 0)


@pytest.mark.skipif(sys._is_gil_enabled(), reason="Free-threaded python is required")
def test_python_parallel_mandelbrot():
    c = np.zeros((1, 1), dtype=np.complex128)
    c[0][0] = np.complex128(1, 0)
    assert mandelbrot_threaded(c) == 3


@pytest.mark.skipif(sys._is_gil_enabled(), reason="Free-threaded python is required")
def test_compare_serial_parallel(c_values):
    serial = mandelbrot(c_values)
    parallel = mandelbrot_threaded(c_values)
    rows = c_values.shape[0]
    assert serial.shape == parallel.shape
    for row in range(rows):
        assert np.all(serial[row] == parallel[row]), f"Row {row} doesn't match"
