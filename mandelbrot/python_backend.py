import numpy as np
import sys


def __k_iterations(z: np.complex128, c: np.complex128, k_iterations: int = 0) -> np.int32:
    """Iterate the linear mapping for the mandelbrot set exactly k times

    This is used when computing the convergence scheme

     Parameters
    ----------
    z : complex
        Initial z value
    c : complex
        Intial c value
    k_iterations: int, optional (default=100)
        The number of iterations to bail out at

    Returns
    -------
    int
        The number of iterations it takes for z to become unbounded, up to max_iter
    """
    for _ in range(k_iterations):
        mask = np.abs(z) <= 2,
        z[mask] = z[mask] * z[mask] + c[mask]
    return z


def __iteration(z: np.complex128, c: np.complex128, max_iterations: int = 100, z_k: np.complex128 = None, epsilon: float = None) -> np.int32:
    """Iterate the linear mapping for the mandelbrot set

     Parameters
    ----------
    z : np.complex128
        Initial z value
    c : np.complex128
        Intial c value
    max_iterations: int, optional (default=100)
        The number of iterations to bail out at
    z_k : np.complex128, optional (default=None)
        z pre-run k steps forward, only needed when doing a convergence check
    epsilon : float, optional (default=None)
        the convergence epsilon |z_k - z| < epsilon

    Returns
    -------
    int
        The number of iterations it takes for z to become unbounded, up to max_iter
    """

    iterations = np.zeros_like(c, dtype=np.int32)
    mask = np.abs(z) <= 2,
    for _ in range(max_iterations):
        if (not np.any(mask)):
            break
        z[mask] = z[mask] * z[mask] + c[mask]
        iterations[mask] += 1
        mask = np.abs(z) <= 2,
    return iterations


def mandelbrot(c_values: np.ndarray, max_iterations: int = 100, k_iterations: int = None, epsilon: float = None) -> np.int32:
    """Compute the mandelbrot set

     Parameters
    ----------
    c_values : np.ndarray
        All c values we will iterate the mandelbrot map for
    max_iterations: int, optional (default=100)
        The number of iterations to bail out at
    k_iterations: int, optional (default=None)
        Supplied when computing the convergence scheme, this is the number of iterations to continue
        computing the linear mapping for. When supplied, the return value will be 2 ndarrays
    epsilon: float, optional (default=None)
        The epsilon check to use for convergence, when k_iterations is supplied

    Returns
    -------
    np.ndarray
        The number of iterations for each c value where z became unbounded
    """
    counts = np.zeros_like(c_values, dtype=np.int32)
    z = np.zeros_like(c_values, dtype=np.complex128)

    # iterate z once to generate the mandelbrot set
    counts = __iteration(z, c_values, max_iterations=max_iterations)

    # for convergence coloring
    if k_iterations:
        # pixels represents everything in the mandelbrot set, those points that didn't diverge
        # only operate on those
        pixels = np.abs(z) < 2

        # will will run several a total of k_bands ahead of z, this allows us to color individual k-converged atoms
        # and allows us to color other atoms black (see below)
        k_band = 8
        zks = []
        for k in range(k_iterations+k_band):
            z_k = np.zeros_like(z, dtype=np.complex128)
            # then lead z by k iterations to obtain z_k, but only for points that didn't diverge
            z_k[pixels] = __k_iterations(z_k[pixels], c_values[pixels], k_iterations=k+1)
            zks.append(z_k)
        
        # we need to carry three pieces of information to properly apply convergence coloring
        # 1. points on the mandelbrot set and which neither converge or diverge, our boundary, we will set this to 0
        #   - "...the Julia set ... is approximated by ... c(i, j) = M and d(i, j) = M"
        #   - ^ from https://www.sekinoworld.com/fractal/coloring.htm
        #   - those words means that the boundary is the difference between things that converged and things that didn't diverge
        # 2. points on the mandelbrot set that do converge (positive number, set to which period k it convergences to)
        # 3. points that diverge. For this color scheme, we will indicate them with -1

        # reset z to find convergence
        z[:] = 0
        _z = z[pixels]
        _c_values = c_values[pixels]
        _counts = counts[pixels]
        _zks = [zk[pixels] for zk in zks]
        for _ in range(max_iterations):
            _z = _z * _z + _c_values
            for k in range(k_iterations+k_band):
                alive = np.abs(_zks[k]) < 2
                _zks[k][alive] = _zks[k][alive] * _zks[k][alive] + _c_values[alive]
                mask = (np.abs(_zks[k] - _z) < epsilon) & ((_counts == max_iterations) | (_counts > k + 1))
                _counts[mask] = k+1
        counts[pixels] = _counts
        boundary = pixels & (counts == max_iterations)
        # boundary points
        counts[boundary] = 0
        # diverged points
        counts[~pixels] = -1
        # atoms which are not on the boundary and are not within k iterations shouldn't be colored
        counts[counts > k_iterations] = -1

    return counts


def mandelbrot_threaded(c_values: np.ndarray, max_iterations: int = 100, n_workers: int = 8, k: int = None) -> np.int32:
    """Compute the mandelbrot set in parallel

    This requires free-threaded python (python>=3.14t)

    Parameters
    ----------
    c_values : np.ndarray
        All c values we will iterate the mandelbrot map for
    max_iterations: int, optional (default=100)
        The number of iterations to bail out at
    n_workers: int, optinal (default=8)
        The number of workers to run in parallel

    Returns
    -------
    np.ndarray
        The number of iterations for each c value where z became unbounded
    """

    if sys._is_gil_enabled():
        raise RuntimeError("Free-threaded python is required")

    from concurrent.futures import ThreadPoolExecutor

    rows = c_values.shape[0]
    cols = c_values.shape[1]
    counts = np.zeros((rows, cols), dtype=np.int32)
    z = np.zeros((rows, cols), dtype=np.complex128)

    if rows < n_workers:
        n_workers = rows

    def work(row_chunks):
        nonlocal counts
        start, end = row_chunks
        counts[start:end] = __iteration(z[start:end], c_values[start:end], max_iterations=max_iterations)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        div = rows // n_workers
        chunks = list(range(0, rows + div, div))
        row_chunks = [(start, end) for start, end in zip(chunks[:-1], chunks[1:])]
        list(pool.map(work, row_chunks))

    return counts
