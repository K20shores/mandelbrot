import numpy as np
from PIL import Image

def mandelbrot(z: complex, c: complex, max_iter: int = 100) -> int:
    iterations = 0
    while(np.abs(z) < 2 and iterations < max_iter):
        z = z**2 + c
        iterations += 1
    return iterations

rows = 1000
cols = 1000
max_iter=100

a = np.zeros((rows, cols), dtype=np.complex128)

xs = np.linspace(-2.0, 0.5,cols)
ys = np.linspace(-1.12, 1.12, rows)

for i in range(rows):
    for j in range(cols):
        a[i][j] = mandelbrot(0, np.complex128(xs[j], ys[i]), max_iter=max_iter)

# Convert `a` to a PIL Image
img = Image.fromarray(np.real(a))
img.show()