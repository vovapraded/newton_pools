import numpy as np
import matplotlib.pyplot as plt

width, height = 4000, 4000
dpi = 1200
re_min, re_max = -1.5, 1.5
im_min, im_max = -1.5, 1.5
tolerance = 1e-8
max_iter = 500

width_in_inches = width / dpi
height_in_inches = height / dpi
fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')


def newton_fractal(f_coeffs, max_iter):
    roots = np.array(np.roots(f_coeffs))

    f = lambda z: np.polyval(f_coeffs, z)
    f_prime = lambda z: np.polyval(np.polyder(f_coeffs), z)

    re = np.linspace(re_min, re_max, width)
    im = np.linspace(im_min, im_max, height)
    X, Y = np.meshgrid(re, im)
    Z = X + 1j * Y

    for i in range(max_iter):
        f_prime_val = f_prime(Z)
        small_derivative = np.abs(f_prime_val) < tolerance

        Z[~small_derivative] -= f(Z[~small_derivative]) / f_prime_val[~small_derivative]

    distances = np.abs(Z[..., np.newaxis] - roots)
    closest_root = np.argmin(distances, axis=2)

    return closest_root


def make_and_save(f_coeffs, max_iter, name):
    image = newton_fractal(f_coeffs, max_iter)

    filename = f"newton_fractal_{name}_{max_iter}.png"

    plt.imshow(image, cmap='inferno', extent=(re_min, re_max, im_min, im_max))
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)


polynomials = {
    "z^3 - 1": [1, 0, 0, -1],
    "z^5 - 1": [1, 0, 0, 0, 0, -1],
    "z^4 - 1": [1, 0, 0, 0, -1],
    "z^4 + 1": [1, 0, 0, 0, 1],
    "z^5 - z^2 - 1": [1, 0, 0, -1, 0, -1]
}

for poly in polynomials:
    make_and_save(polynomials[poly], max_iter, poly)