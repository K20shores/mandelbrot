# Mandelbrot, many ways

This repository is a companion set of code along with a series of blog posts. 
The posts detail how Python can wrap many lower level languages. 
I do so by demonstrating generation of the Mandelbrot set, in many languages 
and compute devices.

In the creation of this repository, I was looking for a good colormap
to apply to the generated sets. I stumbled upon this 
[fantastic website](https://www.sekinoworld.com/fractal/#intro) which is all 
about fractals. It contains incredibly detailed descriptions of fractals
of all kinds, and tips for generating them, including colormaps.

# Installation

(soon)
```bash
pip install k20shores-mandelbrot
```

# Development
I recommend [`uv`](https://docs.astral.sh/uv/)

```bash
uv sync
```

## Testing
```bash
uv run pytest
```

## CLI
```bash
uv run mandelbrot -h
```