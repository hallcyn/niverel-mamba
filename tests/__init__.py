"""Test package marker.

Without these, pytest puts tests/ on sys.path and the directory
`tests/mlx` becomes a namespace package that SHADOWS the real MLX import.
That made `mlx_available()` return True on a machine with no MLX at all.
"""
