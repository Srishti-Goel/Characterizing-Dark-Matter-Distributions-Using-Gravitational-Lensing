import autolens as al
import autolens.plot as aplt

"""
To describe the deflection of light by mass, two-dimensional grids of (y,x) Cartesian
coordinates are used.
"""

grid_2d = al.Grid2D.uniform(
    shape_native=(50, 50),
    pixel_scales=0.05,  # <- Conversion from pixel units to arc-seconds.
)

"""
The lens galaxy has an elliptical isothermal mass profile and is at redshift 0.5.
"""

sie = al.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=1.6
)

lens_galaxy = al.Galaxy(redshift=0.5, mass=sie)

"""The source galaxy has an elliptical exponential light profile and is at redshift 1.0."""

exponential = al.lp.Exponential(
    centre=(0.3, 0.2),
    ell_comps=(0.05, 0.25),
    intensity=0.05,
    effective_radius=0.5,
)

source_galaxy = al.Galaxy(redshift=1.0, light=exponential)

"""
We create the strong lens using a Tracer, which uses the galaxies, their redshifts
and an input cosmology to determine how light is deflected on its path to Earth.
"""

tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy], cosmology=al.cosmo.Planck15()
)

"""
We can use the Grid2D and Tracer to perform many lensing calculations, for example
plotting the image of the lensed source.
"""

aplt.Tracer.image(tracer=tracer, grid=grid_2d)