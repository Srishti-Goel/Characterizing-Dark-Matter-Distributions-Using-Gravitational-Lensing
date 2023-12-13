import autolens as al
# import autolens.plot as aplt
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import random

import os
import time
import datetime

WIDTH = 600
HEIGHT = 600
SIM_NO = 0
NO_OF_IMAGES_TO_GENERATE = 1
PIXEL_SCAlES = 0.01

grid = al.Grid2D.uniform(shape_native=(HEIGHT,WIDTH), pixel_scales=PIXEL_SCAlES)

while 1:
    file_name = "sim" + str(SIM_NO) + '.jpeg'
    if not os.path.exists(file_name):
        break
    SIM_NO += 1
print("Simulation number:", SIM_NO)
file_name = "sim" + str(SIM_NO) + '.jpeg'

def generate_ell_comps():
    fac = min(random.random(), .5)
    angle = random.random() * 2 * np.pi

    return (fac * np.sin(2*angle), fac * np.cos(2*angle))

for iter in range(NO_OF_IMAGES_TO_GENERATE):
    tic = time.time()

    # ----CREATING THE SOURCE GALAXY------
    source_centre = (random.random()*HEIGHT*PIXEL_SCAlES/2, random.random()*WIDTH*PIXEL_SCAlES/2)

    # DECIDING THE SOURCE'S BULDGE
    source_buldge_type = random.randint(0,3)
    source_intensity = random.random() + 1e-1
    source_ell_comps = generate_ell_comps()
    source_buldge = None
    if source_buldge_type == 0:
        source_buldge = al.lp.Gaussian(
            centre=source_centre,
            ell_comps=source_ell_comps,
            intensity=source_intensity,
            sigma=random.random()
        )
    elif source_buldge_type == 1:
        source_buldge = al.lp.Sersic(
            centre=source_centre,
            ell_comps=source_ell_comps,
            intensity=source_intensity,
            effective_radius=max(random.random(), 0.5),
            sersic_index=random.randint(1, 10)
        )
    elif source_buldge_type == 2:
        r1 = max(random.random(), 0.5)
        r2 = max(random.random(), 0.5)
        source_buldge = al.lp.Chameleon(
            centre=source_centre,
            ell_comps=source_ell_comps,
            intensity=source_intensity,
            core_radius_0=min(r1, r2),
            core_radius_1=max(r1,r2)
        )
    else:
        source_buldge = al.lp.Exponential(
            centre=source_centre,
            ell_comps=source_ell_comps,
            intensity=source_intensity,
            effective_radius=max(random.random(), 0.5)
        )
    
    # DECIDING THE SOURCE'S DISK
    source_disk_type = random.randint(0,3)
    source_intensity = random.random()
    source_ell_comps = generate_ell_comps()
    source_disk = None
    if source_disk_type == 0:
        source_disk = al.lp.Gaussian(
            centre=source_centre,
            ell_comps=source_ell_comps,
            intensity=source_intensity,
            sigma=random.random()
        )
    elif source_disk_type == 1:
        source_disk = al.lp.Sersic(
            centre=source_centre,
            ell_comps=source_ell_comps,
            intensity=source_intensity,
            effective_radius=max(random.random(),.5),
            sersic_index=random.randint(1, 10)
        )
    elif source_disk_type == 2:
        r1 = max(random.random(), 0.5)
        r2 = max(random.random(), 0.5)
        source_disk = al.lp.Chameleon(
            centre=source_centre,
            ell_comps=source_ell_comps,
            intensity=source_intensity,
            core_radius_0=min(r1, r2),
            core_radius_1=max(r1,r2)
        )
    else:
        source_disk = al.lp.Exponential(
            centre=source_centre,
            ell_comps=source_ell_comps,
            intensity=source_intensity,
            effective_radius=max(random.random(), 0.5)
        )
    
    # ACTUAL SOURCE CREATION
    r1 = random.random()
    r2 = random.random()
        # Redshift of the source is the max of r1 and r2, that of the lensing galaxy is the min
    source = al.Galaxy(
        redshift=max(r1, r2),
        buldge = source_buldge,
        disk = source_disk
    )


    # ----CREATING THE SOURCE GALAXY------
    lenser_center = (random.random()*HEIGHT* PIXEL_SCAlES/2, random.random()*WIDTH* PIXEL_SCAlES/2)
    
    # Deciding the normal matter mass profile:
    normal_matter_mp_type = random.randint(0,3)
    nm_center = (random.random()*HEIGHT* PIXEL_SCAlES/2, random.random()*WIDTH* PIXEL_SCAlES/2)
    nm_ell_comps = generate_ell_comps()
    nm_intensity = random.random() + 1e-1
    nm_m = None
    nm_l = None

    if normal_matter_mp_type == 0:
        sigma=random.random()
        nm_m = al.mp.Gaussian(
            centre=nm_center,
            ell_comps=nm_ell_comps,
            intensity=nm_intensity,
            sigma=sigma
        )
        nm_l = al.lp.Gaussian(
            centre=nm_center,
            ell_comps=nm_ell_comps,
            intensity=nm_intensity,
            sigma=sigma
        )
    elif normal_matter_mp_type == 1:
        eff_radius = max(random.random(), 0.5)
        sersic_index = random.randint(1,10)
        nm_m = al.mp.Sersic(
            centre=nm_center,
            ell_comps=nm_ell_comps,
            intensity=nm_intensity,
            effective_radius=eff_radius,
            sersic_index=sersic_index
        )

        nm_l = al.lp.Sersic(
            centre=nm_center,
            ell_comps=nm_ell_comps,
            intensity=nm_intensity,
            effective_radius=eff_radius,
            sersic_index=sersic_index
        )
    elif normal_matter_mp_type == 2:
        rr1 = random.random()
        rr2 = random.random()
        nm_m = al.mp.Chameleon(
            centre=nm_center,
            ell_comps=nm_ell_comps,
            intensity=nm_intensity,
            core_radius_0=min(rr1, rr1),
            core_radius_1=max(rr1, rr2)
        )
        nm_l = al.lp.Chameleon(
            centre=nm_center,
            ell_comps=nm_ell_comps,
            intensity=nm_intensity,
            core_radius_0=min(rr1, rr1),
            core_radius_1=max(rr1, rr2)
        )
    elif normal_matter_mp_type == 3:
        eff_radius = max(random.random(), 0.5)
        nm_m = al.mp.Exponential(
            centre=nm_center,
            ell_comps=nm_ell_comps,
            intensity=nm_intensity,
            effective_radius=eff_radius
        )
        nm_l = al.lp.Exponential(
            centre=nm_center,
            ell_comps=nm_ell_comps,
            intensity=nm_intensity,
            effective_radius=eff_radius
        )
    lenser = al.Galaxy(
        redshift=min(r1, r2),
        nm = nm_m,
        nm_l = nm_l
    )

    toc = time.time()
    
    tracer = al.Tracer.from_galaxies(galaxies=[source])
    traced_image = tracer.image_2d_from(grid=grid)
    plt.imshow(traced_image.native)
    plt.show()
    
    tracer = al.Tracer.from_galaxies(galaxies=[lenser])
    traced_image = tracer.image_2d_from(grid=grid)
    plt.imshow(traced_image.native)
    plt.show()
    
    tracer = al.Tracer.from_galaxies(galaxies=[source, lenser])
    traced_image = tracer.image_2d_from(grid=grid)
    plt.imshow(traced_image.native)
    plt.show()

    print("Simulated specs:", normal_matter_mp_type, source_buldge_type, source_disk_type)

    max_val = np.max(traced_image.native)

    tr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    tr[:, :, 0] = traced_image.native * 255 * random.random() / max_val
    tr[:, :, 1] = traced_image.native * 255 * random.random() / max_val
    tr[:, :, 2] = traced_image.native * 255 * random.random() / max_val

    img = Image.fromarray(tr, 'RGB')
    img.save(file_name)

    print("Created source in", toc-tic, "secs")