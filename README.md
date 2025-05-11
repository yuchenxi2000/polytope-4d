# Polytope-4D

Visualization of 4 dimensional regular convex polytopes

## Background

There are 6 regular convex polytopes in 4 dimensional space, the 5-cell, 8-cell, 16-cell, 24-cell, 120-cell and 600-cell. As a 3D creature, we can only see their projections into our 3D space. I wrote Python scripts to visualize their 3D projections in Blender and VMD softwares.

## Blender animation

Here's the video I made using `polytope_blender.py` :

(Youtube) https://youtu.be/37Eh7hyGgmE

(Bilibili) https://www.bilibili.com/video/BV1yXL2ztEPX

To make your own animation, you need to

1. generate polytope data files using `polytope_4d.py` (data files are under `data` directory)
2. paste `polytope_blender.py` script into Blender. You need to do some editing (file path, materials, transformations, etc.) for your own animation.

> The script `polytope_blender.py` has no extra dependencies, which means it can be run directly in Blender.
>
> However, to generate polytope data files, you need to install numpy and scipy. You can also use the data files in this repository under `data` directory.

## VMD Visualization

The script `polytope_vmd.py` writes a vtf file that can be opened in VMD software.

VMD is a software for analysing results of molecular dynamics simulations, but I found it can also visualize 4D polytopes!

Open VMD, then File -> New Molecule, select the vtf file and Load. For best visualization, you can set Display -> Orthographic and Graphics -> Representations -> Drawing Method -> Lines, increase line thickness to a large value such as 5.

> Packages numpy and scipy are required.

## Acknowledgements

When I was in junior high, my math teacher played a film named Dimensions (link: http://dimensions-math.org/) in our math class. The chapter 3 and 4 of Dimensions are about the 4-polytopes in 4D space. Now I have adequate knowledges to reproduce the projection of these rotating 4D objects into our 3D space. Great thanks to Jos Leys for the excellent film!

## License

Copyright (c) 2025 yuchenxi2000. Licensed under MIT License.

