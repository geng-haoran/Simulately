---
sidebar_position: 0
---


## Object Shapes

- [**ShapeNet**](https://shapenet.org/): ShapeNet is an ongoing effort to establish a richly-annotated, large-scale
  dataset of 3D shapes. We provide researchers around the world with this data to enable research in computer graphics,
  computer vision, robotics, and other related
  disciplines. [[Download]](https://shapenet.org/login/) [[Paper]](https://arxiv.org/abs/1512.03012)
  
- [**FineSeg**](http://kevinkaixu.net/projects/shape2motion.html): Shape2Motion paper releases FineSeg dataset, which
  contains about 3000 3D shapes over six shape categories: chair (1000), table (500), airplanes (600), sofa (600),
  helicopter (100) and bike (140). The models are collected from a subset of
  ShapeNet.[[Download]](https://drive.google.com/file/d/1ZtWgMqYSNl1MSXKaTnMdN6xHXme9AjXb/view?usp=sharing) [[Paper]](https://arxiv.org/abs/1903.00709) [[Code]](https://github.com/wangxiaogang866/Shape2Motion)

- [**PartNet**](https://partnet.cs.stanford.edu/): PartNet is a consistent, large-scale dataset of 3D objects annotated
  with fine-grained, instance-level, and hierarchical 3D part
  information. [[Download]](https://www.shapenet.org/login/) [[Code]](https://github.com/daerduoCarey/partnet_dataset) [[Paper]](https://arxiv.org/abs/1812.02713)
- [**YCB**](https://www.ycbbenchmarks.com/): YCB Object and Model Set is designed for facilitating benchmarking in
  robotic manipulation. The set consists of objects in daily life with different shapes, sizes, textures, weight and
  rigidity, as well as some widely used manipulation
  tests. [[Download]](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/); [[Paper]](https://journals.sagepub.com/doi/full/10.1177/0278364917700714)
- [**OmniObject3D**](https://omniobject3d.github.io/): OmniObject3D, a large vocabulary 3D object dataset with massive
  high-quality real-scanned 3D objects to facilitate the development of 3D perception, reconstruction, and generation in
  the real
  world. [[Download]](https://opendatalab.com/OpenXDLab/OmniObject3D-New/tree/main) [[Paper]](https://arxiv.org/abs/2301.07525) [[Code]](https://github.com/omniobject3d/OmniObject3D/tree/main)
- [**Objaverse-XL**](https://objaverse.allenai.org/): A Universe of 10M+ 3D Objects. Objaverse-XL is 12x larger than
  Objaverse 1.0 and 100x larger than all other 3D datasets
  combined. [[Download]](https://docs.google.com/forms/d/e/1FAIpQLScNOWKTHk3a7CGiegNjROFNfOcpzr5gt6G0FMEMQ8qXRTbs0Q/viewform) [[Paper]](https://arxiv.org/abs/2307.05663) [[Code]](https://github.com/allenai/objaverse-xl)

## Articulated Objects

- [**PartNet Moblity**](https://sapien.ucsd.edu/browse): SAPIEN releases PartNet-Mobility dataset, which is a collection
  of 2K articulated objects with motion annotations and rendering material. The dataset powers research for
  generalizable computer vision and
  manipulation. [[Download]](https://sapien.ucsd.edu/downloads) [[Paper]](https://arxiv.org/abs/2003.08515)

- [**AKB48**](https://liuliu66.github.io/articulationobjects/index.html): This is a realistic and physics-rich object
  repository for articulation analysis. It enables various robotic vision and interaction tasks that require detailed
  part-level understanding. More categories and download link will be released
  soon. [[Download]](https://liuliu66.github.io/articulationobjects/download.html) [[Paper]](https://arxiv.org/abs/2202.08432)

- [**GAPartNet**](https://pku-epic.github.io/GAPartNet/) (Merge and provide more annotations from PartNet Mobility and
  AKB48): By identifying and defining 9 GAPart classes (lids, handles, etc.) in 27 object categories, we construct a
  large-scale part-centric interactive dataset, GAPartNet, where we provide rich, part-level annotations (semantics,
  poses). [[Download]](https://forms.gle/3qzv8z5vP2BT5ARN7) [[Paper]](https://arxiv.org/abs/2211.05272) [[Code]](https://github.com/PKU-EPIC/GAPartNet)

- [**UniDoorManip**](https://unidoormanip.github.io/): This environment consists of the diverse door assets and door manipulation simulation with mechanisms.
  The dataset consists of door bodies and handles covering 6 categories (Interior, Window, Car, Safe, StorageFurniture, Refrigerator) for PartNet-Mobility and 3D Warehouse.
  The simulation provides mechanisms (such as locking and latching) of different doors. 
  [[Download]](https://github.com/sectionZ6/UniDoorManip) [[Paper]](https://arxiv.org/abs/2211.05272) [[Code]](https://github.com/sectionZ6/UniDoorManip)

## Deformable Objects

- [**ClothesNet**](https://sites.google.com/view/clothesnet/): 
  This dataset consists of around 4400 models covering 11 categories
  annotated with clothes features, boundary lines, and key points.
  Further, the authors establish benchmark tasks for clothes perception, including classification, boundary line segmentation.
  [[Download]](https://docs.google.com/forms/d/e/1FAIpQLSdE-cUxWSzvC-D99RqkIHI9yLHjvT_5QygszjfqxnB6vIt8vw/viewform) [[Paper]](https://arxiv.org/abs/2308.09987)

- [**GarmentLab**](https://garmentlab.github.io/): 
  This environment provides physical simulations of garments in ClothesNet.
  Further, it provides benchmarking tasks of garment manipulation tasks,
  and models of real-world garments easily accessible globally for evaluation.
  [[Download]](https://garmentlab.readthedocs.io/en/latest/tutorial/data/index.html) [[Paper]](https://arxiv.org/abs/2411.01200) [[Code]](https://github.com/GarmentLab/GarmentLab)

## Multi-modal

- [**ObjectFolder**](https://objectfolder.stanford.edu/): 
  This dataset models the multisensory behaviors of real objects with 1) ObjectFolder 2.0, a dataset of 1,000 neural objects in the form of implicit neural representations with simulated multisensory data, and 2) ObjectFolder Real, a dataset that contains the multisensory measurements for 100 real-world household objects, building upon a newly designed pipeline for collecting the 3D meshes, videos, impact sounds, and tactile readings of real-world objects. It also contains a standard benchmark suite of 10 tasks for multisensory object-centric learning, centered around object recognition, reconstruction, and manipulation with sight, sound, and touch.  [[Download]](https://objectfolder.stanford.edu/objectfolder-real-download) [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Gao_The_ObjectFolder_Benchmark_Multisensory_Learning_With_Neural_and_Real_Objects_CVPR_2023_paper.html)