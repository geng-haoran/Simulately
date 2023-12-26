---
sidebar_position: 6
title: Dataset
---

# Dataset for Robotics

### Object Shapes

- [**ShapeNet**](https://shapenet.org/): ShapeNet is an ongoing effort to establish a richly-annotated, large-scale
  dataset of 3D shapes. We provide researchers around the world with this data to enable research in computer graphics,
  computer vision, robotics, and other related
  disciplines. [[Download]](https://shapenet.org/login/) [[Paper]](https://arxiv.org/abs/1512.03012)
  
- [**FineSeg**](http://kevinkaixu.net/projects/shape2motion.html) Shape2Motion paper release FineSeg dataset, which
  contains about 3000 3D shapes over six shape categories: chair (1000), table (500), airplanes (600), sofa (600),
  helicopter (100) and bike (140). The models are collected from a subset of
  ShapeNet.[[Download]](https://drive.google.com/file/d/1ZtWgMqYSNl1MSXKaTnMdN6xHXme9AjXb/view?usp=sharing) [[Paper]](https://arxiv.org/abs/1903.00709) [[Code]](https://github.com/wangxiaogang866/Shape2Motion)

- [**PartNet**](https://partnet.cs.stanford.edu/): PartNet: a consistent, large-scale dataset of 3D objects annotated
  with fine-grained, instance-level, and hierarchical 3D part
  information. [[Download]](https://www.shapenet.org/login/) [[Code]](https://github.com/daerduoCarey/partnet_dataset) [[Paper]](https://arxiv.org/abs/1812.02713)
- [**YCB**](https://www.ycbbenchmarks.com/): YCB Object and Model Set is designed for facilitating benchmarking in
  robotic manipulation. The set consists of objects of daily life with different shapes, sizes, textures, weight and
  rigidity, as well as some widely used manipulation
  tests. [[Download]](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/); [[Paper]](https://journals.sagepub.com/doi/full/10.1177/0278364917700714)
- [**OmniObject3D**](https://omniobject3d.github.io/): OmniObject3D, a large vocabulary 3D object dataset with massive
  high-quality real-scanned 3D objects to facilitate the development of 3D perception, reconstruction, and generation in
  the real
  world. [[Download]](https://opendatalab.com/OpenXDLab/OmniObject3D-New/tree/main) [[Paper]](https://arxiv.org/abs/2301.07525) [[Code]](https://github.com/omniobject3d/OmniObject3D/tree/main)
- [**Objaverse-XL**](https://objaverse.allenai.org/): A Universe of 10M+ 3D Objects. Objaverse-XL is 12x larger than
  Objaverse 1.0 and 100x larger than all other 3D datasets
  combined. [[Download]](https://docs.google.com/forms/d/e/1FAIpQLScNOWKTHk3a7CGiegNjROFNfOcpzr5gt6G0FMEMQ8qXRTbs0Q/viewform) [[Paper]](https://arxiv.org/abs/2307.05663) [[Code]](https://github.com/allenai/objaverse-xl)

### Articulated Objects

- [**PartNet Moblity**](https://sapien.ucsd.edu/browse): SAPIEN releases PartNet-Mobility dataset, which is a collection
  of 2K articulated objects with motion annotations and rendernig material. The dataset powers research for
  generalizable computer vision and
  manipulation. [[Download]](https://sapien.ucsd.edu/downloads) [[Paper]](https://arxiv.org/abs/2003.08515)

- [**AKB48**](https://liuliu66.github.io/articulationobjects/index.html): This is a realistic and physics-rich object
  repository for articulation analysis. It enables various robotic vision and interaction tasks that require detailed
  part-level understanding. More categories and download link will be released
  soon. [[Download]](https://liuliu66.github.io/articulationobjects/download.html) [[Paper]](https://arxiv.org/abs/2202.08432)

- [**GAPartNet**](https://pku-epic.github.io/GAPartNet/) (Merge and Provide more annotation from PartNet Mobility and
  AKB48): By identifying and defining 9 GAPart classes (lids, handles, etc.) in 27 object categories, we construct a
  large-scale part-centric interactive dataset, GAPartNet, where we provide rich, part-level annotations (semantics,
  poses). [[Download]](https://forms.gle/3qzv8z5vP2BT5ARN7) [[Paper]](https://arxiv.org/abs/2211.05272) [[Code]](https://github.com/PKU-EPIC/GAPartNet)

### 3D Scene Datasets

|                                                              Dataset                                                               |                       Type                        |                                                             Scale                                                              |                                                             Object / Scene<br />Annotations                                                              |                                                                                        Note                                                                                        |
|:----------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|            **[ProcTHOR](https://github.com/allenai/procthor), [ProcTHOR-10k](https://github.com/allenai/procthor-10k)**            |  Interior scenes<br />CAD models<br />Synthetic   | 10K<br />with diverse multi-room layouts<br />1,633 interactive household objects across 108 categories, with 3,278 materials. |                                                                                                                                                          |          Compatible with AI2-THOR.<br />Meshes can be extracted via Unity with [YandanYang/ai2thor](https://github.com/YandanYang/ai2thor) for usage in other simulators.          |
|                                 **[Replica](https://github.com/facebookresearch/Replica-Dataset)**                                 |        Interior scenes<br />Scanned meshes        |                                   18 in initial release<br />with a few single-room layouts                                    |                                                          3D Sematic segmentation;<br />Navmesh.                                                          |                                                                                                                                                                                    |
|                              **[Aria Ditgital Twin Dataset**](https://www.projectaria.com/datasets/adt/)**                               | Interior scenes<br />Digital twin with CAD models |                               398 objects (324 stationary, 74 dynamic)<br />2 real indoor scenes                               | FPV activities within 200 RGB-D sequences(~400 mins),<br />with device trajectory, semantic segmentation, 2D/3D bounding boxes, skeleton, and eye gaze). |                                                                                                                                                                                    |
|                            **[Aria Synthetic Environments](https://www.projectaria.com/datasets/ase/)**                            |  Interior scenes<br />CAD models<br />Synthetic   |                                                10K scenes<br />~8000 3D objects                                                |                                 Simulated cam RGBD with semantics segmentation, and 2D bounding box;<br />3D floor plan.                                 |                                                                                                                                                                                    |
|                                        **[HyperSim](https://github.com/apple/ml-hypersim)**                                        |       Interior scenes<br /><br />CAD models       |                                                           461 scenes                                                           |                                         ~77K rendered images with<br />camera trajectory, and 3D bounding boxes.                                         | Built upon some bundles of Evermotion interior data, available [here](https://www.turbosquid.com/Search/3D-Models?include_artist=evermotion).<br />Comes with the Hypersim Toolkit |
|                       **[3D Front](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)**                       |       Interior scenes<br /><br />CAD models       |                                                    18K rooms, 7K furnitures                                                    |                                                                                                                                                          |                                                  Comes with a rendering tool called "Trescope" for 2D rendering with annotations.                                                  |
| **[Matterport 3D](https://niessner.github.io/Matterport/)**<br />**[Habitat-Matterport 3D](https://aihabitat.org/datasets/hm3d/)** | Interior scenes<br />Scanned meshes / CAD models  |                   M3D: 10,800 panoramic views from 90 scanned scenes<br />HM3D: 1,000 scenes/digital twins.                    |                                                           Semantic and instance segmentation.                                                            |                                                                                                                                                                                    |
