# TAIP
Trunk-Axis-Inversion-Based Partitioning for Individual Tree Segmentation
# TAIP: Trunk-Axis-Inversion-Based Partitioning for Individual Tree Segmentation

## Overview

TAIP (Trunk-Axis-Inversion-Based Partitioning) is an individual tree segmentation method developed for forest point clouds. It integrates point cloud inversion (PCI), trunk-axis extraction, spatial axis grouping, and trunk-axis-guided partitioning to identify individual trees.

By exploiting the geometric characteristics and spatial continuity of tree trunks, TAIP aims to improve individual tree segmentation in structurally complex forest environments, particularly under conditions of understory vegetation and overlapping tree crowns.

## Methodology

The TAIP workflow consists of the following steps:

**1. Point Cloud Inversion (PCI)**

The input normalized point cloud is voxelized, and PCI is applied to enhance trunk-related structures by exploiting the spatial distribution of occupied and empty voxels.

**2. Candidate Trunk-Axis Extraction**

Three constraints, including vertical orientation, structural continuity, and local point density, are integrated to identify candidate trunk axes. Dynamic programming-based shortest-path extraction is performed on the PCI-enhanced trunk-only point cloud.

**3. Trunk-Axis Grouping**

A convex hull-based procedure is employed to identify and merge fragmented candidate axes belonging to the same tree, generating grouped trunk-axis representations.

**4. Trunk-Axis-Guided Individual Tree Segmentation**

The grouped trunk axes are transferred to the original normalized point cloud. A trunk-axis-guided partitioning strategy, incorporating a DBH-normalized distance criterion, is applied to assign points to individual trees.

**5. Output**

The final results are exported as individually segmented tree point clouds in LAS format.


## Citation

If you use TAIP in your research, please cite the associated publication:

[Publication information to be added.]
