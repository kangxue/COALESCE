In this folder we provide the code for sampling points for training our networks.  Note that you don't need to run code in this folder to test our method. The processed dataset used in the paper has been provided in this <a href="https://drive.google.com/u/0/uc?id=1htY0dARRDrOid4gjPHkWtzZNVvbP_rqo&export=download">LINK</a>.


If you need to process your own dataset, you can follow the following steps.

# step 0


- Download surface point samples of shapeNet meshes <a href="https://drive.google.com/file/d/1dAFfZcIOcjhO_PwS1F5KAaPsDvCovIfw/view?usp=sharing">here</a>,  strore them under "../densePointCloud".  You can use Poisson disk sampling to get similar samples for your own meshes, and store them in same way.

- download the shapenet part dataset <a href="https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip">here</a>, extract the data, and store it under the folder "../yi2016"

- downlaod the 256^3 voxel dataset <a href="https://drive.google.com/file/d/1xtJz5CEEPgYOtWP6Dr6nUWbUXPDMswh0/view?usp=sharing">here</a> (link originally from <a href="https://github.com/chaene/hsp">HSP</a>), extract the folder "modelBlockedVoxels256",  and store it under "../modelBlockedVoxels256"



# step 1

Get segmented and eroded input point cloud with the script
``` 1_extract_parts2048_ply_erodedX_withnormal.py ```

# step 2

Sample query point samples for implicit function with this script:
```2_point_sampling_for_joint_synsthsis_256.py```

# step 3

In the above script, you can use the two arguments '--pstart', '--pend' to specify  the starting and ending position in the processing list, so that the data processing can be parallelized.   This will give you result folders such as "03001627_sampling_erode0.025_256_0_500", 
"03001627_sampling_erode0.025_256_500_1000", ... 

You can use the script ```3_merge_subset.py``` to merge them into one single hdf5.


# step 4

run ```4_sort_points.py``` to sort point clouds stored in hdf5 according to their distance to part boundaries.


# Part mesh processing

the code is provided under ```../poisson-blending```