# COALESCE-poisson-blending
Poisson blending code for paper COALESCE: Component Assembly by Learning to Synthesize Connections.

## Requirements
Python 3.6 with numpy and scipy.

## 1. Data preparation

Create a folder and put the following folders/files inside:
- *prepare*: copy the *prepare* folder in this repo.
- *<ref_txt_dir>*: a list of object names for pre-processing. See the text files in *prepare* folder for some examples.
- *<in_obj_dir>*: a folder containing normalized and re-oriented shapes in obj format.
- *<ref_ply_dir>*: a folder containing segmented and noise-removed point clouds and labels.


Open *prepare/get_part_mesh.py* and modify the directories to point to the above folders/files. Also change *part_names* and *part_list* if you are not using the predefined categories.


Open *prepare/utils.py* and modify the erosion radius. The default radius is 0.05:
```
erode_threshold = 0.05
erode_threshold_in = 0.04
erode_threshold_out = 0.06
```
*erode_threshold* is the erosion radius, *erode_threshold_in* and *erode_threshold_out* define two boundaries for mesh subdivision. If you want to change the radius to 0.025:
```
erode_threshold = 0.025
erode_threshold_in = 0.015
erode_threshold_out = 0.035
```


Next, go to the *prepare* folder and run the code. Since the code is written in python, the execution is rather slow. We recommend using multiple processes:
```
python get_part_mesh.py <process_id> <total_num_of_processes>
```
for instance, open 4 terminals and run one of the following commands in each terminal:
```
python get_part_mesh.py 0 4
python get_part_mesh.py 1 4
python get_part_mesh.py 2 4
python get_part_mesh.py 3 4
```
The output part meshes and edges are written to a folder *<out_part_dir>*.

## 2. Poisson blending

After running the network, the aligned parts and synthesized joints are written to a folder *<aligned_part_dir>*. The data preparation step also produces a folder *<out_part_dir>*.

Create a folder and copy the *blend* folder inside.

Open *blend/poisson_blend.py* and modify the directories to point to the above *<aligned_part_dir>* and *<out_part_dir>*. Change the *part_name_list* according to the category.

Next, go to the *blend* folder and run the code. Since the code is written in python, the execution is rather slow. We recommend using multiple processes:
```
python poisson_blend.py <process_id> <total_num_of_processes>
```
for instance, open 4 terminals and run one of the following commands in each terminal:
```
python poisson_blend.py 0 4
python poisson_blend.py 1 4
python poisson_blend.py 2 4
python poisson_blend.py 3 4
```
The outputs are written to a folder *<output_dir>*.

For each shape, there will be several outputs:
- *combined_nojoint.ply*: combining aligned parts without the synthesized joint.
- *combined_beforeblend.ply*: combining aligned parts with the synthesized joint, but before Poisson blending.
- *joint.ply*: the synthesized joint after removing redundant portions.
- *combined_blended.ply*: the blended result which blends both open and close boundaries.
- *combined_blended_2.ply*: the blended result which blends only close boundaries.






