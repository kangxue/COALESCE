import os
import numpy as np 
import math
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("idx", type=int)
parser.add_argument("total", type=int)
args = parser.parse_args()


predefined_category = "chair" # "chair", "mug", "airplane"

if predefined_category=="chair":
    in_obj_dir = "../reoriented-03797390-obj/" #the folder storing the meshes
    ref_txt_dir = "../03001627_vox.txt" #the list storing the names of the shapes
    ref_ply_dir = "../03001627_parts2048_ply_eroded0.05_withnormal/" #the folder storing the ground truth point cloud segmentations
    out_part_dir = "../part_mesh0050/" #the output folder

    #12 Chair 03001627 1 back
    #13 Chair 03001627 2 seat
    #14 Chair 03001627 3 leg
    #15 Chair 03001627 4 arm
    part_names = ["12_back", "13_seat", "14_leg", "15_arm"]
    part_list = [1,2,3,4]
    part_num = len(part_list)

if predefined_category=="mug":
    in_obj_dir = "../reoriented-03797390-obj/"
    ref_txt_dir = "../03797390_vox.txt"
    ref_ply_dir = "../03797390_parts2048_ply_eroded0.05_withnormal/"
    out_part_dir = "../part_mesh0050/"

    #36 Mug 03797390 1 handle
    #37 Mug 03797390 2 body
    part_names = ["1_handle", "2_body"]
    part_list = [1,2]
    part_num = len(part_list)
    
if predefined_category=="airplane":
    in_obj_dir = "../reoriented-02691156-obj/"
    ref_txt_dir = "../02691156_vox.txt"
    ref_ply_dir = "../02691156_parts2048_ply_eroded0.025_withnormal/"
    out_part_dir = "../part_mesh0025/"

    #0 Airplane 02691156 1 body
    #1 Airplane 02691156 2 wing
    #2 Airplane 02691156 3 tail
    #3 Airplane 02691156 4 engine
    part_names = ["1_body", "2_wing", "3_tail", "4_engine"]
    part_list = [1,2,3,4]
    part_num = len(part_list)


#read the names of the shapes
txt_list_ = open(ref_txt_dir)
txt_list = txt_list_.readlines()
txt_list_.close()
txt_list = [name.strip() for name in txt_list]
txt_list = sorted(txt_list)
txt_list = txt_list[int(len(txt_list)*0.8):] #only testing shapes (20%) are processed



idx = args.idx
total = args.total

for kkk in range(idx,len(txt_list),total): #this loop processes each shape individually
    print(kkk,txt_list[kkk])
    
    obj_in_name = in_obj_dir+txt_list[kkk]+".reoriented.obj"
    ref_txt_name = ref_ply_dir+txt_list[kkk]
    vertices, triangles = load_obj(obj_in_name)
    part_points = read_ply_and_labels(ref_txt_name, part_list)


    vertices = list(vertices)
    triangles = triangles.tolist()
    label_vertices = np.full([len(vertices)],-1,np.int32)
    label_triangles_list = np.full([part_num,len(triangles)],-1,np.int32)

    #now we have:
    #vertices, triangles: the vertices and triangles of the mesh.
    #part_points: the points and labels of the ground truth point cloud segmentation of that shape.

    #this function below will use part_points to label vertices and triangles -> label_vertices and label_triangles_list.
    #each mesh vertex will receive one part label, by finding the nearest neighbor in part_points.
    #each mesh triangle may receive one or more part labels.
    #for each mesh triangle, we use its 3 vertex points and we also sample some points on the triangle.
    #then we use nearest-neighbor to label each point, and label the triangle according to the labels of those points.
    #therefore the triangle may belong to one or multiple parts, depending on the labels of those points.
    get_face_color_multiple(vertices, triangles, part_points, label_vertices, label_triangles_list)

    obj_dir = out_part_dir+txt_list[kkk]
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)


    for i in range(part_num): #now segment each part from the mesh individually
        #obtain vertices and triangles containing the part (part #i)
        part_vertices_, part_triangles_ = collect_part_face(vertices, triangles, label_vertices, label_triangles_list[i], i)
        if len(part_triangles_)==0: continue

        part_vertices_ = list(part_vertices_)
        part_triangles_ = part_triangles_.tolist()
        #we initialize the labels below with "-1", meaning "unknown".
        #The labels will be filled gradually in the subdivision below, but some "-1" may remain for triangles that need to be subdivided.
        label_part_vertices = [-1]*len(part_vertices_)
        label_part_vertices_out = [-1]*len(part_vertices_)
        label_part_vertices_in = [-1]*len(part_vertices_)
        label_part_triangles = [-1]*len(part_triangles_)

        #the while loop below will subdivide the part mesh at most 5 times and stop when the number of triangles is 40000 or more.
        #note that only triangles close to the erosion boundary is subdivided.
        #The erosion radius is hard-coded in utils.py, explained below. You will need to change it if you prefer a different value.
        #erode_threshold: the erosion radius
        #erode_threshold_in: the radius used to determine which triangles to subdivide. it should be smaller than erode_threshold.
        #erode_threshold_out: the radius used to determine which triangles to subdivide. it should be larger than erode_threshold.
        #triangles that have vertices falling into [erode_threshold_in,erode_threshold_out) will be subdivided.
        itrc = 0
        while itrc<5 and len(part_triangles_)<40000:
            #fill in label_part_vertices, label_part_vertices_out, label_part_vertices_in, label_part_triangles
            get_face_color(part_vertices_, part_triangles_, part_points, label_part_vertices, label_part_vertices_out, label_part_vertices_in, label_part_triangles)
            print("part-"+part_names[i])
            print("subdiv-"+str(itrc))
            print("before:", len(part_vertices_), len(part_triangles_))
            #subdivide the part mesh using midpoint. only triangles close to the erosion boundary is subdivided.
            adaptive_subdiv_according_to_color_mid(part_vertices_, part_triangles_, part_points, label_part_vertices, label_part_vertices_out, label_part_vertices_in, label_part_triangles)
            print("after:", len(part_vertices_), len(part_triangles_))
            itrc += 1
        
        #fill in label_part_vertices and label_part_triangles before the final subdivision
        label_part_triangles = [-1]*len(part_triangles_)
        get_face_color_no_tolerance(part_vertices_, part_triangles_, part_points, label_part_vertices, label_part_triangles)

        print("subdiv-final")
        print("before:", len(part_vertices_), len(part_triangles_))
        #final subdivision. instead of midpoint, we use the actual erosion boundary point to be more accurate
        adaptive_subdiv_according_to_color_boundary(part_vertices_, part_triangles_, part_points, label_part_vertices, label_part_triangles)
        print("after:", len(part_vertices_), len(part_triangles_))
        
        #fill in label_part_vertices and label_part_triangles. this time, "-1" is not allowed.
        force_assign_face_color(part_vertices_, part_triangles_, part_points, label_part_vertices, label_part_triangles)

        #collect vertices, triangles, and erosion-boundary edges of the eroded part. useless vertices and triangles are removed.
        part_vertices, part_triangles, part_edges = collect_part_face_and_edge(part_vertices_, part_triangles_, label_part_vertices, label_part_triangles, i)
        if len(part_triangles)>0:
            part_dir = obj_dir+"/"+part_names[i]+".ply"
            write_ply_triangle(part_dir, part_vertices, part_triangles)
            part_edge_dir = obj_dir+"/"+part_names[i]+"_edge.ply"
            write_ply_edge(part_edge_dir, part_vertices, part_edges)
            
            #find loops
            #loop_e, _ = find_loops(part_vertices,part_edges)
            #part_loop_dir = obj_dir+"/"+part_names[i]+"_loop.ply"
            #write_ply_edgeloop(part_loop_dir,part_vertices,loop_e)



