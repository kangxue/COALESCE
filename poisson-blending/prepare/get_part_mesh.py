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
    in_obj_dir = "../reoriented-03797390-obj/"
    ref_txt_dir = "../03001627_vox.txt"
    ref_ply_dir = "../03001627_parts2048_ply_eroded0.05_withnormal/"
    out_part_dir = "../part_mesh0050/"

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



txt_list_ = open(ref_txt_dir)
txt_list = txt_list_.readlines()
txt_list_.close()
txt_list = [name.strip() for name in txt_list]
txt_list = sorted(txt_list)
txt_list = txt_list[int(len(txt_list)*0.8):]



idx = args.idx
total = args.total

for kkk in range(idx,len(txt_list),total):
    print(kkk,txt_list[kkk])
    
    obj_in_name = in_obj_dir+txt_list[kkk]+".reoriented.obj"
    ref_txt_name = ref_ply_dir+txt_list[kkk]
    vertices, triangles = load_obj(obj_in_name)
    part_points = read_ply_and_labels(ref_txt_name, part_list)


    vertices = list(vertices)
    triangles = triangles.tolist()
    label_vertices = np.full([len(vertices)],-1,np.int32)
    label_triangles_list = np.full([part_num,len(triangles)],-1,np.int32)


    get_face_color_multiple(vertices, triangles, part_points, label_vertices, label_triangles_list)

    obj_dir = out_part_dir+txt_list[kkk]
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)


    for i in range(part_num):
        part_vertices_, part_triangles_ = collect_part_face(vertices, triangles, label_vertices, label_triangles_list[i], i)
        if len(part_triangles_)==0: continue

        part_vertices_ = list(part_vertices_)
        part_triangles_ = part_triangles_.tolist()
        label_part_vertices = [-1]*len(part_vertices_)
        label_part_vertices_out = [-1]*len(part_vertices_)
        label_part_vertices_in = [-1]*len(part_vertices_)
        label_part_triangles = [-1]*len(part_triangles_)

        itrc = 0
        while itrc<5 and len(part_triangles_)<40000:
            get_face_color(part_vertices_, part_triangles_, part_points, label_part_vertices, label_part_vertices_out, label_part_vertices_in, label_part_triangles)
            print("part-"+part_names[i])
            print("subdiv-"+str(itrc))
            print("before:", len(part_vertices_), len(part_triangles_))
            adaptive_subdiv_according_to_color_mid(part_vertices_, part_triangles_, part_points, label_part_vertices, label_part_vertices_out, label_part_vertices_in, label_part_triangles)
            print("after:", len(part_vertices_), len(part_triangles_))
            itrc += 1
        
        label_part_triangles = [-1]*len(part_triangles_)
        get_face_color_no_tolerance(part_vertices_, part_triangles_, part_points, label_part_vertices, label_part_triangles)

        print("subdiv-final")
        print("before:", len(part_vertices_), len(part_triangles_))
        adaptive_subdiv_according_to_color_boundary(part_vertices_, part_triangles_, part_points, label_part_vertices, label_part_triangles)
        print("after:", len(part_vertices_), len(part_triangles_))
        
        force_assign_face_color(part_vertices_, part_triangles_, part_points, label_part_vertices, label_part_triangles)


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



