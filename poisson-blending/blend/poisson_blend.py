import os
import argparse
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("idx", type=int)
parser.add_argument("total", type=int)
args = parser.parse_args()


share_id = args.idx
share_total = args.total

jjj = 50 # the test-time optimization steps
total_num = 1000 # total number of output shapes


aligned_part_dir = "/local-scratch/results/chair/test_res_PWarp_overall_Epoch200_diffshapes_JS-128-120-0.002-0.0001"
original_part_dir = "/local-scratch/shapenet_part_seg/chair/part_mesh0050"
gen_joint_dir = aligned_part_dir


part_name_list = ["12_back", "13_seat", "14_leg", "15_arm"]
#part_names = ["1_handle", "2_body"]
#part_names = ["1_body", "2_wing", "3_tail", "4_engine"]





output_dir = "output_mesh"
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

for kkk in range(share_id,total_num,share_total):
	try:
		print("--------------",kkk)
		src_txt_dir = aligned_part_dir+"/"+str(kkk)+"_"+str(jjj)+"/"+str(kkk)+".name.txt"
		if not os.path.exists(src_txt_dir): continue
		src_txt = open(src_txt_dir)
		src_part_names = [tmp.strip() for tmp in src_txt.readlines()]
		src_txt.close()

		ply_save_dir = output_dir+"/"+str(kkk)+"_"+str(jjj)
		if not os.path.exists(ply_save_dir):
			os.makedirs(ply_save_dir)
		
		#read joint
		joint_dir = gen_joint_dir+"/"+str(kkk)+".out."+str(jjj)+".ply"
		joint_v, joint_t = read_ply_triangle(joint_dir)
		joint_p = compute_plane_for_each_triangle(joint_v, joint_t)
		joint_map_v2t = compute_mapping_vertex_to_triangles(joint_v, joint_t)
		joint_vn = compute_vertex_normal(joint_v, joint_t, joint_p, joint_map_v2t)

		#gather edges
		all_v = []
		all_t = []
		all_loop_vn = []
		all_loop_e = []
		all_v_count = 0
		all_v_count_list = [all_v_count]
		for j in range(len(part_name_list)):
			edge_vertices_dir = aligned_part_dir+"/"+str(kkk)+"_"+str(jjj)+"/"+part_name_list[j]+".ply"
			edge_edges_dir = original_part_dir+"/"+src_part_names[j]+"/"+part_name_list[j]+"_edge.ply"
			if os.path.exists(edge_vertices_dir) and os.path.exists(edge_edges_dir):
				v,t = read_ply_triangle(edge_vertices_dir)
				_,e = read_ply_edge(edge_edges_dir)

				#important: edges need to form loops
				loop_e, v_use_flag = find_loops(v,e)
				loop_vn = compute_vertex_normal_for_masked_vertices(v, t, v_use_flag)
				print(len(loop_e))
				#####write_ply_edge(ply_save_dir+"/"+part_name_list[j]+".edge.ply",v,e)
				#####write_ply_edgeloop(ply_save_dir+"/"+part_name_list[j]+".loop.ply",loop_vn,loop_e)

				all_v.append(v)
				all_t.append(t)
				all_loop_vn.append(loop_vn)
				all_loop_e.append(loop_e)
				all_v_count += len(v)
				all_v_count_list.append(all_v_count)

		corresponding_loop_e = []
		corresponding_vertices = []
		seam_t_all = []
		for j in range(len(all_v)):
				loop_vn = all_loop_vn[j]
				loop_e = all_loop_e[j]
				all_v_count = all_v_count_list[j]
				all_v_count_last = all_v_count_list[-1]

				#find correspondence
				cle, cvv, seam_t = find_loop_correspondence_and_fill_seam(loop_vn,loop_e,joint_vn,joint_t,joint_map_v2t)
				#####write_ply_edgeloop(ply_save_dir+"/"+part_name_list[j]+".cloop.ply", joint_v, cle)
				corresponding_loop_e += cle
				corresponding_vertices += cvv

				#correct vertex indices in seam triangles
				for ii in range(len(seam_t)):
					for jj in range(len(seam_t[ii])):
						for kk in range(3):
							if seam_t[ii][jj][kk]<0:
								seam_t[ii][jj][kk] = all_v_count_last-seam_t[ii][jj][kk]-1
							else:
								seam_t[ii][jj][kk] = all_v_count+seam_t[ii][jj][kk]
						if seam_t[ii][jj][1] != seam_t[ii][jj][2]:
							seam_t_all.append(seam_t[ii][jj])
		seam_t_all = np.array(seam_t_all,np.int32)


		#correct vertex indices in triangles
		for j in range(len(all_t)):
			all_t[j][:] = all_t[j][:] + all_v_count_list[j]


		#write combined
		all_v2 = all_v
		all_t2 = all_t
		all_v2 = np.concatenate(all_v2,axis=0)
		all_t2 = np.concatenate(all_t2,axis=0)
		write_ply_triangle(ply_save_dir+"/combined_nojoint.ply", all_v2, all_t2)

		#write combined
		all_v2 = all_v+[joint_v]
		all_t2 = all_t+[joint_t+all_v_count_list[-1]]
		all_v2 = np.concatenate(all_v2,axis=0)
		all_t2 = np.concatenate(all_t2,axis=0)
		write_ply_triangle(ply_save_dir+"/combined_beforeblend.ply", all_v2, all_t2)

		#filter_triangles
		joint_v, joint_t = filter_triangles(corresponding_loop_e, joint_v, joint_t, joint_map_v2t)
		if len(joint_t)==0: continue
		write_ply_triangle(ply_save_dir+"/joint.ply", joint_v, joint_t)


		#poisson blending 1
		joint_v1 = np.copy(joint_v)
		poisson_blending(joint_v1, joint_t, corresponding_loop_e, corresponding_vertices, force_correspondence=True)

		#write combined
		all_v2 = all_v+[joint_v1]
		all_t2 = all_t+[joint_t+all_v_count_list[-1],seam_t_all]
		all_v2 = np.concatenate(all_v2,axis=0)
		all_t2 = np.concatenate(all_t2,axis=0)
		write_ply_triangle(ply_save_dir+"/combined_blended.ply", all_v2, all_t2)


		#poisson blending 2
		joint_v2 = np.copy(joint_v)
		poisson_blending(joint_v2, joint_t, corresponding_loop_e, corresponding_vertices, force_correspondence=False)

		#write combined
		all_v2 = all_v+[joint_v2]
		all_t2 = all_t+[joint_t+all_v_count_list[-1],seam_t_all]
		all_v2 = np.concatenate(all_v2,axis=0)
		all_t2 = np.concatenate(all_t2,axis=0)
		write_ply_triangle(ply_save_dir+"/combined_blended_2.ply", all_v2, all_t2)

	except:
		print("ERROR!!!")
