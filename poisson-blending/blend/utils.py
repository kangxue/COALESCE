import numpy as np 
import math
import os

from scipy.sparse import csr_matrix,csc_matrix
from scipy.sparse.linalg import lsqr


def read_ply_point_normal(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()

    start = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
        start += 1

    if vertex_num==0: return []
    vertices_normals = np.zeros([vertex_num,6], np.float32)
    for i in range(vertex_num):
        line = lines[i+start].split()
        vertices_normals[i,0] = float(line[0]) #X
        vertices_normals[i,1] = float(line[1]) #Y
        vertices_normals[i,2] = float(line[2]) #Z
        vertices_normals[i,3] = float(line[3]) #normalX
        vertices_normals[i,4] = float(line[4]) #normalY
        vertices_normals[i,5] = float(line[5]) #normalZ
    return vertices_normals

def read_ply_triangle(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()
    vertices = []
    triangles = []

    start = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
            if line[1] == "face":
                face_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num,3], np.float32)
    triangles = np.zeros([face_num,3], np.int32)

    for i in range(vertex_num):
        line = lines[start].split()
        vertices[i,0] = float(line[0])
        vertices[i,1] = float(line[1])
        vertices[i,2] = float(line[2])
        start += 1

    for i in range(face_num):
        line = lines[start].split()
        triangles[i,0] = int(line[1])
        triangles[i,1] = int(line[2])
        triangles[i,2] = int(line[3])
        start += 1

    return vertices, triangles


def read_ply_edge(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()
    vertices = []
    edges = []

    start = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
            if line[1] == "edge":
                edge_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num,3], np.float32)
    edges = np.zeros([edge_num,2], np.int32)

    for i in range(vertex_num):
        line = lines[start].split()
        vertices[i,0] = float(line[0])
        vertices[i,1] = float(line[1])
        vertices[i,2] = float(line[2])
        start += 1

    for i in range(edge_num):
        line = lines[start].split()
        edges[i,0] = int(line[0])
        edges[i,1] = int(line[1])
        start += 1

    return vertices, edges


def write_obj(dire, vertices, triangles, flip=False, switch=False):
    v = np.array(vertices, np.float32)
    t = np.array(triangles, np.int32)
    vertices = v
    triangles = t
    if flip:
        v2 = np.copy(v)
        v2[:,0] = -v[:,0]
        t2 = np.copy(t)
        t2[:,1] = t[:,2]
        t2[:,2] = t[:,1]
        vertices = v2
        triangles = t2
    if switch:
        v2 = np.copy(v)
        v2[:,0] = -v[:,2]
        v2[:,2] = v[:,0]
        vertices = v2
        triangles = t
    fout = open(dire, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(triangles[ii,0]+1)+" "+str(triangles[ii,1]+1)+" "+str(triangles[ii,2]+1)+"\n")
    fout.close()


def write_ply_triangle(dire, vertices, triangles):
    fout = open(dire, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(triangles))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    
    for i in range(len(vertices)):
        fout.write(str(vertices[i][0])+" "+str(vertices[i][1])+" "+str(vertices[i][2])+"\n")
    
    for i in range(len(triangles)):
        fout.write("3 "+str(triangles[i][0])+" "+str(triangles[i][1])+" "+str(triangles[i][2])+"\n")
    
    fout.close()


def write_ply_point_normal(name, vertices, normals=None):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
    else:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
    fout.close()


def write_ply_edge(dire, vertices, edges):
    fout = open(dire, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element edge "+str(len(edges))+"\n")
    fout.write("property int vertex1\n")
    fout.write("property int vertex2\n")
    fout.write("end_header\n")
    
    for i in range(len(vertices)):
        fout.write(str(vertices[i][0])+" "+str(vertices[i][1])+" "+str(vertices[i][2])+"\n")
    
    for i in range(len(edges)):
        fout.write(str(edges[i][0])+" "+str(edges[i][1])+"\n")
    
    fout.close()


def write_ply_edgeloop(dire, vertices, loops):
    edge_num = 0
    for i in range(len(loops)):
        if loops[i][-1]<0: continue
        edge_num += len(loops[i])
    fout = open(dire, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element edge "+str(edge_num)+"\n")
    fout.write("property int vertex1\n")
    fout.write("property int vertex2\n")
    fout.write("end_header\n")
    
    for i in range(len(vertices)):
        fout.write(str(vertices[i][0])+" "+str(vertices[i][1])+" "+str(vertices[i][2])+"\n")
    
    for i in range(len(loops)):
        if loops[i][-1]<0: continue
        for j in range(len(loops[i])):
            fout.write(str(loops[i][j])+" "+str(loops[i][(j+1)%len(loops[i])])+"\n")
    
    fout.close()



def sqdist(p1,p2):
    return np.sum(np.square(p1-p2))
def midpoint(p1,p2):
    return (p1+p2)/2
def min_dist2(p1,plist2):
    if len(plist2)==0: return 666666.666
    return np.min(np.sum(np.square(p1-plist2),axis=1))



def find_loops(vertices, edges):
    vertices = np.array(vertices,np.float32)
    edges = np.array(edges,np.int32)
    merge_threshold = 1e-4

    #first, remove non-edge points
    print("merging")
    edge_use_flag = np.zeros([len(vertices)], np.uint8)
    for i in range(len(edges)):
        edge_use_flag[edges[i,0]]=1
        edge_use_flag[edges[i,1]]=1
    tmp_mapping = edge_use_flag.nonzero()[0]
    tmp_vertices = vertices[tmp_mapping]
    tmp_vertices_len = len(tmp_vertices)

    #second, merge same vertex
    loop_tv = []
    inverse_mapping = []
    loop_te = []
    mapping = np.zeros([len(vertices)], np.int32)
    use_flag = np.zeros([len(vertices)], np.uint8)
    counter=0
    for i in range(tmp_vertices_len):
        if i==0:
            mapping[tmp_mapping[i]]=counter
            counter += 1
            use_flag[tmp_mapping[i]]=1
            continue
        tmp_within = np.sum(np.abs(tmp_vertices[i]-tmp_vertices[:i]),axis=1)<merge_threshold
        max_idx = np.argmax(tmp_within)
        if tmp_within[max_idx]:
            mapping[tmp_mapping[i]]=mapping[tmp_mapping[max_idx]]
        else:
            mapping[tmp_mapping[i]]=counter
            counter += 1
            use_flag[tmp_mapping[i]]=1
    for i in range(len(vertices)):
        if use_flag[i]:
            loop_tv.append(vertices[i])
            inverse_mapping.append(i)
    for i in range(len(edges)):
        e0 = mapping[edges[i,0]]
        e1 = mapping[edges[i,1]]
        if e0!=e1:
            loop_te.append([e0,e1])

    print("merging - end")
    print("finding loop")
    #find loops
    loop_le = []
    prev_vertex = np.full([len(loop_tv)], -1, np.int32)
    next_vertex = np.full([len(loop_tv)], -1, np.int32)
    vertex_used_flag = np.zeros([len(loop_tv)], np.uint8)
    for i in range(len(loop_te)):
        if loop_te[i][0]!=loop_te[i][1]:
            prev_vertex[loop_te[i][1]] = loop_te[i][0]
            next_vertex[loop_te[i][0]] = loop_te[i][1]

    while(np.min(vertex_used_flag)==0):
        target_v = -1
        #find a start
        for i in range(len(loop_tv)):
            if vertex_used_flag[i]==0:
                if prev_vertex[i]<0 and next_vertex[i]<0:
                    vertex_used_flag[i]=1
                else:
                    target_v = i
                    break
        if target_v<0: break
        vertex_used_flag[target_v]=1
        #find origin
        prev_v_list = [target_v]
        prev_v = target_v
        while prev_vertex[prev_v]>=0 and prev_vertex[prev_v] not in prev_v_list:
            prev_v = prev_vertex[prev_v]
            vertex_used_flag[prev_v]=1
            prev_v_list.append(prev_v)

        #get loop
        next_v_list = prev_v_list[::-1]
        if prev_vertex[next_v_list[0]]<0:
            target_v = next_v_list[-1]
            next_v = target_v
            while next_vertex[next_v]>=0 and next_vertex[next_v] not in next_v_list:
                next_v = next_vertex[next_v]
                vertex_used_flag[next_v]=1
                next_v_list.append(next_v)
            if next_vertex[next_v]<0:
                next_v_list.append(-1)
            else:
                if next_vertex[next_v]==next_v_list[-2]:
                    next_v_list.append(-1)
                else:
                    next_v_list = next_v_list[next_v_list.index(next_vertex[next_v]):]
        else:
            loop_le.append(next_v_list[:next_v_list.index(prev_vertex[next_v_list[0]])+1])
            next_v_list = next_v_list[next_v_list.index(prev_vertex[next_v_list[0]]):]
            target_v = next_v_list[-1]
            next_v = target_v
            while next_vertex[next_v]>=0 and next_vertex[next_v] not in next_v_list:
                next_v = next_vertex[next_v]
                vertex_used_flag[next_v]=1
                next_v_list.append(next_v)
            if next_vertex[next_v]<0:
                next_v_list.append(-1)
            else:
                if next_vertex[next_v]==next_v_list[-2]:
                    next_v_list.append(-1)
                else:
                    next_v_list = next_v_list[next_v_list.index(next_vertex[next_v]):]

        loop_le.append(next_v_list)


    print('loop -> lines', len(loop_le))

    #close loops
    loop_fe = []
    loop_le_used_flag = np.zeros([len(loop_le)], np.uint8)
    for i in range(len(loop_le)):
        if loop_le[i][-1]>=0:
            loop_fe.append(loop_le[i])
            loop_le_used_flag[i]=1

    print('loop -> closed ', len(loop_fe))




    distance_threshold = 0.02
    distance_threshold2 = distance_threshold*distance_threshold
    for i in range(len(loop_le)):
        if loop_le_used_flag[i]==0:
            tmp_used_flag = np.copy(loop_le_used_flag)
            tmp_used_flag[i]=1
            newloop_idx = [i]
            newloop = loop_le[i][:-1]
            newloop_ordercount = len(newloop)
            newloop_starti = newloop[0]
            newloop_start = loop_tv[newloop_starti]
            newloop_endi = newloop[-1]
            newloop_end = loop_tv[newloop_endi]
            prev_seq_headi = newloop_starti
            prev_seq_head = loop_tv[prev_seq_headi]
            prev_endi = newloop_endi
            looped_flag = False
            while True:
                nearest_id = -1
                nearest_dist2 = 666666.666
                for j in range(len(loop_le)):
                    if tmp_used_flag[j]==0:
                        tmp_vi = loop_le[j][0]
                        tmp_v = loop_tv[tmp_vi]
                        tmp_dist2 = np.sum(np.square(newloop_end-tmp_v))
                        if tmp_dist2<nearest_dist2:
                            nearest_id = j
                            nearest_dist2 = tmp_dist2
                if nearest_dist2<distance_threshold2:
                    tmp_used_flag[nearest_id]=1
                    newloop_idx.append(nearest_id)
                    newloop = newloop + loop_le[nearest_id][:-1]
                    newloop_ordercount += len(loop_le[nearest_id])-1
                    newloop_endi = loop_le[nearest_id][-2]
                    newloop_end = loop_tv[newloop_endi]
                    prev_seq_headi = loop_le[nearest_id][0]
                    prev_seq_head = loop_tv[prev_seq_headi]

                tmp_dist2 = np.sum(np.square(newloop_end-newloop_start))
                if tmp_dist2<distance_threshold2:
                    looped_flag = True
                    break
                if prev_endi==newloop_endi:
                    break
                prev_endi=newloop_endi
            
            if looped_flag:
                loop_fe.append(newloop)
                loop_le_used_flag = tmp_used_flag

    print('loop + found ', len(loop_fe))




    distance_threshold = 0.04
    distance_threshold2 = distance_threshold*distance_threshold
    for i in range(len(loop_le)):
        if loop_le_used_flag[i]==0:
            tmp_used_flag = np.copy(loop_le_used_flag)
            tmp_used_flag[i]=1
            newloop_idx = [i]
            newloop = loop_le[i][:-1]
            newloop_ordercount = len(newloop)
            newloop_starti = newloop[0]
            newloop_start = loop_tv[newloop_starti]
            newloop_endi = newloop[-1]
            newloop_end = loop_tv[newloop_endi]
            prev_seq_headi = newloop_starti
            prev_seq_head = loop_tv[prev_seq_headi]
            prev_endi = newloop_endi
            looped_flag = False
            while True:
                nearest_id = -1
                nearest_dist2 = 666666.666
                for j in range(len(loop_le)):
                    if tmp_used_flag[j]==0:
                        tmp_vi = loop_le[j][0]
                        tmp_v = loop_tv[tmp_vi]
                        tmp_dist2 = np.sum(np.square(newloop_end-tmp_v))
                        if tmp_dist2<nearest_dist2:
                            nearest_id = j
                            nearest_dist2 = tmp_dist2
                if nearest_dist2<distance_threshold2:
                    tmp_used_flag[nearest_id]=1
                    newloop_idx.append(nearest_id)
                    newloop = newloop + loop_le[nearest_id][:-1]
                    newloop_ordercount += len(loop_le[nearest_id])-1
                    newloop_endi = loop_le[nearest_id][-2]
                    newloop_end = loop_tv[newloop_endi]
                    prev_seq_headi = loop_le[nearest_id][0]
                    prev_seq_head = loop_tv[prev_seq_headi]

                tmp_dist2 = np.sum(np.square(newloop_end-newloop_start))
                if tmp_dist2<distance_threshold2:
                    looped_flag = True
                    break
                if prev_endi==newloop_endi:
                    break
                prev_endi=newloop_endi
            
            if looped_flag:
                loop_fe.append(newloop)
                loop_le_used_flag = tmp_used_flag

    print('loop + found ', len(loop_fe))





    distance_threshold = 0.06
    distance_threshold2 = distance_threshold*distance_threshold
    for i in range(len(loop_le)):
        if loop_le_used_flag[i]==0:
            tmp_used_flag = np.copy(loop_le_used_flag)
            tmp_used_flag[i]=1
            newloop_idx = [i]
            newloop = loop_le[i][:-1]
            newloop_ordercount = len(newloop)
            newloop_starti = newloop[0]
            newloop_start = loop_tv[newloop_starti]
            newloop_endi = newloop[-1]
            newloop_end = loop_tv[newloop_endi]
            prev_seq_headi = newloop_starti
            prev_seq_head = loop_tv[prev_seq_headi]
            prev_endi = newloop_endi
            looped_flag = False
            while True:
                nearest_id = -1
                nearest_dist2 = 666666.666
                for j in range(len(loop_le)):
                    if tmp_used_flag[j]==0:
                        tmp_vi = loop_le[j][0]
                        tmp_v = loop_tv[tmp_vi]
                        tmp_dist2 = np.sum(np.square(newloop_end-tmp_v))
                        if tmp_dist2<nearest_dist2:
                            nearest_id = j
                            nearest_dist2 = tmp_dist2
                if nearest_dist2<distance_threshold2:
                    tmp_used_flag[nearest_id]=1
                    newloop_idx.append(nearest_id)
                    newloop = newloop + loop_le[nearest_id][:-1]
                    newloop_ordercount += len(loop_le[nearest_id])-1
                    newloop_endi = loop_le[nearest_id][-2]
                    newloop_end = loop_tv[newloop_endi]
                    prev_seq_headi = loop_le[nearest_id][0]
                    prev_seq_head = loop_tv[prev_seq_headi]

                tmp_dist2 = np.sum(np.square(newloop_end-newloop_start))
                if tmp_dist2<distance_threshold2:
                    looped_flag = True
                    break
                if prev_endi==newloop_endi:
                    break
                prev_endi=newloop_endi
            
            if looped_flag:
                loop_fe.append(newloop)
                loop_le_used_flag = tmp_used_flag

    print('loop + found ', len(loop_fe))


    '''
    #add lines that are not forming loops
    for i in range(len(loop_le)):
        if loop_le_used_flag[i]==0:
            loop_fe.append(loop_le[i])

    print('loop + lines ', len(loop_fe))
    '''
    
    #remove duplicate and short ones
    loop_de = []
    minimum_loop_len = 6
    for i in range(len(loop_fe)):
        if len(loop_de)==0:
            if len(loop_fe[i])>minimum_loop_len:
                loop_de.append(loop_fe[i])
        else:
            same_flag = False
            
            if loop_fe[i][-1]<0:
                for j in range(i):
                    if loop_fe[j][-1]<0 and loop_fe[i]==loop_fe[j]:
                        same_flag = True
                        break
            else:
                first = loop_fe[i][0]
                for j in range(i):
                    if loop_fe[j][-1]>=0 and first in loop_fe[j]:
                        first_j = loop_fe[j].index(first)
                        new_j = loop_fe[j][first_j:] + loop_fe[j][:first_j]
                        if loop_fe[i]==new_j:
                            same_flag = True
                            break
            if not same_flag:
                if len(loop_fe[i])>minimum_loop_len:
                    loop_de.append(loop_fe[i])

    print('loop - non-duplicate ', len(loop_de))
    

    '''
    #compute how many vertices are used
    #only for filtering parts
    #delete when testing to save time
    loop_count = 0
    delete_flag = False
    delete_v_used_flag = np.zeros([len(loop_tv)], np.uint8)
    for i in range(len(loop_de)):
        if loop_de[i][-1]>=0:
            loop_count += 1
            for j in range(len(loop_de[i])):
                delete_v_used_flag[loop_de[i][j]] = 1
    if loop_count>8:
        delete_flag = True
    if np.sum(delete_v_used_flag)< 0.9*len(loop_tv):
        delete_flag = True
    '''


    #map loop_tv back to vertices
    for i in range(len(loop_de)):
        for j in range(len(loop_de[i])):
            if loop_de[i][j]<0: continue
            loop_de[i][j] = inverse_mapping[loop_de[i][j]]

    #return loop_de, use_flag, delete_flag
    return loop_de, use_flag


def compute_plane_for_each_triangle(vertices,triangles):
    epsilon = 1e-8
    plane_list = np.zeros([len(triangles),4],np.float32)
    for i in range(len(triangles)):
        a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
        x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
        ti = b*z-c*y
        tj = c*x-a*z
        tk = a*y-b*x
        area2 = math.sqrt(ti*ti+tj*tj+tk*tk)
        if area2<epsilon:
            plane_list[i,0] = 0
            plane_list[i,1] = 0
            plane_list[i,2] = 0
            plane_list[i,3] = 0
        else:
            plane_list[i,0] = ti/area2 #a
            plane_list[i,1] = tj/area2 #b
            plane_list[i,2] = tk/area2 #c
            plane_list[i,3] = -(plane_list[i,0]*vertices[triangles[i,0],0]+plane_list[i,1]*vertices[triangles[i,0],1]+plane_list[i,2]*vertices[triangles[i,0],2]) #d = -ax-by-cz
    return plane_list


v2t_mapping_slot_num = 32
def compute_mapping_vertex_to_triangles(vertices,triangles):
    mapping = np.full([len(vertices),v2t_mapping_slot_num],-1,np.int32)
    for i in range(len(triangles)):
        for j in range(3):
            vi = triangles[i][j]
            #full_flag = True
            for k in range(v2t_mapping_slot_num):
                if mapping[vi,k]<0:
                    mapping[vi,k]=i
                    #full_flag = False
                    break
            #if full_flag:
            #    print("ERROR: The mapping slots are full!!")
            #    exit(0)
    return mapping

def compute_vertex_normal(vertices, triangles, part_planes, part_map_v2t):
    vertices_normals = np.zeros([len(vertices),6],np.float32)
    for k in range(len(vertices)):
        normal = np.zeros([3],np.float32)
        for i in range(v2t_mapping_slot_num):
            ti = part_map_v2t[k,i]
            if ti<0: break
            normal = normal+part_planes[ti,:3]
        sqsum = np.sum(np.square(normal))
        if sqsum<1e-10:
            sqsum = 1e-10
        vertices_normals[k,:3] = vertices[k]
        vertices_normals[k,3:] = normal/np.sqrt(sqsum)
    return vertices_normals

def compute_vertex_normal_for_masked_vertices(vertices, triangles, v_use_flag):
    epsilon = 1e-8
    vertices_normals = np.zeros([len(vertices),6],np.float32)

    for i in range(len(triangles)):
        v0 = triangles[i,0]
        v1 = triangles[i,1]
        v2 = triangles[i,2]
        if v_use_flag[v0] or v_use_flag[v1] or v_use_flag[v2]:
            a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
            x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
            ti = b*z-c*y
            tj = c*x-a*z
            tk = a*y-b*x
            area = math.sqrt(ti*ti+tj*tj+tk*tk)
            if area>epsilon:
                plane_a = ti/area #a
                plane_b = tj/area #b
                plane_c = tk/area #c
                if v_use_flag[v0]:
                    vertices_normals[v0,3] += plane_a
                    vertices_normals[v0,4] += plane_b
                    vertices_normals[v0,5] += plane_c
                if v_use_flag[v1]:
                    vertices_normals[v1,3] += plane_a
                    vertices_normals[v1,4] += plane_b
                    vertices_normals[v1,5] += plane_c
                if v_use_flag[v2]:
                    vertices_normals[v2,3] += plane_a
                    vertices_normals[v2,4] += plane_b
                    vertices_normals[v2,5] += plane_c

    for k in range(len(vertices)):
        if v_use_flag[k]:
            sqsum = np.sum(np.square(vertices_normals[k,3:]))
            if sqsum<1e-10:
                sqsum = 1e-10
            vertices_normals[k,:3] = vertices[k]
            vertices_normals[k,3:] = vertices_normals[k,3:]/np.sqrt(sqsum)
    return vertices_normals


def find_nearest_vertex_custom(this_vn,joint_vn):
    return np.argmin( np.sum(np.square(this_vn[:3]-joint_vn[:,:3]),axis=1) * (2.0-np.sum(this_vn[3:]*joint_vn[:,3:],axis=1)) )

def find_nearest_vertex_in_neighbors_custom(this_vn,prev_vi,joint_vn,joint_t,joint_map_v2t):
    min_value = 666666.666
    min_idx = -1
    for i in range(v2t_mapping_slot_num):
        neighbor_ti = joint_map_v2t[prev_vi,i]
        if neighbor_ti<0: break
        for j in range(3):
            neighbor_vi = joint_t[neighbor_ti][j]
            neighbor_v = joint_vn[neighbor_vi]
            #dist2 = np.sum(np.square(neighbor_v-this_vn))
            dist2 = np.sum(np.square(neighbor_v[:3]-this_vn[:3])) * (2.0-np.sum(neighbor_v[3:]*this_vn[3:]))
            if dist2<min_value:
                min_value = dist2
                min_idx = neighbor_vi
    return min_idx, min_value


def find_loop_correspondence_and_fill_seam(loop_vn,loop_e,joint_vn,joint_t,joint_map_v2t,force_correspondence=True):
    #thoughts:
    # 1. find a seed: randomly sample a point in loop_e, then find its nearest vertex in joint_vn
    # 2. find a loop in joint_t by iteratively looking at neighbor vertices (and itself)
    # 3. If a loop is formed, good.
    # 4. If cannot find a loop, use another initial vertex.
    interp_num = 2
    dist_max = 0.05
    dist_max2 = dist_max*dist_max

    corresponding_loop_e = []
    corresponding_vertices = []
    seam_t = []
    for ek in range(len(loop_e)):
        loop_found_flag = False
        save_some_time_prev_v = np.array([666666.666,666666.666,666666.666], np.float32)
        save_some_time_threshold2 = 1e-4
        if loop_e[ek][-1]>=0:
            for ik in range(len(loop_e[ek])):
                #initial seed
                init_vi = loop_e[ek][ik]
                init_v = loop_vn[init_vi]

                #save some time
                if np.sum(np.square(init_v[:3]-save_some_time_prev_v))<save_some_time_threshold2:
                    continue
                save_some_time_prev_v = init_v[:3]

                corr_joint_vi = find_nearest_vertex_custom(init_v,joint_vn)
                corr_joint_vi_list = [corr_joint_vi]
                corr_joint_targetv_list = [init_v]
                ve_dist = 0

                #fill seam
                seam_t_joint_v = corr_joint_vi
                seam_t_list = []

                failed_flag = False
                for i in range(len(loop_e[ek])):
                    current_vi = loop_e[ek][ (ik+i)%len(loop_e[ek]) ]
                    current_v = loop_vn[current_vi]
                    next_vi = loop_e[ek][ (ik+i+1)%len(loop_e[ek]) ]
                    next_v = loop_vn[next_vi]

                    #fill seam
                    seam_t_list.append( [current_vi, next_vi, -1-seam_t_joint_v] )

                    for j in range(interp_num):
                        #interpolate interp_num points for dense sampling
                        this_v = ((j+1)*next_v + (interp_num-j-1)*current_v)/interp_num
                        sqsum = np.sum(np.square(this_v[3:]))
                        if sqsum<1e-10:
                            sqsum = 1e-10
                        this_v[3:] = this_v[3:]/np.sqrt(sqsum)
                        prev_vi = corr_joint_vi_list[-1]
                        corr_joint_vi, curr_ve_dist = find_nearest_vertex_in_neighbors_custom(this_v,prev_vi,joint_vn,joint_t,joint_map_v2t)
                        if np.sum(np.square(joint_vn[corr_joint_vi,:3]-this_v[:3]))>dist_max2:
                            failed_flag = True
                            break
                        if corr_joint_vi!=prev_vi:
                            corr_joint_vi_list.append(corr_joint_vi)
                            corr_joint_targetv_list.append(this_v)
                            ve_dist=curr_ve_dist

                            #fill seam
                            seam_t_list.append( [next_vi, -1-corr_joint_vi, -1-seam_t_joint_v] )
                            seam_t_joint_v = corr_joint_vi
                        elif ve_dist>curr_ve_dist:
                            ve_dist=curr_ve_dist
                            corr_joint_targetv_list[-1][:]=this_v[:]
                        
                    if failed_flag:
                        break

                if (not failed_flag) and len(corr_joint_vi_list)>2 and corr_joint_vi_list[0]==corr_joint_vi_list[-1]:
                    #remove small rings
                    tmp_vi_list = corr_joint_vi_list[:-1]
                    tmp_targetv_list = corr_joint_targetv_list[:-1]
                    i = 0
                    while i<len(tmp_vi_list)-1:
                        i += 1
                        if tmp_vi_list[i] in tmp_vi_list[:i]:
                            kk = tmp_vi_list[i]
                            iidx = tmp_vi_list[:i].index(kk)
                            if i-iidx<len(tmp_vi_list)/2:
                                tmp_vi_list = tmp_vi_list[:iidx] + tmp_vi_list[i:]
                                tmp_targetv_list = tmp_targetv_list[:iidx] + tmp_targetv_list[i:]
                            else:
                                tmp_vi_list = tmp_vi_list[iidx:i]
                                tmp_targetv_list = tmp_targetv_list[iidx:i]
                            
                            #fix seam_t_list
                            for ii in range(len(seam_t_list)):
                                for jj in range(3):
                                    if seam_t_list[ii][jj]<0:
                                        ttt = -seam_t_list[ii][jj]-1
                                        if ttt not in tmp_vi_list:
                                            seam_t_list[ii][jj] = -kk-1

                            i = 0

                    corresponding_loop_e.append(tmp_vi_list)
                    corresponding_vertices.append(tmp_targetv_list)
                    seam_t.append(seam_t_list)
                    loop_found_flag = True
                    break

        if not loop_found_flag and force_correspondence:
            corr_joint_vi_list = []
            corr_joint_targetv_list = []
            for i in range(len(loop_e[ek])):
                this_vi = loop_e[ek][i]
                if this_vi<0: break
                this_v = loop_vn[this_vi]

                corr_joint_vi = find_nearest_vertex_custom(this_v,joint_vn)
                if np.sum(np.square(joint_vn[corr_joint_vi,:3]-this_v[:3]))>dist_max2: continue
                corr_joint_vi_list.append(corr_joint_vi)
                corr_joint_targetv_list.append(this_v)
            
            corr_joint_vi_list.append(-1)

            corresponding_loop_e.append(corr_joint_vi_list)
            corresponding_vertices.append(corr_joint_targetv_list)

    return corresponding_loop_e, corresponding_vertices, seam_t


def filter_triangles(corresponding_loop_e, joint_v, joint_t, joint_map_v2t):
    joint_v_used_flag = np.zeros([len(joint_v)],np.uint8)
    joint_t_used_flag = np.zeros([len(joint_t)],np.uint8)
    border_edge_list = []
    for i in range(len(corresponding_loop_e)):
        if corresponding_loop_e[i][-1]<0: continue
        for j in range(len(corresponding_loop_e[i])):
            if (corresponding_loop_e[i][j],corresponding_loop_e[i][(j+1)%len(corresponding_loop_e[i])]) not in border_edge_list:
                border_edge_list.append( (corresponding_loop_e[i][j],corresponding_loop_e[i][(j+1)%len(corresponding_loop_e[i])]) )
            joint_v_used_flag[corresponding_loop_e[i][j]]=1

    #add triangle in queue
    vqueue = []
    for i in range(len(joint_t)):
        vi0 = joint_t[i,0]
        vi1 = joint_t[i,1]
        vi2 = joint_t[i,2]
        if (vi0, vi1) in border_edge_list or (vi1, vi2) in border_edge_list or (vi2, vi0) in border_edge_list:
            if joint_v_used_flag[vi0]==0:
                vqueue.append(vi0)
                joint_v_used_flag[vi0]=1
            if joint_v_used_flag[vi1]==0:
                vqueue.append(vi1)
                joint_v_used_flag[vi1]=1
            if joint_v_used_flag[vi2]==0:
                vqueue.append(vi2)
                joint_v_used_flag[vi2]=1
            joint_t_used_flag[i]=1
    qp = 0
    while (qp<len(vqueue)):
        this_vi = vqueue[qp]
        qp+=1
        #if qp==200: break #force stop - this line should be removed if not debugging
        for i in range(v2t_mapping_slot_num):
            neighbor_ti = joint_map_v2t[this_vi,i]
            if neighbor_ti<0: break
            if joint_t_used_flag[neighbor_ti]>0: continue
            joint_t_used_flag[neighbor_ti]=1
            vi0 = joint_t[neighbor_ti,0]
            vi1 = joint_t[neighbor_ti,1]
            vi2 = joint_t[neighbor_ti,2]
            if joint_v_used_flag[vi0]==0:
                vqueue.append(vi0)
                joint_v_used_flag[vi0]=1
            if joint_v_used_flag[vi1]==0:
                vqueue.append(vi1)
                joint_v_used_flag[vi1]=1
            if joint_v_used_flag[vi2]==0:
                vqueue.append(vi2)
                joint_v_used_flag[vi2]=1

    return joint_v, joint_t[joint_t_used_flag==1]



def poisson_blending(joint_v, joint_t, corresponding_loop_e, corresponding_vertices, force_correspondence=True):
    '''
    #naive way: only move the boundary points to boundary
    for i in range(len(corresponding_loop_e)):
        for j in range(len(corresponding_loop_e[i])):
            joint_v[corresponding_loop_e[i][j]] = corresponding_vertices[i][j][:3]
    return
    '''
    corresponding_loop_v = []
    corresponding_loop_targetv = []
    for i in range(len(corresponding_loop_e)):
        if corresponding_loop_e[i][-1]<0:
            if force_correspondence:
                corresponding_loop_v = corresponding_loop_v + corresponding_loop_e[i][:-1]
                corresponding_loop_targetv = corresponding_loop_targetv + corresponding_vertices[i]
        else:
            corresponding_loop_v = corresponding_loop_v + corresponding_loop_e[i]
            corresponding_loop_targetv = corresponding_loop_targetv + corresponding_vertices[i]
    #----------poisson blending----------
    for channel_id in range(3):
        print('poisson blending -- channel',str(channel_id))
        num_of_equations = len(joint_t)*3+len(corresponding_loop_v)
        num_of_params = len(joint_t)*6+len(corresponding_loop_v)
        #prepare huge matrices
        data = np.zeros(num_of_params, np.int32)
        row = np.zeros(num_of_params, np.int32)
        col = np.zeros(num_of_params, np.int32)
        b = np.zeros((num_of_equations), np.float32)
        print('constructing A & b')
        #triangles
        row_counter = 0
        data_counter = 0
        for i in range(len(joint_t)):
            for j in range(3):
                vi0 = joint_t[i,j]
                vi1 = joint_t[i,(j+1)%3]
                dist = joint_v[vi1,channel_id] - joint_v[vi0,channel_id]

                row[data_counter] = row_counter
                col[data_counter] = vi1
                data[data_counter] = 1
                data_counter+=1
                row[data_counter] = row_counter
                col[data_counter] = vi0
                data[data_counter] = -1
                data_counter+=1
                b[row_counter] = dist
                row_counter+=1
        #fixed vertices
        for i in range(len(corresponding_loop_v)):
            vi0 = corresponding_loop_v[i]
            dist = corresponding_loop_targetv[i][channel_id]

            row[data_counter] = row_counter
            col[data_counter] = vi0
            data[data_counter] = 10
            data_counter+=1
            b[row_counter] = dist*10
            row_counter+=1
        print('computing ...')
        #compute least square
        if num_of_params != data_counter:
            print("num_of_params != data_counter")
            print(num_of_params, data_counter)
            exit(0)
        A = csr_matrix((data, (row, col)), shape=(row_counter, len(joint_v)))
        solution = lsqr(A, b)[0]
        for i in range(len(joint_v)):
            joint_v[i,channel_id] = solution[i]

    #for i in range(len(corresponding_loop_e)):
    #    for j in range(len(corresponding_loop_e[i])):
    #        joint_v[corresponding_loop_e[i][j]] = corresponding_vertices[i][j][:3]


