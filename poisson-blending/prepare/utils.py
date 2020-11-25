import numpy as np 
import math
import os


def load_obj(dire):
    fin = open(dire,'r')
    lines = fin.readlines()
    fin.close()
    
    vertices = []
    triangles = []
    
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        if line[0] == 'v':
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            vertices.append([x,y,z])
        if line[0] == 'f':
            x = int(line[1].split("/")[0])
            y = int(line[2].split("/")[0])
            z = int(line[3].split("/")[0])
            triangles.append([x-1,y-1,z-1])
    
    vertices = np.array(vertices, np.float32)
    
    #remove isolated points
    triangles_ = np.array(triangles, np.int32).reshape([-1])
    vertices_ = vertices[triangles_]
    
    
    #normalize diagonal=1
    x_max = np.max(vertices_[:,0])
    y_max = np.max(vertices_[:,1])
    z_max = np.max(vertices_[:,2])
    x_min = np.min(vertices_[:,0])
    y_min = np.min(vertices_[:,1])
    z_min = np.min(vertices_[:,2])
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = math.sqrt(x_scale*x_scale + y_scale*y_scale + z_scale*z_scale)
    
    '''
    #normalize max=1
    x_max = np.max(vertices_[:,0])
    y_max = np.max(vertices_[:,1])
    z_max = np.max(vertices_[:,2])
    x_min = np.min(vertices_[:,0])
    y_min = np.min(vertices_[:,1])
    z_min = np.min(vertices_[:,2])
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = max( max(x_scale, y_scale), z_scale)
    '''
    
    #print(len(vertices), len(triangles))
    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)
    
    vertices[:,0] = (vertices[:,0]-x_mid)/scale
    vertices[:,1] = (vertices[:,1]-y_mid)/scale
    vertices[:,2] = (vertices[:,2]-z_mid)/scale
    
    return vertices, triangles

def read_ply_and_labels(name, part_list):
    part_num = len(part_list)

    txt_file = open(name+"_yi2016-afterFilter.ply", 'r')
    lines = txt_file.readlines()
    txt_file.close()

    txt_file2 = open(name+"_yi2016-afterFilter.labels", 'r')
    lines2 = txt_file2.readlines()
    txt_file2.close()
    
    point_num = 0
    start_pos = 0
    for i in range(len(lines)):
        line = lines[i].split()
        if line[0]=="element" and line[1]=="vertex":
            point_num = int(line[2])
        if line[0]=="end_header":
            start_pos = i+1
            break

    part_points_ = []
    for j in range(part_num):
        part_points_.append([])

    for i in range(point_num):
        line = lines[start_pos+i].split()
        line2 = lines2[i]
        for j in range(part_num):
            if int(line2)==part_list[j]:
                part_points_[j].append([float(line[0]),float(line[1]),float(line[2])])

    part_points = []
    for j in range(part_num):
        if len(part_points_[j])==0:
            part_points.append([])
        else:
            part_points.append(np.array(part_points_[j],np.float32))

    return part_points

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


def write_ply_triangle(dire, vertices, triangles):
    #output ply
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

    #put the actual loops in another file
    fout = open(dire[:-3]+"txt", 'w')
    for i in range(len(loops)):
        for j in range(len(loops[i])):
            fout.write(str(loops[i][j])+" ")
        fout.write("\n")
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
        #find origin
        prev_v_list = [target_v]
        if prev_vertex[target_v]>=0:
            prev_v = prev_vertex[target_v]
            prev_v_list.append(prev_v)
            vertex_used_flag[target_v]=1
            vertex_used_flag[prev_v]=1
            while prev_vertex[prev_v]>=0 and prev_vertex[prev_v] not in prev_v_list:
                prev_v = prev_vertex[prev_v]
                vertex_used_flag[prev_v]=1
                prev_v_list.append(prev_v)

        #get loop
        next_v_list = prev_v_list[::-1]
        if prev_vertex[next_v_list[0]]<0:
            if next_vertex[target_v]>=0:
                next_v = next_vertex[target_v]
                prev_v_list.append(next_v)
                vertex_used_flag[target_v]=1
                vertex_used_flag[next_v]=1
                while next_vertex[next_v]>=0 and next_vertex[next_v] not in next_v_list:
                    next_v = next_vertex[next_v]
                    vertex_used_flag[next_v]=1
                    next_v_list.append(next_v)
                if next_vertex[next_v]<0:
                    next_v_list.append(-1)
            else:
                next_v_list.append(-1)

        loop_le.append(next_v_list)


    print('loop_le', len(loop_le))

    #close loops
    loop_fe = []
    loop_le_used_flag = np.zeros([len(loop_le)], np.uint8)
    for i in range(len(loop_le)):
        if loop_le[i][-1]>=0:
            loop_fe.append(loop_le[i])
            loop_le_used_flag[i]=1

    print('loop_fe', len(loop_fe))


    distance_threshold = 0.02
    distance_threshold2 = distance_threshold*distance_threshold
    #overlap_threshold = 0.00001
    #overlap_threshold2 = overlap_threshold*overlap_threshold
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
                        #tmp_vi = loop_le[j][-2]
                        #tmp_v = loop_tv[tmp_vi]
                        #tmp_dist2 = np.sum(np.square(prev_seq_head-tmp_v))
                        #if tmp_dist2>overlap_threshold2:
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

    print('loop_fe', len(loop_fe))
    
    
    for i in range(len(loop_le)):
        if loop_le_used_flag[i]==0:
            loop_fe.append(loop_le[i])

    print('loop_fe', len(loop_fe))

    print("finding loop - end")
    

    #map loop_tv back to vertices
    for i in range(len(loop_fe)):
        for j in range(len(loop_fe[i])):
            if loop_fe[i][j]<0: continue
            loop_fe[i][j] = inverse_mapping[loop_fe[i][j]]

    return loop_fe, use_flag




erode_threshold = 0.05
erode_threshold_in = 0.04
erode_threshold_out = 0.06
erode_threshold2 = erode_threshold*erode_threshold
erode_threshold_in2 = erode_threshold_in*erode_threshold_in
erode_threshold_out2 = erode_threshold_out*erode_threshold_out
def create_eroded(part_points_normals_origin):
    part_num = len(part_points_normals_origin)

    #last slot to store eroded points
    tmp_points_normals = []
    for i in range(part_num+1):
        tmp_points_normals.append([])

    #erode
    for i in range(part_num):
        for j in range(len(part_points_normals_origin[i])):
            p = part_points_normals_origin[i][j,:3]
            erode_flag = False
            for k in range(part_num):
                if i==k: continue
                if len(part_points_normals_origin[k])==0:
                    current_distance = 666666.666
                else:
                    current_distance = min_dist2(p,part_points_normals_origin[k][:,:3])
                if current_distance<erode_threshold2:
                    erode_flag = True
                    break
            if erode_flag:
                tmp_points_normals[part_num].append(part_points_normals_origin[i][j])
            else:
                tmp_points_normals[i].append(part_points_normals_origin[i][j])

    eroded_part_points_normals = []
    for i in range(part_num+1):
        eroded_part_points_normals.append(np.array(tmp_points_normals[i], np.float32))

    return eroded_part_points_normals


def get_closest_class(p,part_points):
    idx = 0
    min_value = 666666.666
    for j in range(len(part_points)):
        current_distance = min_dist2(p,part_points[j])
        if current_distance<min_value:
            min_value = current_distance
            idx = j
    return idx

def get_class_by_eroding(p,part_points_not_eroded,threshold2):
    idx = -1
    erode_flag = False
    for j in range(len(part_points_not_eroded)):
        current_distance = min_dist2(p,part_points_not_eroded[j])
        if current_distance<threshold2:
            if idx<0:
                idx = j
            else:
                erode_flag=True
                break
    if idx<0:
        return get_closest_class(p,part_points_not_eroded)
    if erode_flag:
        return len(part_points_not_eroded)
    return idx


def force_assign_face_color(vertices, triangles, part_points, label_vertices, label_triangles):
    for i in range(len(label_vertices)-1,-1,-1):
        if label_vertices[i]<0:
            label_vertices[i] = get_class_by_eroding(vertices[i],part_points,erode_threshold2)
        else:
            break
    for i in range(len(triangles)):
        if label_triangles[i]<0:
            pi0 = triangles[i][0]
            pi1 = triangles[i][1]
            pi2 = triangles[i][2]
            p0 = vertices[pi0]
            p1 = vertices[pi1]
            p2 = vertices[pi2]

            vote_triangles = [0]*(len(part_points)+1)
            
            p_mid = (p0+p1+p2)/3
            vote_triangles[get_class_by_eroding(p_mid,part_points,erode_threshold2)] += 1
            
            p01 = (p0+p1)/2
            vote_triangles[get_class_by_eroding(p01,part_points,erode_threshold2)] += 1
            p12 = (p1+p2)/2
            vote_triangles[get_class_by_eroding(p12,part_points,erode_threshold2)] += 1
            p02 = (p2+p0)/2
            vote_triangles[get_class_by_eroding(p02,part_points,erode_threshold2)] += 1
            
            p = (p01+p02+p0)/3
            vote_triangles[get_class_by_eroding(p,part_points,erode_threshold2)] += 1
            p = (p01+p12+p1)/3
            vote_triangles[get_class_by_eroding(p,part_points,erode_threshold2)] += 1
            p = (p12+p02+p2)/3
            vote_triangles[get_class_by_eroding(p,part_points,erode_threshold2)] += 1
            
            vote_triangles[label_vertices[pi0]] += 1
            vote_triangles[label_vertices[pi1]] += 1
            vote_triangles[label_vertices[pi2]] += 1
            
            vote_triangles_max = np.argmax(vote_triangles)
            label_triangles[i] = vote_triangles_max


def get_vertex_color(points_normals, part_points, label_points_normals):
    for i in range(len(label_points_normals)):
        label_points_normals[i] = get_class_by_eroding(points_normals[i,:3],part_points,erode_threshold2)
def collect_part_points_normals(points_normals, label_points_normals, part_id):
    return points_normals[label_points_normals==part_id]


def get_face_color_multiple(vertices, triangles, part_points, label_vertices, label_triangles_list):
    for i in range(len(label_vertices)):
        label_vertices[i] = get_closest_class(vertices[i],part_points)
    for i in range(len(triangles)):
        pi0 = triangles[i][0]
        pi1 = triangles[i][1]
        pi2 = triangles[i][2]
        p0 = vertices[pi0]
        p1 = vertices[pi1]
        p2 = vertices[pi2]
        
        p_mid = (p0+p1+p2)/3
        cpi = get_closest_class(p_mid,part_points)
        label_triangles_list[cpi,i] = cpi
        
        p01 = (p0+p1)/2
        cpi = get_closest_class(p01,part_points)
        label_triangles_list[cpi,i] = cpi
        p12 = (p1+p2)/2
        cpi = get_closest_class(p12,part_points)
        label_triangles_list[cpi,i] = cpi
        p02 = (p2+p0)/2
        cpi = get_closest_class(p02,part_points)
        label_triangles_list[cpi,i] = cpi
        
        p = (p01+p02+p0)/3
        cpi = get_closest_class(p,part_points)
        label_triangles_list[cpi,i] = cpi
        p = (p01+p12+p1)/3
        cpi = get_closest_class(p,part_points)
        label_triangles_list[cpi,i] = cpi
        p = (p12+p02+p2)/3
        cpi = get_closest_class(p,part_points)
        label_triangles_list[cpi,i] = cpi
        
        cpi = label_vertices[pi0]
        label_triangles_list[cpi,i] = cpi
        cpi = label_vertices[pi1]
        label_triangles_list[cpi,i] = cpi
        cpi = label_vertices[pi2]
        label_triangles_list[cpi,i] = cpi


split_threshold = 0.2
split_threshold2 = split_threshold*split_threshold
def get_face_color(vertices, triangles, part_points, label_vertices, label_part_vertices_out, label_part_vertices_in, label_triangles):
    for i in range(len(label_vertices)-1,-1,-1):
        if label_vertices[i]<0:
            label_vertices[i] = get_class_by_eroding(vertices[i],part_points,erode_threshold2)
            label_part_vertices_out[i] = get_class_by_eroding(vertices[i],part_points,erode_threshold_out2)
            label_part_vertices_in[i] = get_class_by_eroding(vertices[i],part_points,erode_threshold_in2)
        else:
            break

    eroded_id = len(part_points)

    for i in range(len(triangles)):
        if label_triangles[i]<0:
            pi0 = triangles[i][0]
            pi1 = triangles[i][1]
            pi2 = triangles[i][2]
            p0 = vertices[pi0]
            p1 = vertices[pi1]
            p2 = vertices[pi2]

            #if sqdist(p0,p1)>split_threshold2 or sqdist(p1,p2)>split_threshold2 or sqdist(p2,p0)>split_threshold2:
            #    continue


            vote_triangles = [0]*(eroded_id+1)
            
            #in
            p_mid = (p0+p1+p2)/3
            vote_triangles[get_class_by_eroding(p_mid,part_points,erode_threshold_in2)] += 1
            
            p01 = (p0+p1)/2
            vote_triangles[get_class_by_eroding(p01,part_points,erode_threshold_in2)] += 1
            p12 = (p1+p2)/2
            vote_triangles[get_class_by_eroding(p12,part_points,erode_threshold_in2)] += 1
            p02 = (p2+p0)/2
            vote_triangles[get_class_by_eroding(p02,part_points,erode_threshold_in2)] += 1
            
            p = (p01+p02+p0)/3
            vote_triangles[get_class_by_eroding(p,part_points,erode_threshold_in2)] += 1
            p = (p01+p12+p1)/3
            vote_triangles[get_class_by_eroding(p,part_points,erode_threshold_in2)] += 1
            p = (p12+p02+p2)/3
            vote_triangles[get_class_by_eroding(p,part_points,erode_threshold_in2)] += 1
            
            vote_triangles[label_part_vertices_in[pi0]] += 1
            vote_triangles[label_part_vertices_in[pi1]] += 1
            vote_triangles[label_part_vertices_in[pi2]] += 1


            #out
            p_mid = (p0+p1+p2)/3
            vote_triangles[get_class_by_eroding(p_mid,part_points,erode_threshold_out2)] += 1
            
            p01 = (p0+p1)/2
            vote_triangles[get_class_by_eroding(p01,part_points,erode_threshold_out2)] += 1
            p12 = (p1+p2)/2
            vote_triangles[get_class_by_eroding(p12,part_points,erode_threshold_out2)] += 1
            p02 = (p2+p0)/2
            vote_triangles[get_class_by_eroding(p02,part_points,erode_threshold_out2)] += 1
            
            p = (p01+p02+p0)/3
            vote_triangles[get_class_by_eroding(p,part_points,erode_threshold_out2)] += 1
            p = (p01+p12+p1)/3
            vote_triangles[get_class_by_eroding(p,part_points,erode_threshold_out2)] += 1
            p = (p12+p02+p2)/3
            vote_triangles[get_class_by_eroding(p,part_points,erode_threshold_out2)] += 1
            
            vote_triangles[label_part_vertices_out[pi0]] += 1
            vote_triangles[label_part_vertices_out[pi1]] += 1
            vote_triangles[label_part_vertices_out[pi2]] += 1

            vote_triangles_max = np.argmax(vote_triangles)
            if vote_triangles[vote_triangles_max]==20:
                label_triangles[i] = vote_triangles_max


#note: only check the 3 vertices of each triangle
def get_face_color_no_tolerance(vertices, triangles, part_points, label_vertices, label_triangles):
    for i in range(len(label_vertices)-1,-1,-1):
        if label_vertices[i]<0:
            label_vertices[i] = get_class_by_eroding(vertices[i],part_points,erode_threshold2)
        else:
            break
    for i in range(len(triangles)):
        if label_triangles[i]<0:
            vote_triangles = [0]*(len(part_points)+1)
            vote_triangles[label_vertices[triangles[i][0]]] += 1
            vote_triangles[label_vertices[triangles[i][1]]] += 1
            vote_triangles[label_vertices[triangles[i][2]]] += 1
            vote_triangles_max = np.argmax(vote_triangles)
            if vote_triangles[vote_triangles_max]==3:
                label_triangles[i] = vote_triangles_max


def compute_plane_for_each_triangle(vertices,triangles):
    epsilon = 1e-8
    plane_list = np.zeros([len(triangles),4],np.float32)
    for i in range(len(triangles)):
        a,b,c = vertices[triangles[i][1]]-vertices[triangles[i][0]]
        x,y,z = vertices[triangles[i][2]]-vertices[triangles[i][0]]
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
            plane_list[i,3] = -(plane_list[i,0]*vertices[triangles[i][0]][0]+plane_list[i,1]*vertices[triangles[i][0]][1]+plane_list[i,2]*vertices[triangles[i][0]][2]) #d = -ax-by-cz
    return plane_list

v2t_mapping_slot_num = 32
def compute_mapping_vertex_to_triangles(vertices,triangles):
    mapping = np.full([len(vertices),v2t_mapping_slot_num],-1,np.int32)
    for i in range(len(triangles)):
        for j in range(3):
            vi = triangles[i][j]
            for k in range(v2t_mapping_slot_num):
                if mapping[vi,k]<0:
                    mapping[vi,k]=i
                    break
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


def collect_part_face(vertices, triangles, label_vertices, label_triangles, part_id):
    part_vertices = []
    part_triangles = []
    vertex_mapping = np.full([len(vertices)], -1, np.int32)

    #collect faces
    for i in range(len(triangles)):
        if label_triangles[i] == part_id:
            tmpface = []
            for j in range(3):
                pi0 = triangles[i][j]
                if vertex_mapping[pi0]<0:
                    vertex_mapping[pi0] = len(part_vertices)
                    part_vertices.append(vertices[pi0])
                    vi0 = vertex_mapping[pi0]
                else:
                    vi0 = vertex_mapping[pi0]
                tmpface.append(vi0)
            part_triangles.append(tmpface)

    part_vertices = np.array(part_vertices,np.float32)
    part_triangles = np.array(part_triangles,np.int32)

    return part_vertices, part_triangles


def collect_part_face_and_edge(vertices, triangles, label_vertices, label_triangles, part_id):
    part_vertices = []
    part_triangles = []
    part_edges = []
    vertex_mapping = np.full([len(vertices)], -1, np.int32)

    #collect faces
    for i in range(len(triangles)):
        if label_triangles[i] == part_id:
            tmpface = []
            for j in range(3):
                pi0 = triangles[i][j]
                if vertex_mapping[pi0]<0:
                    vertex_mapping[pi0] = len(part_vertices)
                    part_vertices.append(vertices[pi0])
                    vi0 = vertex_mapping[pi0]
                else:
                    vi0 = vertex_mapping[pi0]
                tmpface.append(vi0)
            part_triangles.append(tmpface)

    #collect edges
    for i in range(len(triangles)):
        count = 0
        for j in range(3):
            if vertex_mapping[triangles[i][j]]>=0:
                count += 1
        if count==2:
            if vertex_mapping[triangles[i][0]]<0:
                part_edges.append( [vertex_mapping[triangles[i][1]],vertex_mapping[triangles[i][2]]] )
            elif vertex_mapping[triangles[i][1]]<0:
                part_edges.append( [vertex_mapping[triangles[i][2]],vertex_mapping[triangles[i][0]]] )
            elif vertex_mapping[triangles[i][2]]<0:
                part_edges.append( [vertex_mapping[triangles[i][0]],vertex_mapping[triangles[i][1]]] )
    
    part_vertices = np.array(part_vertices,np.float32)
    part_triangles = np.array(part_triangles,np.int32)
    part_edges = np.array(part_edges,np.int32)

    return part_vertices, part_triangles, part_edges


max_num_interation = 10
def adaptive_midpoint_mid(p1,p2,p1_id,p2_id,part_points):
    left = p1
    right = p2
    pmid = (left+right)/2
    for i in range(max_num_interation):
        pmid_id = get_class_by_eroding(pmid,part_points,erode_threshold2)
        if pmid_id!=p1_id and pmid_id!=p2_id:
            return pmid
        if pmid_id==p1_id:
            left = pmid
            pmid = (left+right)/2
        else:
            right = pmid
            pmid = (left+right)/2
    return (p1+p2)/2

def adaptive_subdiv_according_to_color_mid(vertices, triangles, part_points, label_vertices, label_part_vertices_out, label_part_vertices_in, label_triangles):
    index_p1_p2 = []
    index_pmid = []
    
    for i in range(len(triangles)):
        if label_triangles[i]<0:
            pi1 = triangles[i][0]
            pi2 = triangles[i][1]
            pi3 = triangles[i][2]
            p1 = vertices[pi1]
            p2 = vertices[pi2]
            p3 = vertices[pi3]
            
            #subdiv
            current_len = len(vertices)
            current_counter = 0
            
            if (pi1,pi2) in index_p1_p2:
                pi12 = index_pmid[index_p1_p2.index((pi1,pi2))]
            elif (pi2,pi1) in index_p1_p2:
                pi12 = index_pmid[index_p1_p2.index((pi2,pi1))]
            else:
                vertices.append(adaptive_midpoint_mid(p1,p2,label_vertices[pi1],label_vertices[pi2],part_points))
                label_vertices.append(-1)
                label_part_vertices_out.append(-1)
                label_part_vertices_in.append(-1)
                pi12 = current_len+current_counter
                current_counter +=1
                index_p1_p2.append((pi1,pi2))
                index_pmid.append(pi12)
            
            if (pi3,pi2) in index_p1_p2:
                pi23 = index_pmid[index_p1_p2.index((pi3,pi2))]
            elif (pi2,pi3) in index_p1_p2:
                pi23 = index_pmid[index_p1_p2.index((pi2,pi3))]
            else:
                vertices.append(adaptive_midpoint_mid(p3,p2,label_vertices[pi3],label_vertices[pi2],part_points))
                label_vertices.append(-1)
                label_part_vertices_out.append(-1)
                label_part_vertices_in.append(-1)
                pi23 = current_len+current_counter
                current_counter +=1
                index_p1_p2.append((pi3,pi2))
                index_pmid.append(pi23)
            
            if (pi3,pi1) in index_p1_p2:
                pi13 = index_pmid[index_p1_p2.index((pi3,pi1))]
            elif (pi1,pi3) in index_p1_p2:
                pi13 = index_pmid[index_p1_p2.index((pi1,pi3))]
            else:
                vertices.append(adaptive_midpoint_mid(p3,p1,label_vertices[pi3],label_vertices[pi1],part_points))
                label_vertices.append(-1)
                label_part_vertices_out.append(-1)
                label_part_vertices_in.append(-1)
                pi13 = current_len+current_counter
                current_counter +=1
                index_p1_p2.append((pi1,pi3))
                index_pmid.append(pi13)
            
            
            triangles[i][1]=pi12
            triangles[i][2]=pi13
            triangles.append([pi2,pi23,pi12])
            triangles.append([pi3,pi13,pi23])
            triangles.append([pi12,pi23,pi13])
            label_triangles.append(-1)
            label_triangles.append(-1)
            label_triangles.append(-1)

def adaptive_midpoint_boundary(p1,p2,p1_id,p2_id,part_points):
    left = p1
    right = p2
    pmid = (left+right)/2
    for i in range(max_num_interation):
        pmid_id = get_class_by_eroding(pmid,part_points,erode_threshold2)
        if pmid_id!=p1_id and pmid_id!=p2_id:
            return pmid
        if pmid_id==p1_id:
            left = pmid
            pmid = (left+right)/2
        else:
            right = pmid
            pmid = (left+right)/2
    if p1_id==p2_id:
        return (p1+p2)/2
    return pmid

def adaptive_subdiv_according_to_color_boundary(vertices, triangles, part_points, label_vertices, label_triangles):
    index_p1_p2 = []
    index_pmid = []
    
    for i in range(len(triangles)):
        if label_triangles[i]<0:
            pi1 = triangles[i][0]
            pi2 = triangles[i][1]
            pi3 = triangles[i][2]
            p1 = vertices[pi1]
            p2 = vertices[pi2]
            p3 = vertices[pi3]
            
            #subdiv
            current_len = len(vertices)
            current_counter = 0
            
            if (pi1,pi2) in index_p1_p2:
                pi12 = index_pmid[index_p1_p2.index((pi1,pi2))]
            elif (pi2,pi1) in index_p1_p2:
                pi12 = index_pmid[index_p1_p2.index((pi2,pi1))]
            else:
                vertices.append(adaptive_midpoint_boundary(p1,p2,label_vertices[pi1],label_vertices[pi2],part_points))
                label_vertices.append(-1)
                pi12 = current_len+current_counter
                current_counter +=1
                index_p1_p2.append((pi1,pi2))
                index_pmid.append(pi12)
            
            if (pi3,pi2) in index_p1_p2:
                pi23 = index_pmid[index_p1_p2.index((pi3,pi2))]
            elif (pi2,pi3) in index_p1_p2:
                pi23 = index_pmid[index_p1_p2.index((pi2,pi3))]
            else:
                vertices.append(adaptive_midpoint_boundary(p3,p2,label_vertices[pi3],label_vertices[pi2],part_points))
                label_vertices.append(-1)
                pi23 = current_len+current_counter
                current_counter +=1
                index_p1_p2.append((pi3,pi2))
                index_pmid.append(pi23)
            
            if (pi3,pi1) in index_p1_p2:
                pi13 = index_pmid[index_p1_p2.index((pi3,pi1))]
            elif (pi1,pi3) in index_p1_p2:
                pi13 = index_pmid[index_p1_p2.index((pi1,pi3))]
            else:
                vertices.append(adaptive_midpoint_boundary(p3,p1,label_vertices[pi3],label_vertices[pi1],part_points))
                label_vertices.append(-1)
                pi13 = current_len+current_counter
                current_counter +=1
                index_p1_p2.append((pi1,pi3))
                index_pmid.append(pi13)
            
            
            triangles[i][1]=pi12
            triangles[i][2]=pi13
            triangles.append([pi2,pi23,pi12])
            triangles.append([pi3,pi13,pi23])
            triangles.append([pi12,pi23,pi13])
            label_triangles.append(-1)
            label_triangles.append(-1)
            label_triangles.append(-1)







