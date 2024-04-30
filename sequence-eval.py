import torch
import numpy as np
import time
import os
import cv2
from funcs import *
from AnyOKP import AnyOKP

sequence_name = 'box4'
sequence_folder = "./dataset/frame-sequences/{}/".format(
    sequence_name)
support_img_file = sequence_folder+'support.jpg'
query_img_files = [sequence_folder+filename for filename in os.listdir(
    sequence_folder) if not 'support' in filename and filename.endswith('jpg')]
support_label_file = os.path.join(sequence_folder, sequence_name+'.txt')
support_img = load_img(support_img_file, 512, 512)


# sort
idxs = [int(os.path.basename(file)[:-4]) for file in query_img_files]
idxs1 = np.argsort(idxs)
query_img_files = [query_img_files[idx] for idx in idxs1]


if os.path.exists(support_label_file):
    support_pts,support_edges = load_support_labels(support_label_file)
else:
    support_pts = select_keypoints(support_img)
    # edges
    support_edges = []
    Nse = len(support_pts)
    if Nse > 1:
        for i in range(Nse-1):
            for j in range(i+1, Nse):
                support_edges += [(i, j)]
    
    save_support_labels(support_label_file, support_pts, support_edges)


model = AnyOKP(260, 'dino', binning=1)


rec_pt_all, prec_pt_all,rec_group_all,prec_group_all = [],[],[],[]

with torch.no_grad():
    model.learn_from_support(support_img, support_pts, support_edges)

    for i, query_img_file in enumerate(query_img_files):
        query_img = load_img(query_img_file, 512, 512)
        
        t0 = time.time()
        query_kp_uv_groups, query_kp_id_groups, query_kp_edge_groups = model.extract_from_query(
            query_img, 0, i == 0)
        t1 = time.time()
        print('------Extraction Time: {}ms'.format((t1-t0)*1000))

        
        def load_query_kp_gt(file):
            kp_uvs_groups = []
            kp_ids_groups = []
            with open(file, 'r') as f:
                datas_groups =  f.read().split('O')
                for datas_group in datas_groups:
                    if len(datas_group)>0:
                        datas = datas_group.split('\n')
                        kp_uvs = []
                        kp_ids = []
                        cnt = 1
                        for i,data in enumerate(datas):
                            if len(data)>0:
                                uv = data.split(',')
                                kp_uvs.append((int(uv[0]),int(uv[1])))
                                kp_ids.append(cnt)
                                cnt+=1
                        kp_uvs_groups.append(kp_uvs)
                        kp_ids_groups.append(kp_ids)
                            
            return kp_uvs_groups,kp_ids_groups
        
        gt_kp_uv_groups,gt_kp_id_groups = load_query_kp_gt(query_img_file[:-3]+'txt')
        
        def eval_query_object_kp(query_kp_uv_groups,query_kp_id_groups, gt_kp_uv_groups,gt_kp_id_groups, dist_thresh):
            ################## all kps
            query_kp_uvs, gt_kp_uvs = [],[]
            for kps in query_kp_uv_groups:
                query_kp_uvs+=kps
            for kps in gt_kp_uv_groups:
                gt_kp_uvs+=kps
            query_kp_ids, gt_kp_ids = [],[]
            for ids in query_kp_id_groups:
                query_kp_ids+=ids
            for ids in gt_kp_id_groups:
                gt_kp_ids+=ids                
        
            #calc TP and FP of keypoints
            TP = 0
            FP = 0
            Nq = len(query_kp_uvs)
            Ng = len(gt_kp_uvs)
            for i in range(Nq):
                u,v = query_kp_uvs[i]
                _id = query_kp_ids[i]
                for j in range(Ng):
                    ug,vg = gt_kp_uvs[j]
                    _idg = gt_kp_ids[j]
                    
                    du,dv = ug-u,vg-v
                    if _id==_idg and np.sqrt(du**2+dv**2)<=dist_thresh:
                        TP+=1
                    else:
                        FP+=1
            rec_pt = TP/(Ng+1e-6)
            prec_pt = TP/(Nq+1e-6)
            
            #calc TP and FP of instances
            TP_group = 0
            FP_group = 0
            Nq_group = len(query_kp_uv_groups)
            Ng_group = len(gt_kp_uv_groups)
            for m in range(Nq_group):
                uvs = query_kp_uv_groups[m]
                ids = query_kp_id_groups[m]
                for n in range(Ng_group):
                    gt_uvs = gt_kp_uv_groups[n]
                    gt_ids = gt_kp_id_groups[n]
                    TP = 0
                    Nq = len(uvs)
                    Ng = len(gt_uvs)
                    for i in range(Nq):
                        u,v = uvs[i]
                        _id = ids[i]
                        for j in range(Ng):
                            ug,vg = gt_uvs[j]
                            _idg = gt_ids[j]
                            
                            du,dv = ug-u,vg-v
                            if _id==_idg and np.sqrt(du**2+dv**2)<=dist_thresh:
                                TP+=1
                                
                    if Ng==2 or Ng==3:
                        min_pts = 2
                    elif Ng==4:
                        min_pts = 3
                    elif Ng>4:
                        min_pts = 4            
                    
                    if TP>=min_pts and TP<=Ng:
                        TP_group+=1
                        break
                    else:
                        FP_group+=1
            rec_group = TP_group/(Ng_group+1e-6)
            prec_group = TP_group/(Nq_group+1e-6)
            
            
            return rec_pt, prec_pt,rec_group,prec_group
        
        rec_pt, prec_pt,rec_group,prec_group = eval_query_object_kp(query_kp_uv_groups, query_kp_id_groups, gt_kp_uv_groups,gt_kp_id_groups, dist_thresh=512*0.05)
        print('KP recall: {},    KP precision: {}'.format(rec_pt,prec_pt))
        print('INS recall: {},    INS precision: {}'.format(rec_group,prec_group))
        
        rec_pt_all.append(rec_pt)
        prec_pt_all.append(prec_pt)
        rec_group_all.append(rec_group)
        prec_group_all.append(prec_group)
        
            
        query_img_show = query_img.copy()
        support_img_show = support_img.copy()

        query_img_show = vis_keypoint_groups(query_img_show, query_kp_uv_groups, query_kp_id_groups, query_kp_edge_groups)

        support_img_show = vis_keypoint_groups(support_img_show, [support_pts], [
                                               [i+1 for i in range(len(support_pts))]], [support_edges])

        show = np.hstack((support_img_show, np.ones(
            (512, 32, 3), np.uint8), query_img_show))

        cv2.imshow("support-query", show)
        cv2.waitKey(1)

    
    rec_pt_mean = np.mean(np.array(rec_pt_all))
    prec_pt_mean = np.mean(np.array(prec_pt_all))
    rec_group_mean = np.mean(np.array(rec_group_all))
    prec_group_mean = np.mean(np.array(prec_group_all))
    
    print('KP mean recall: {},    KP mean precision: {}'.format(rec_pt_mean,prec_pt_mean))
    print('INS mean recall: {},    INS mean precision: {}'.format(rec_group_mean,prec_group_mean))