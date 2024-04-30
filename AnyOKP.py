import argparse
import torch
from pathlib import Path
import numpy as np
import time
import cv2

import types
from funcs import *
import timm


class AnyOKP:
    
    def __init__(self, input_size:int, feature_extractor_name: str, weight_dir:str='',n_edge_seg:int=8, binning:bool=True):
        self.feature_extractor_list = ['dino','dinov2','sam','superpoint','loftr','sam','dinores50']
        self.feature_extractor_name = feature_extractor_name
        self.weight_dir = weight_dir
        self.input_size = input_size
        self._init_feature_extractor()
        self.binning = binning
        self.n_edge_seg = n_edge_seg
    
    def _init_feature_extractor(self):
        print("feature extractor init...")

        
        if self.feature_extractor_name=='dinov2':
            self.fe = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.patch_size = self.fe.patch_embed.patch_size[0]
            # fix the stride
            self.stride = 7
            self.fe.patch_embed.proj.stride = (self.stride,self.stride)
            # fix the positional encoding code
            self.fe.interpolate_pos_encoding = types.MethodType(fix_pos_enc(self.patch_size, (self.stride,self.stride)), self.fe)
            self.patch_resolution = [63,63]
            self.fe.cuda()
        elif self.feature_extractor_name=='dino':
            self.fe = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
            self.patch_size = self.fe.patch_embed.patch_size
            # fix the stride
            self.stride = 4
            self.fe.patch_embed.proj.stride = (self.stride,self.stride)
            # fix the positional encoding code
            self.fe.interpolate_pos_encoding = types.MethodType(fix_pos_enc(self.patch_size, (self.stride,self.stride)), self.fe)
            self.patch_resolution = [64,64]
            self.fe.cuda()
        elif self.feature_extractor_name=='loftr':
            
            from loftr.loftr import LoFTR, default_cfg
            from copy import deepcopy
            _default_cfg = deepcopy(default_cfg)
            _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
            self.fe = LoFTR(config=_default_cfg)
            self.fe.load_state_dict(torch.load("/home/user/SURA_Projects/LoFTR/weights/indoor_ds_new.ckpt")['state_dict'])
            self.fe = self.fe.eval().cuda()
            self.patch_resolution = [64,64]
            self.stride = 8
            self.patch_size = 1
        elif self.feature_extractor_name=='sam':
            self.fe = sam_model_registry["vit_b"](checkpoint="/home/user/SURA_Projects/segment-anything/checkpoints/sam_vit_b_01ec64.pth").cuda() 
            self.patch_resolution = [64,64]
            self.stride = 8
            self.patch_size = 16
        elif self.feature_extractor_name=='dinores50':
            self.fe = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            self.patch_resolution = [64,64]
            self.stride = 8
            self.patch_size = 1
            self.fe.cuda()
    
        print("feature extractor init done")
    
    def _forward_feature_map(self,img):
        print("forwarding...")
        t0 = time.time()

        if self.feature_extractor_name=='dinov2':
            inp = preprocess_numpy(img, self.input_size)
            fm = self.fe.get_intermediate_layers(inp.cuda(), 1)[0].unsqueeze(1)

        elif self.feature_extractor_name=='dino':
            inp = preprocess_numpy(img, self.input_size)
            fm = self.fe.get_intermediate_layers(inp.cuda(), 1)[0].unsqueeze(1)[:,:,1:,:]            
        elif self.feature_extractor_name=='loftr':
            inp = preprocess_numpy(img, self.input_size,is_gray=True)
            fm = self.fe.backbone(inp.cuda())[0]
            fm = torch.reshape(fm,[fm.shape[0],1,fm.shape[1],-1]).transpose(3,2)
        elif self.feature_extractor_name=='sam':
            inp = preprocess_numpy(img, self.input_size)
            fm = self.fe.image_encoder(inp.cuda())
            fm = torch.reshape(fm,(fm.shape[0],1,fm.shape[1],-1)).permute(0,1,3,2)            
        elif self.feature_extractor_name=='dinores50':
            inp = preprocess_numpy(img, self.input_size)
            fm = self.fe.forward(inp.cuda())
            fm = torch.reshape(fm,[fm.shape[0],1,fm.shape[1],-1]).transpose(3,2)

        att = torch.mean(torch.abs(fm),dim=-1)
        att = torch.sigmoid((att-torch.min(att))/(torch.max(att)-torch.min(att))*10.0-5.0)
        # att = (att-torch.min(att))/(torch.max(att)-torch.min(att))
        
        vis_att = torch.reshape(att*255.0,(self.patch_resolution[0],self.patch_resolution[1],1)).cpu().numpy().astype(np.uint8)
        vis_att = cv2.applyColorMap(cv2.resize(vis_att,(512,512),cv2.INTER_NEAREST),cv2.COLORMAP_OCEAN)
        cv2.imshow('objectness',vis_att)
        
        
        

        fm = fm*att.unsqueeze(-1)
        
        t2 = time.time()
        
        if self.binning:
            fm = log_bin(fm, self.patch_resolution)  
        t1 = time.time()
        print('Foward Time: {}ms,   Binning Time: {}ms'.format((t2-t0)*1000,(t1-t2)*1000))       
        return fm,vis_att

        
    def learn_from_support(self,support_img,support_pts,support_edges):
        self.Hs, self.Ws, _ = support_img.shape
        self.fm_support,_ = self._forward_feature_map(support_img)
        
        
        # support label points
        self.idx_protos = []
        self.support_xys = []
        for support_pt in support_pts:
            u, v = support_pt[0]*self.input_size /self.Ws, support_pt[1]*self.input_size/self.Hs  # pixel pos
            x = np.around((u-(self.patch_size-1)/2)*1.0/self.stride).astype(int)
            y = np.around((v-(self.patch_size-1)/2)*1.0/self.stride).astype(int)
            idx = y*self.patch_resolution[1]+x
            self.idx_protos.append(idx)
            self.support_xys.append((x, y))

        # support edge features
        self.support_edge_ftrs = []
        self.edge_idx_dict = {}
        self.support_edges = support_edges
        for n, eij in enumerate(support_edges):
            i, j = eij
            self.edge_idx_dict[(i, j)] = n
            xi, yi = self.support_xys[i]
            xj, yj = self.support_xys[j]
            L = np.linalg.norm((xi-xj, yi-yj))
            vx = (xj-xi)/L
            vy = (yj-yi)/L
            l = 2
            efs = []
            while l <= L-2:
                x = np.around(xi+l*vx).astype(int)
                y = np.around(yi+l*vy).astype(int)
                idx = y*self.patch_resolution[1]+x
                efs.append(self.fm_support[:, :, idx:idx+1, :])
                l += 1
            Nsap = len(efs)
            Nseg = self.n_edge_seg
            lseg = Nsap/Nseg
            efss = [[] for nseg in range(Nseg)]
            for nseg in range(Nseg):
                for ie in range(int(nseg*lseg), max(int(nseg*lseg+1), int((nseg+1)*lseg))):
                    efss[nseg].append(efs[ie])
            efs_pool = []
            for efs in efss:
                efs_pool.append(torch.mean(torch.concat(efs, dim=-2), dim=-2, keepdims=True))

            ef = torch.concat(efs_pool, dim=-1)
            self.support_edge_ftrs.append(ef)

    def extract_from_query(self, query_image, K_pca=0, fit_pca=True):
        t0 = time.time()
        
        self.Hq, self.Wq, _ = query_image.shape
        self.fm_query,self.vis_obj_att = self._forward_feature_map(query_image)
        
        t1 = time.time()
        print('FE Time: {}ms'.format((t1-t0)*1000))
        
        
        # PCA
        if K_pca>0:
            Xq = self.fm_query[0, 0, :, :]
            Xs = self.fm_support[0,0,  :, :]
            if fit_pca:
                self.pca = PCA(n_components=K_pca)
                self.pca.fit(Xq)
                
            
            Xq1 = self.pca.transform(Xq)[np.newaxis, :][np.newaxis, :]
            Xs1 = self.pca.transform(Xs)[np.newaxis, :][np.newaxis, :]
            support_edge_ftrs1 = []
            for i in range(len(self.support_edge_ftrs)):
                support_edge_ftrs1.append( torch.reshape(self.pca.transform(torch.reshape(self.support_edge_ftrs[i],(self.n_edge_seg,-1))),(1,1,1,-1)))
            Xs = Xs1
            Xq = Xq1
        else:
            Xq = self.fm_query
            Xs = self.fm_support
            support_edge_ftrs1 = self.support_edge_ftrs
            
        t2 = time.time()
        print('PCA Time: {}ms'.format((t2-t1)*1000))
            
        # best-proto pairs
        # similarities = chunk_cosine_sim(Xs, Xq).float()
        norm_s = torch.sqrt(torch.sum(torch.square(Xs),dim=-1,keepdim=True))
        norm_q = torch.sqrt(torch.sum(torch.square(Xq),dim=-1,keepdim=True))
        
        similarities = torch.matmul(Xs/norm_s,torch.transpose(Xq/norm_q, 3, 2))
        # indices of support desc closest to query desc
        sim_s2q, idx_s = torch.max(similarities, dim=-2)
        score_query_pts = sim_s2q[0, 0, :].cpu().numpy()
        
        Wp,Hp = self.patch_resolution

        # get candidate keypoints
        idx_query_pts_matched_IDs = []
        id_query_pts_matched_IDs = []
        for _id, idx in enumerate(self.idx_protos):
            mask_query_pts = idx_s[0, 0, :] == idx
            mask_query_pts = torch.bitwise_or(
                mask_query_pts, idx_s[0, 0, :] == idx+1)
            mask_query_pts = torch.bitwise_or(
                mask_query_pts, idx_s[0, 0, :] == idx-1)
            mask_query_pts = torch.bitwise_or(
                mask_query_pts, idx_s[0, 0, :] == idx+Wp)
            mask_query_pts = torch.bitwise_or(
                mask_query_pts, idx_s[0, 0, :] == idx-Wp)

            idx_query_pts_matched = torch.nonzero(
                mask_query_pts)[:, 0].cpu().numpy()
            id_query_pts_matched = np.ones_like(idx_query_pts_matched)*(_id+1)

            idx_query_pts_matched_IDs += idx_query_pts_matched.tolist()
            id_query_pts_matched_IDs += id_query_pts_matched.tolist()

        # point NMS
        idx_query_pts, id_query_pts = idxPoint_NMS(np.array(idx_query_pts_matched_IDs), np.array(id_query_pts_matched_IDs),
                                                   score_query_pts, Hp, Wp, 3, 0.0)
        
        
        t3 = time.time()
        print('KP Time: {}ms'.format((t3-t2)*1000))
        
        # get candidate edges
        query_edges = []
        query_edge_scores = []
        query_edge_ids = []
        Np = len(idx_query_pts)
        if Np > 1:
            for i in range(Np-1):
                for j in range(i+1, Np):
                    if id_query_pts[i] == id_query_pts[j]:
                        continue
                    if not (id_query_pts[i]-1, id_query_pts[j]-1) in self.support_edges:
                        continue
                    
                    xi, yi = (idx_query_pts[i] % Wp,
                              idx_query_pts[i]//Wp)
                    xj, yj = (idx_query_pts[j] % Wp,
                              idx_query_pts[j]//Wp)
                    L = np.linalg.norm((xi-xj, yi-yj))
                    vx = (xj-xi)/L
                    vy = (yj-yi)/L
                    l = 1
                    efs = []
                    while l <= L-1:
                        x = np.around(xi+l*vx).astype(int)
                        y = np.around(yi+l*vy).astype(int)
                        idx = y*Wp+x
                        efs.append(Xq[:, :, idx:idx+1, :])
                        l += 1
                    Nsap = len(efs)
                    Nseg = self.n_edge_seg

                    lseg = Nsap/Nseg
                    efss = [[] for nseg in range(Nseg)]

                    for nseg in range(Nseg):
                        for ie in range(int(nseg*lseg), max(int(nseg*lseg+1), int((nseg+1)*lseg))):
                            efss[nseg].append(efs[ie])

                    sum_seg_scores = 0
                    cnt_valid_seg = 0
                    sef = support_edge_ftrs1[self.edge_idx_dict[(
                        id_query_pts[i]-1, id_query_pts[j]-1)]]
                    for n,efs in enumerate(efss):
                        ef_pool = torch.mean(torch.concat(efs, dim=-2), dim=-2, keepdims=True)
                        seg_score = torch.nn.CosineSimilarity(dim=-1)(ef_pool, sef[:,:,:,n*ef_pool.shape[-1]:(n+1)*ef_pool.shape[-1]])
                        sum_seg_scores += seg_score

                    
                    score = sum_seg_scores.cpu().numpy()/Nseg
                    
                    if False or score > 0.3:
                        query_edges.append((i, j))
                        query_edge_scores.append(score)
                        query_edge_ids.append(
                            self.edge_idx_dict[(id_query_pts[i]-1, id_query_pts[j]-1)])

        # edge NMS
        query_edges = edge_NMS(query_edges, query_edge_ids, query_edge_scores)

        # grouping
        query_edge_groups = edges_grouping(query_edges)
        

        t4 = time.time()
        print('Group Time: {}ms'.format((t4-t3)*1000))
        
        
        
        
        # make point groups, convert idx to uv coords,
        def idx2uv(idx_pt):
            x = idx_pt % Wp
            y = idx_pt//Wp
            u = x*self.stride+(self.patch_size-1)/2
            v = y*self.stride+(self.patch_size-1)/2
            u = int(u*self.Wq/self.input_size)
            v = int(v*self.Hq/self.input_size)
            return (u,v)
        
        
        
        query_kp_uv_groups = []
        query_kp_id_groups = [] 
        query_kp_edge_groups = []

        for eg in query_edge_groups:
            
            uv_group, id_group, visited = [],[],[]
            mapping_dict = {}
            # collect points
            for e in eg:
                # new point
                if not e[0] in visited:
                    uv_group.append(idx2uv(idx_query_pts[e[0]]))
                    id_group.append(id_query_pts[e[0]])
                    mapping_dict[e[0]]=len(uv_group)-1
                    visited.append(e[0])
                if not e[1] in visited:
                    uv_group.append(idx2uv(idx_query_pts[e[1]]))
                    id_group.append(id_query_pts[e[1]])
                    mapping_dict[e[1]]=len(uv_group)-1
                    visited.append(e[1])

            # modify edges
            edge_group = [(mapping_dict[e[0]],mapping_dict[e[1]]) for e in eg]
            
            if len(self.support_xys)==2 or len(self.support_xys)==3:
                min_pts = 2
            elif len(self.support_xys)==4:
                min_pts = 3
            elif len(self.support_xys)>4:
                min_pts = 4
                
            
            if False or (len(uv_group)<=len(self.support_xys) and len(uv_group)>=min_pts):
                query_kp_uv_groups.append(uv_group)
                query_kp_id_groups.append(id_group)
                query_kp_edge_groups.append(edge_group)
        
        
        return query_kp_uv_groups,query_kp_id_groups,query_kp_edge_groups
        

