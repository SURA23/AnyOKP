import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
import math
import cv2


color_table = [(0, 0, 0),  (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),(255, 50, 100),
               (50, 100, 230), (138, 43, 150), (220, 100, 200), (200, 255, 125), (200, 190, 140), (100, 255, 180), (150, 150, 200), (250, 100, 55), (130, 150, 230), (210, 110, 160), (90, 215, 155)]


def edge_NMS(edges,edge_ids,edge_scores):
    Ne = len(edges)
    edge_valids = [1 for n in range(Ne)]
    for n in range(Ne):
        for k in range(Ne):
            if k == n:
                continue
            # same vertex
            if edge_ids[n] == edge_ids[k] and (edges[n][0] == edges[k][0] or edges[n][1] == edges[k][1] or
                                                           edges[n][1] == edges[k][0] or edges[n][0] == edges[k][1]):
                if edge_scores[n] > edge_scores[k]:
                    edge_valids[k] = 0
                else:
                    edge_valids[n] = 0
    edges = [edges[n] for n in range(Ne) if edge_valids[n]]
    return edges



def edges_grouping(edges):

    Ne = len(edges)
    if Ne==0:
        return []
    elif Ne==1:
        return [[edges[0]]]
    else:
        edge_groups = [[edge] for edge in edges]
        while True:
            Ng = len(edge_groups)
            if Ng == 1:
                break
            updated = False
            for i in range(0,Ng-1):
                for j in range(i+1,Ng):
                    edge_group1 = edge_groups[i]
                    edge_group2 = edge_groups[j]
                    connected = False
                    for e1 in edge_group1:
                        for e2 in edge_group2:
                            if e1[0]==e2[0] or e1[1]==e2[0] or e1[0]==e2[1] or e1[1]==e2[1]:
                                connected = True
                                break
                        if connected:
                            break
                    if connected:
                        edge_groups[i] += edge_group2
                        edge_groups[j]=[]
                        updated = True
            edge_groups = [edge_group for edge_group in edge_groups if len(edge_group)>0]
            if updated==False:
                break
    
    return edge_groups



# 1-D arrays of point indexs/scores/ID
def idxPoint_NMS(idx_pts, id_pts, score_pts, H, W, dist_thresh=4, score_thresh=0.3):
    N_idxs = idx_pts.shape[0]

    if N_idxs <= 1:
        idx_pts1 = idx_pts
        score_pts1 = score_pts
        id_pts1 = id_pts
    elif N_idxs > 1:
        idx_pts1 = []
        score_pts1 = []
        id_pts1 = []

        for i in range(N_idxs):
            xi = idx_pts[i] % W
            yi = idx_pts[i]//W
            si = score_pts[idx_pts[i]]
            if si < score_thresh:
                continue
            is_max = True

            for j in range(N_idxs):
                if i == j:
                    continue
                xj = idx_pts[j] % W
                yj = idx_pts[j]//W
                sj = score_pts[idx_pts[j]]
                if abs(xi-xj) <= dist_thresh and abs(yi-yj) <= dist_thresh:
                    if si < sj:
                        is_max = False

                if not is_max:
                    break

            if is_max:
                idx_pts1.append(idx_pts[i])
                score_pts1.append(score_pts[i])
                id_pts1.append(id_pts[i])

    return np.array(idx_pts1), np.array(id_pts1)


def preprocess_numpy(image, load_size=None, is_gray = False) -> torch.Tensor:
    """
    Preprocesses an image before extraction.
    :param image: rgb image to be extracted.
    :param load_size: optional. Size to resize image before the rest of preprocessing.
    :return: a tuple containing:
                (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                (2) the pil image in relevant dimensions
    """
    pil_image = Image.fromarray(image)
    if load_size is not None:
        pil_image = transforms.Resize(
            (load_size, load_size), interpolation=transforms.InterpolationMode.BILINEAR)(pil_image)
    if is_gray:
        prep = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor() ])
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    prep_img = prep(pil_image)[None, ...]
    return prep_img



def log_bin( x: torch.Tensor,  num_patches:Tuple[int, int], hierarchy: int = 2) -> torch.Tensor:
    """
    create a log-binned descriptor.
    :param x: tensor of features. Has shape Bxhxtxd.
    :param hierarchy: how many bin hierarchies to use.
    """
    B = x.shape[0]
    num_bins = 1 + 8 * hierarchy

    bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
    bin_x = bin_x.permute(0, 2, 1)
    bin_x = bin_x.reshape(B, bin_x.shape[1], num_patches[0], num_patches[1]) # 1*768*55*55
    # Bx(dxh)xnum_patches[0]xnum_patches[1]
    sub_desc_dim = bin_x.shape[1]

    avg_pools = []
    # compute bins of all sizes for all spatial locations.
    for k in range(0, hierarchy):
        # avg pooling with kernel 3**kx3**k
        win_size = 3 ** (k)
        avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
        avg_pools.append(avg_pool(bin_x))

    bin_x = torch.zeros((B, sub_desc_dim * num_bins, num_patches[0], num_patches[1])).cuda()
    for y in range(num_patches[0]):
        for x in range(num_patches[1]):
            part_idx = 0
            # fill all bins for a spatial location (y, x)
            for k in range(0, hierarchy):
                kernel_size = 3 ** (k)
                for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                    for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                        if i == y and j == x and k != 0:
                            continue
                        if 0 <= i < num_patches[0] and 0 <= j < num_patches[1]:
                            bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                        :, :, i, j]
                        else:  # handle padding in a more delicate way than zero padding
                            temp_i = max(0, min(i, num_patches[0] - 1))
                            temp_j = max(0, min(j, num_patches[1] - 1))
                            bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                        :, :, temp_i,
                                                                                                        temp_j]
                        part_idx += 1
    bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
    # Bx1x(t-1)x(dxh)
    return bin_x






def fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
    """
    Creates a method for position encoding interpolation.
    :param patch_size: patch size of the model.
    :param stride_hw: A tuple containing the new height and width stride respectively.
    :return: the interpolation method
    """
    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size) // stride_hw[1]
        h0 = 1 + (h - patch_size) // stride_hw[0]
        assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                        stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding

def vis_keypoint_groups(img_show,kp_uv_groups,kp_id_groups,kp_edge_groups,r_draw=8,w_draw=4):
    Ng = len(kp_uv_groups)
    for n in range(Ng):
        uv_group,id_group,edge_group = kp_uv_groups[n],kp_id_groups[n],kp_edge_groups[n]
        for eij in edge_group:
            i, j = eij
            cv2.line(img_show, uv_group[i],uv_group[j], color_table[-n-1], w_draw)
        for uv,_id in zip(uv_group,id_group):
            u,v = uv
            cv2.circle(img_show, (u, v), r_draw, color_table[_id], r_draw//2)

    return img_show
        
def vis_edges(img_show,edge_groups,keypoints,w_draw=2):
    for ng, group in enumerate(edge_groups):
        for n, eij in enumerate(group):
            i, j = eij
            cv2.line(img_show, keypoints[i],keypoints[j], color_table[-(ng)], 2)
    return img_show

def select_keypoints(img):
    label_pts = []
    show = img.copy()
    # draw grid
    H, W, _ = show.shape
    l = 8
    for j in range(int(W/l)):
        for i in range(int(H/l)):
            show[int(i*l), int(j*l), :] *= 0

    # label on 1st image
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            label_pts.append((x, y))
            for n, pt in enumerate(label_pts):
                cv2.circle(show, pt, 10, color_table[n+1], 5)
            cv2.imshow('UI', show)

    cv2.namedWindow('UI')
    cv2.setMouseCallback('UI', on_mouse)
    cv2.imshow('UI', show)
    cv2.waitKey(0)
#    cv2.destroyWindow('UI')
    return label_pts    


def load_support_labels(file):
    keypoints,edges = [],[]
    with open(file, 'r') as f:
        data = f.read()
        datas = data[2:-1].split('\nE\n')
        data_pts = datas[0].split('\n')
        data_edges= datas[1].split('\n')
        for data_pt in data_pts:
            if len(data_pt)>0:
                u,v = data_pt.split(',')
                keypoints.append((int(u),int(v)))
        for data_edge in data_edges:
            if len(data_edge)>0:
                i,j = data_edge.split(',')
                edges.append((int(i),int(j)))
    return keypoints,edges
    
# save support labels
def save_support_labels(file, keypoints, edges):
    with open(file, 'w') as f:
        f.write('P:\n')
        for kp in keypoints:
            f.write('{},{}\n'.format(kp[0], kp[1]))
        f.write('E\n')
        for e in edges:
            f.write('{},{}\n'.format(e[0], e[1]))
    f.close()
    
def load_img(img_file, resize_W, resize_H, square_pad=True):
    img = cv2.imread(img_file)
    if square_pad:
        h0,w0,_ = img.shape
        if h0<=w0:
            pad1 = (w0-h0)//2
            pad2 = w0-h0-pad1
            img = np.vstack([np.zeros((pad1,w0,3),np.uint8),img, np.zeros((pad2,w0,3),np.uint8)])
        else:
            pad1 = (h0-w0)//2
            pad2 = h0-w0-pad1
            img = np.hstack([np.zeros((h0,pad1,3),np.uint8),img, np.zeros((h0,pad2,3),np.uint8)])
    img = cv2.resize(img, (resize_W, resize_H))
    return img