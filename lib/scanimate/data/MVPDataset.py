# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from torch.utils.data import Dataset
import numpy as np
import os
import random
import pickle
import torch
import trimesh
import logging
from tqdm import tqdm
from glob import glob
from ..utils.net_util import homogenize
from ..utils.mesh_util import save_obj_mesh
from ..utils.ext_trimesh import sample_surface_wnormal
from ..utils.geo_util import computeMeanCurvature, compute_normal_v, detectBoundary

log = logging.getLogger('trimesh')
log.setLevel(40)


def filter_smpl_pose(pose, clip_value=0.1):
    ''' filter noise on wrist (20, 21), hand (22, 23), ankle (7,8) and foot (10, 11) joints
    hand and foot should have 0 pose angles
    wrist can have clip_value freedom to 3 directions
    ankle should have 2*clip_value freedom to 3 directions
    '''

    if len(pose.shape) == 1:  # no batch dim:
        pose[66:72] *= 0
        pose[30:36] *= 0
        pose[60:66] = np.clip(pose[60:66], -clip_value, clip_value)
        pose[21:27] = np.clip(pose[21:27], -2 * clip_value, 2 * clip_value)

    elif len(pose.shape) == 2:  # with batch dim:
        pose[:, 66:72] *= 0
        pose[:, 30:36] *= 0
        pose[:, 60:66] = np.clip(pose[:, 60:66], -clip_value, clip_value)
        pose[:, 21:27] = np.clip(pose[:, 21:27], -2 * clip_value, 2 * clip_value)

    return pose


def load_trimesh(ply_files, param_files):
    meshs = {}
    logging.info("In total there are " + str(len(ply_files)) + " frames.")
    logging.info(ply_files[0].split('/')[-3].split('_')[0])
    key_name = ply_files[0].split('/')[-3].split('_')[0]
    logging.warning("Key name of each frame looks like %s" % key_name)
    logging.info("Loading meshes...")

    for f, smpl_file in tqdm(zip(ply_files, param_files)):
        key_name = f.split('/')[-4].split('_')[0]
        tmp_mesh = trimesh.load(f, process=False, maintain_order=True)

        rot_path = '%s/FViewPOSE/R0.txt' % os.path.dirname(f).replace('Data', 'Params')
        # os.path.join(os.path.dirname(f), 'R0.txt')
        R = np.loadtxt(rot_path)
        tmp_mesh.vertices = np.matmul(tmp_mesh.vertices, R[:3, :3].T)

        smpl_data = np.load(smpl_file)
        tmp_mesh.vertices -= 0.5 * (tmp_mesh.vertices.max(0) + tmp_mesh.vertices.min(0))[None]
        tmp_mesh.vertices -= smpl_data['transl']
        tmp_mesh.vertices /= smpl_data['scale']

        meshs[key_name] = tmp_mesh
        del tmp_mesh
    return meshs


g_mesh_dic = None
g_flag_tri = None
g_valid_tri = None
g_sample_dic = None
g_smpl_data_dic = None


class MVPDataset_scan(Dataset):  # Single subject under data directory!
    def __init__(self, opt, phase='train', smpl=None, smpl_processing=True,
                 ground_level=(0, -0.558, 0), customized_minimal_ply='',
                 full_test=True, device=torch.device('cuda:0'),
                 ext='.obj'):
        logging.info("Loading %s data..." % phase)

        self.g_mesh_dic = None
        self.g_flag_tri = None
        self.g_sample_dic = None
        self.g_smpl_data_dic = None
        self.g_valid_tri = None

        self.opt = opt
        self.phase = phase
        self.is_train = (phase == 'train')
        self.ext = ext

        self.smpl = smpl
        self.full_test = full_test

        self.device = device

        # Path setup
        self.root = self.opt['data_dir']
        if not self.is_train:
            self.root = self.opt['test_dir']
        self.subjects = ['scan']
        self.subjects_minimal_v, self.Tpose_minimal_v = self.get_subjects_minimal()

        self.subjects_minimal_v = torch.Tensor(self.subjects_minimal_v).float().unsqueeze(0).to(self.device)
        self.Tpose_minimal_v = torch.Tensor(self.Tpose_minimal_v).float().unsqueeze(0).to(self.device)
        self.n_subject = len(self.subjects)
        self.subject_map = {}
        for i, k in enumerate(self.subjects):
            self.subject_map[k] = i

        if 'frame_ratio' in self.opt:
            self.pkl_data = self.collect_data(self.opt['frame_ratio'])
        else:
            self.pkl_data = self.collect_data()

        pkl_data_list = []  # for SMPL body paras
        ply_data_list = []  # for scans
        data_ids = []
        invalid_pair = 0
        for k, v in self.pkl_data.items():
            smpl_dir = os.path.join(self.root, 'smpl')
            scan_dir = '/media/liaotingting/usb/Data/%s/indoor/Action03' % self.root.split('/')[-1]
            if not os.path.exists(scan_dir):
                scan_dir = '/media/liaotingting/usb2/Data/%s/indoor/Action03' % self.root.split('/')[-1]

            for f_name, i in v:
                try:
                    pkl_data_list.append(os.path.join(smpl_dir, f_name))
                    ply_data_list.append(glob(os.path.join(scan_dir, '*' + self.ext))[0])
                    data_ids.append(i)
                except:
                    logging.info(os.path.join(scan_dir, f_name[:-4] + self.ext))
                    invalid_pair += 1
        logging.info("Number of valid_pair %s", str(len(pkl_data_list)))
        logging.info("Number of invalid_pair %s", str(invalid_pair))

        self.ground_level = ground_level
        self.pkl_data_list = pkl_data_list
        self.data_id_list = data_ids
        assert len(ply_data_list) > 0
        if self.g_mesh_dic is None and self.is_train:
            logging.info('loading meshes')
            self.g_mesh_dic = load_trimesh(ply_data_list, pkl_data_list)
            self.g_flag_tri = self.compute_convex()
        self.ply_data_list = ply_data_list
        self.g_sample_dic = {}
        self.resample_flag = True
        self.smpl_processing = smpl_processing
        if self.smpl_processing:
            self.smpl_process()

    def smpl_process(self):
        self.g_smpl_data_dic = {}
        logging.info('Processing smpl para to smpl data...')

        for pkl_file in tqdm(self.pkl_data_list):
            ply_file = pkl_file[:-4] + self.ext
            dic_key = ply_file.split('/')[-3].split('_')[0]

            smpl_data = np.load(pkl_file)
            betas = torch.zeros(10).float()
            body_pose = torch.Tensor(smpl_data['pose'])
            transl = torch.Tensor(smpl_data['transl']) * 0

            self.g_smpl_data_dic[dic_key] = {
                'betas': betas,
                'body_pose': body_pose,
                'transl': transl,
            }

            betas = betas.unsqueeze(0).to(self.device)
            body_pose = body_pose.unsqueeze(0).to(self.device)
            transl = transl.unsqueeze(0).to(self.device)
            global_orient = body_pose[:, :3]
            body_pose = body_pose[:, 3:]

            output = self.smpl(betas=betas, body_pose=body_pose,
                               global_orient=global_orient,
                               transl=transl,
                               return_verts=True, custom_out=True,
                               body_neutral_v=self.Tpose_minimal_v.expand(body_pose.shape[0], -1, -1)
                               )
            # for _, mesh in self.g_mesh_dic.items():
            #     save_obj_mesh('/media/liaotingting/usb/mesh.obj', mesh.vertices, mesh.faces)
            # save_obj_mesh('/media/liaotingting/usb/smpl.obj', output.vertices[0].cpu().numpy())
            # exit()
            smpl_posed_joints = output.joints
            rootT = self.smpl.get_root_T(global_orient, transl, smpl_posed_joints[:, 0:1, :])
            smpl_neutral = output.v_shaped
            smpl_cano = output.v_posed.permute(0, 2, 1)
            smpl_posed = output.vertices.contiguous()
            smpl_face = torch.LongTensor(self.smpl.faces[:, [0, 2, 1]].astype(np.int32))[None].to(self.device)
            smpl_n_posed = compute_normal_v(smpl_posed, smpl_face.expand(smpl_posed.shape[0], -1, -1))
            bmax = smpl_posed.max(1)[0]
            bmin = smpl_posed.min(1)[0]
            offset = 0.2 * (bmax - bmin)
            bmax += offset
            bmin -= offset
            jT = output.joint_transform[:, :24]
            inv_rootT = torch.inverse(rootT)

            self.g_smpl_data_dic[dic_key]['smpl_data'] = {
                'smpl_neutral': smpl_neutral.squeeze(0).cpu(),
                'smpl_cano': smpl_cano.squeeze(0).cpu(),
                'smpl_posed': smpl_posed.squeeze(0).cpu(),
                'smpl_n_posed': smpl_n_posed.squeeze(0).cpu(),
                'bmax': bmax.squeeze(0).cpu(),
                'bmin': bmin.squeeze(0).cpu(),
                'jT': jT.squeeze(0).cpu(),
                'inv_rootT': inv_rootT.squeeze(0).cpu()
            }

    def get_subjects_minimal(self):
        file = os.path.join(self.root, 'smpl', 'minimally.obj')
        T_pose_mesh = trimesh.load(file, process=False, maintain_order=True)
        logging.info('Loading minimally dressed body shape mesh ' + file)

        vitruvian_vertices = self.smpl.initiate_vitruvian(device=self.device,
                                                          body_neutral_v=torch.tensor(
                                                              T_pose_mesh.vertices).float().unsqueeze(0).to(
                                                              self.device)).detach().clone().cpu().numpy()[0]

        return vitruvian_vertices, T_pose_mesh.vertices

    def collect_data(self, frame_ratio=None):
        data = {}
        cnt = 0
        for sub in self.subjects:

            smpl_dir = os.path.join(self.root, 'smpl')
            data[sub] = sorted([f for f in os.listdir(smpl_dir) if '.npz' in f])

            if self.is_train:
                if frame_ratio:
                    pick_flag = []
                    if frame_ratio == 1.0:
                        pick_flag = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    elif frame_ratio == 0.8:
                        pick_flag = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
                    elif frame_ratio == 0.6:
                        pick_flag = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
                    elif frame_ratio == 0.5:
                        pick_flag = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
                    elif frame_ratio == 0.4:
                        pick_flag = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
                    elif frame_ratio == 0.2:
                        pick_flag = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                    elif frame_ratio == 0.1:
                        pick_flag = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif frame_ratio == 0.05:
                        pick_flag = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    else:
                        logging.error('Undefined frame_ratio, please choose from 1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1')
                        exit()
                    tmp_list = []
                    for i, content in enumerate(data[sub]):
                        if pick_flag[i % len(pick_flag)]:
                            tmp_list.append(content)
                    data[sub] = tmp_list

            data[sub] = tuple(zip(data[sub], list(range(cnt, cnt + len(data[sub])))))
            cnt += len(data[sub])
        return data

    def compute_convex(self):
        flag_tri = {}
        for k in self.g_mesh_dic.keys():
            mesh = self.g_mesh_dic[k]

            samples = mesh.vertices
            normals = mesh.vertex_normals
            faces = mesh.faces[:, [0, 2, 1]]

            samples = torch.Tensor(samples)[None].to(self.device)
            normals = torch.Tensor(normals.copy())[None].to(self.device)
            faces = torch.LongTensor(faces)[None].to(self.device)

            convex = (-computeMeanCurvature(samples, normals, faces)[:, :, 0]) < 0.2
            flag = convex[0, faces[0, :, 0]] & convex[0, faces[0, :, 1]] & convex[0, faces[0, :, 2]]
            flag_tri[k] = flag.cpu()

        return flag_tri

    def compute_valid_tri(self, lbs_net_p, smpl, lat_vecs_lbs_p, smpl_vitruvian):
        smpl_face = torch.LongTensor(smpl.faces[:, [0, 2, 1]].astype(np.int32))[None].to(self.device)

        if self.g_valid_tri is None:
            self.g_valid_tri = {}

        with torch.no_grad():
            for index in range(len(self.pkl_data_list)):
                pkl_file = self.pkl_data_list[index]
                ply_file = pkl_file[:-4] + self.ext
                ply_key = ply_file.split('/')[-3].split('_')[0]
                flag_tri = self.g_flag_tri[ply_key].to(self.device)

                smpl_data = np.load(pkl_file)
                betas = torch.zeros(10).float()[None].to(self.device)
                body_pose = torch.Tensor(smpl_data['pose'])[None].to(self.device)
                transl = torch.Tensor(smpl_data['transl'])[None].to(self.device) * 0

                f_ids = torch.LongTensor([self.data_id_list[index]]).to(self.device)

                mesh = self.g_mesh_dic[ply_key]
                samples = mesh.vertices
                faces = mesh.faces[:, [0, 2, 1]]
                samples = torch.Tensor(samples).t()[None].to(self.device)
                faces = torch.LongTensor(faces).to(self.device)

                global_orient = body_pose[:, :3]
                body_pose = body_pose[:, 3:]

                output = self.smpl(betas=betas, body_pose=body_pose, global_orient=0 * global_orient, transl=0 * transl,
                                   return_verts=True, custom_out=True,
                                   body_neutral_v=self.Tpose_minimal_v.expand(body_pose.shape[0], -1, -1))
                smpl_posed_joints = output.joints
                rootT = self.smpl.get_root_T(global_orient, transl, smpl_posed_joints[:, 0:1, :])

                smpl_neutral = output.v_shaped
                smpl_cano = output.v_posed.permute(0, 2, 1)
                smpl_posed = output.vertices.contiguous()
                bmax = smpl_posed.max(1)[0]
                bmin = smpl_posed.min(1)[0]
                offset = 0.2 * (bmax - bmin)
                bmax += offset
                bmin -= offset
                smpl_n_posed = compute_normal_v(smpl_posed, smpl_face.expand(smpl_posed.shape[0], -1, -1))
                jT = output.joint_transform[:, :24]
                inv_rootT = torch.inverse(rootT)
                scan_v_posed = torch.einsum('bst,btv->bsv', inv_rootT, homogenize(samples, 1))[:, :3,
                               :]  # remove root transform

                if lbs_net_p.opt['g_dim'] > 0:
                    lat = lat_vecs_lbs_p(f_ids)  # (B, Z)
                    lbs_net_p.set_global_feat(lat)

                feat3d_posed = None
                res_lbs_p = lbs_net_p(feat3d_posed, scan_v_posed, jT=jT, bmin=bmin[:, :, None], bmax=bmax[:, :, None])

                pred_scan_cano = res_lbs_p['pred_smpl_cano']

                e1 = torch.norm(scan_v_posed[0, :, faces[:, 0]] - scan_v_posed[0, :, faces[:, 1]], p=2, dim=0,
                                keepdim=True)
                e2 = torch.norm(scan_v_posed[0, :, faces[:, 1]] - scan_v_posed[0, :, faces[:, 2]], p=2, dim=0,
                                keepdim=True)
                e3 = torch.norm(scan_v_posed[0, :, faces[:, 2]] - scan_v_posed[0, :, faces[:, 0]], p=2, dim=0,
                                keepdim=True)
                e = torch.cat([e1, e2, e3], 0)

                E1 = torch.norm(pred_scan_cano[0, :, faces[:, 0]] - pred_scan_cano[0, :, faces[:, 1]], p=2, dim=0,
                                keepdim=True)
                E2 = torch.norm(pred_scan_cano[0, :, faces[:, 1]] - pred_scan_cano[0, :, faces[:, 2]], p=2, dim=0,
                                keepdim=True)
                E3 = torch.norm(pred_scan_cano[0, :, faces[:, 2]] - pred_scan_cano[0, :, faces[:, 0]], p=2, dim=0,
                                keepdim=True)
                E = torch.cat([E1, E2, E3], 0)

                max_edge = (E / e).max(0)[0]
                min_edge = (E / e).min(0)[0]
                mask = 1.0 - (((max_edge > 2.0) & flag_tri) | (max_edge > 3.0) | (min_edge < 0.1)).cpu().float().numpy()
                # mask = 1.0-(((max_edge > 6.0) & flag_tri) | (max_edge > 9.0) | (min_edge < 0.1)).cpu().float().numpy()
                tri_mask = mask > 0.5

                inside_mask = ((scan_v_posed[0, 0, faces[:, 0]] > bmin[0, 0]) & (
                        scan_v_posed[0, 0, faces[:, 0]] < bmax[0, 0]) & \
                               (scan_v_posed[0, 1, faces[:, 1]] > bmin[0, 1]) & (
                                       scan_v_posed[0, 1, faces[:, 1]] < bmax[0, 1]) & \
                               (scan_v_posed[0, 2, faces[:, 2]] > bmin[0, 2]) & (
                                       scan_v_posed[0, 2, faces[:, 2]] < bmax[0, 2])).cpu().numpy()
                tri_mask = tri_mask & inside_mask
                # save_obj_mesh('before%d.obj' %index, pred_scan_cano[0].t().cpu().numpy(), faces.cpu().numpy()[tri_mask])
                boundary = detectBoundary(faces.cpu().numpy()[tri_mask])
                tri_mask[tri_mask] = np.logical_not(boundary)
                self.g_valid_tri[ply_key] = tri_mask.astype(np.float32)
                # debug
                # print('saving to after%d.obj' %index)
                # save_obj_mesh('after%d.obj' %index, pred_scan_cano[0].t().cpu().numpy(), faces.cpu().numpy()[tri_mask.astype(np.float32)>0.5])
                # exit()

    def __len__(self):
        return len(self.pkl_data_list)

    def get_raw_scan_with_texture(self):
        from lib.common.mesh_util import load_obj_mesh
        vertices, faces, texture, texture_face = load_obj_mesh(self.ply_data_list[0], with_texture=True)
        return vertices, faces, texture, texture_face

    def get_raw_scan_face_and_mask(self, index=None, frame_id=None):
        if not frame_id is None:
            index = self.data_id_list.index(frame_id)

        pkl_file = self.pkl_data_list[index]
        ply_file = pkl_file[:-4] + self.ext
        ply_key = ply_file.split('/')[-3].split('_')[0]

        faces = self.g_mesh_dic[ply_key].faces
        # skip valid triangle
        if self.g_valid_tri is not None:
            mask = self.g_valid_tri[ply_key] > 0.5
            return faces, mask
        return faces

    def get_item(self, f_id, sub_id, subject, pkl_file):
        ply_file = pkl_file[:-4] + self.ext
        ply_key = ply_file.split('/')[-3].split('_')[0]

        betas = self.g_smpl_data_dic[ply_key]['betas']
        body_pose = self.g_smpl_data_dic[ply_key]['body_pose']
        transl = self.g_smpl_data_dic[ply_key]['transl']
        smpl_data = self.g_smpl_data_dic[ply_key]['smpl_data']

        if self.phase == 'train':
            mesh = self.g_mesh_dic[ply_key]
            flag = self.g_flag_tri[ply_key]

            samples = mesh.vertices

            # save_obj_mesh('/media/liaotingting/usb/sam.obj',
            #               mesh.vertices[np.random.randint(0, len(mesh.vertices), 10000)])

            faces = mesh.faces[:, [0, 2, 1]]

            samples_cape = torch.Tensor(samples)
            faces_cape = torch.LongTensor(faces)

            tri_id = torch.randperm(faces_cape.shape[0])[:self.opt['num_sample_edge']]
            faces_select = faces_cape[tri_id]
            tri_samples_cape = samples_cape[faces_select].permute(2, 0, 1).view(3, -1)  # (xyz, F, 3)
            flag = flag[tri_id].float()
            w_tri = flag + 0.0 * (1.0 - flag)

            mask = None
            if self.g_valid_tri is not None:
                mask = self.g_valid_tri[ply_key]

            if self.resample_flag:
                samples, normals, face_id = sample_surface_wnormal(mesh, self.opt['num_sample_surf'], mask)
                self.g_sample_dic[ply_key] = {'samples': samples, 'normals': normals, 'face_id': face_id}
            else:
                if not ply_key in self.g_sample_dic:
                    samples, normals, face_id = sample_surface_wnormal(mesh, self.opt['num_sample_surf'], mask)
                    self.g_sample_dic[ply_key] = {'samples': samples, 'normals': normals, 'face_id': face_id}
                else:
                    samples = self.g_sample_dic[ply_key]['samples']
                    normals = self.g_sample_dic[ply_key]['normals']
                    face_id = self.g_sample_dic[ply_key]['face_id']

            faces = mesh.faces[face_id]

            samples = torch.Tensor(samples)
            normals = torch.Tensor(normals)
            faces = torch.LongTensor(faces)

            res = {
                'frame_name': ply_key,
                'subject': subject,
                'betas': betas,
                'body_pose': body_pose,
                'transl': transl,
                # 'scan_cano': scan_cano,
                'scan_tri_posed': tri_samples_cape,
                'scan_cano_uni': samples,
                'normals_uni': normals,
                'faces_uni': faces,
                'sub_id': sub_id,
                'f_id': f_id,
                'w_tri': w_tri,
                'smpl_data': smpl_data
            }

            if not self.is_train:
                res.update({'scan_posed': samples_cape, 'faces': faces_cape})
        else:
            res = {
                'frame_name': ply_key,
                'subject': subject,
                'betas': betas,
                'body_pose': body_pose,
                'transl': transl,
                'sub_id': sub_id,
                'f_id': f_id
            }

            betas = betas.unsqueeze(0).to(self.device)
            body_pose = body_pose.unsqueeze(0).to(self.device)
            transl = transl.unsqueeze(0).to(self.device)
            global_orient = body_pose[:, :3]
            body_pose = body_pose[:, 3:]

            output = self.smpl(betas=betas, body_pose=body_pose, global_orient=0 * global_orient, transl=0 * transl,
                               return_verts=True, body_neutral_v=self.Tpose_minimal_v, custom_out=True)
            smpl_posed_joints = output.joints
            rootT = self.smpl.get_root_T(global_orient, transl, smpl_posed_joints[:, 0:1, :])
            res['rootT'] = rootT.squeeze(0).cpu()

        return res

    def get_item_by_index(self, index):
        pkl_file = self.pkl_data_list[index]
        f_id = self.data_id_list[index]
        subject = self.subjects[0]
        sub_id = self.subject_map[subject]
        return self.get_item(f_id, sub_id, subject, pkl_file)

    def __getitem__(self, index):
        return self.get_item_by_index(index)
