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

import os
import math
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
import json
from pathlib import Path

import smpl
from lib.scanimate.utils.config import load_config
from lib.scanimate.utils.geo_util import compute_normal_v
from lib.scanimate.utils.mesh_util import reconstruction, save_obj_mesh, replace_hands_feet, replace_hands_feet_mesh
from lib.scanimate.utils.net_util import batch_rod2quat, homogenize, load_network, get_posemap, compute_knn_feat
from lib.scanimate.model.IGRSDFNet import IGRSDFNet
from lib.scanimate.model.LBSNet import LBSNet
from lib.scanimate.data.MVPDataset import MVPDataset_scan
from lib.scanimate.data.THuman import THumanDataset_scan
from lib.common.mesh_util import build_mesh_by_poisson
import logging

logging.basicConfig(level=logging.DEBUG)


def gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin,
              model, smpl_vitruvian, train_data_loader,
              cuda, name='', reference_body_v=None, every_n_frame=10):
    dataset = train_data_loader.dataset

    def process(data, idx=0):
        betas = data['betas'][None].to(device=cuda)
        body_pose = data['body_pose'][None].to(device=cuda)
        scan_posed = data['scan_posed'][None].to(device=cuda)
        transl = data['transl'][None].to(device=cuda)
        f_ids = torch.LongTensor([data['f_id']]).to(device=cuda)
        global_orient = body_pose[:, :3]
        body_pose = body_pose[:, 3:]
        output = model(betas=betas, body_pose=body_pose,
                       global_orient=0 * global_orient,
                       transl=0 * transl,
                       return_verts=True, custom_out=True,
                       body_neutral_v=reference_body_v.expand(body_pose.shape[0], -1, -1) if reference_body_v is not None else None)
        smpl_posed_joints = output.joints
        rootT = model.get_root_T(global_orient, transl, smpl_posed_joints[:, 0:1, :])

        smpl_posed = output.vertices.contiguous()
        bmax = smpl_posed.max(1)[0]
        bmin = smpl_posed.min(1)[0]
        offset = 0.2 * (bmax - bmin)
        bmax += offset
        bmin -= offset
        jT = output.joint_transform[:, :24]

        scan_posed = torch.einsum('bst,bvt->bsv', torch.inverse(rootT), homogenize(scan_posed))[:, :3,
                     :]  # remove root transform
        # remove invalid face and rebuild mesh
        faces = dataset.get_raw_scan_face_and_mask(frame_id=f_ids[0].cpu().numpy())
        _, _, texture, texture_face = dataset.get_raw_scan_with_texture()

        # todo replace hand and feat
        # pred_scan_cano_mesh = trimesh.Trimesh(vertices=pred_scan_cano[0].cpu().numpy(),
        #                                       faces=valid_scan_faces[:, [0, 2, 1]], process=False)
        # pred_body_neutral_mesh = trimesh.Trimesh(vertices=smpl_neutral[0].cpu().numpy(),
        #                                          faces=model.faces[:, [0, 2, 1]], process=False)
        # output_mesh = replace_hands_feet_mesh(pred_scan_cano_mesh, pred_body_neutral_mesh,
        #                                       vitruvian_angle=model.vitruvian_angle)

        # todo save obj mesh
        # valid_scan_faces = scan_faces[scan_mask, :]
        # t = time.time()
        # scan_posed, faces = build_mesh_by_poisson(
        #     scan_posed[0].transpose(0, 1).cpu().numpy(), valid_scan_faces[:, [0, 2, 1]], 150000)
        #
        # scan_posed = torch.from_numpy(scan_posed).float().to(device=cuda).transpose(0, 1)[None]

        if inv_skin_net.opt['g_dim'] > 0:
            lat = lat_vecs_inv_skin(f_ids)  # (B, Z)
            inv_skin_net.set_global_feat(lat)

        # Inference canonical mesh
        pred_scan_cano = []
        pred_lbs = []
        half = scan_posed.shape[2] // 2
        for points in scan_posed.split([half, scan_posed.shape[2] - half], 2):
            res_scan_p = inv_skin_net(None, points, jT=jT, bmin=bmin[:, :, None], bmax=bmax[:, :, None])
            pred_scan_cano.append(res_scan_p['pred_smpl_cano'])
            pred_lbs.append(res_scan_p['pred_lbs_smpl_posed'])
        pred_scan_cano = torch.cat(pred_scan_cano, -1).permute(0, 2, 1)[0].cpu().numpy()
        pred_lbs = torch.cat(pred_lbs, -1).permute(0, 2, 1)[0].cpu().numpy()

        # save_obj_mesh('%s/canon_origin.obj' % result_dir, pred_scan_cano, scan_faces)

        from lib.common.mesh_util import save_obj_data
        save_obj_data('%s/da-pose.obj' % result_dir, pred_scan_cano, faces, texture, texture_face)
        np.savez('%s/skin_weight.npz' % result_dir, skin_weight=pred_lbs)

    if name == '_pt3':
        logging.info("Outputing samples of canonicalization results...")
        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                if not i % every_n_frame == 0:
                    continue
                data = dataset[i]
                process(data, i)


def pretrain_skinning_net(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin,
                          model, smpl_vitruvian, gt_lbs_smpl,
                          train_data_loader, test_data_loader,
                          cuda, reference_body_v=None):
    optimizer_lbs_c = torch.optim.Adam(fwd_skin_net.parameters(), lr=opt['training']['lr_pt1'])
    optimizer_lbs_p = torch.optim.Adam([
        {
            "params": inv_skin_net.parameters(),
            "lr": opt['training']['lr_pt1'],
        },
        {
            "params": lat_vecs_inv_skin.parameters(),
            "lr": opt['training']['lr_pt1'],
        },
    ])
    smpl_face = torch.LongTensor(model.faces[:, [0, 2, 1]].astype(np.int32))[None].to(cuda)

    n_iter = 0
    for epoch in range(opt['training']['num_epoch_pt1']):
        fwd_skin_net.train()
        inv_skin_net.train()

        if epoch % opt['training']['resample_every_n_epoch'] == 0:
            train_data_loader.dataset.resample_flag = True
        else:
            train_data_loader.dataset.resample_flag = False

        if epoch == opt['training']['num_epoch_pt1'] // 2 or epoch == 3 * (opt['training']['num_epoch_pt1'] // 4):
            for j, _ in enumerate(optimizer_lbs_c.param_groups):
                optimizer_lbs_c.param_groups[j]['lr'] *= 0.1
            for j, _ in enumerate(optimizer_lbs_p.param_groups):
                optimizer_lbs_p.param_groups[j]['lr'] *= 0.1
        for train_idx, train_data in enumerate(train_data_loader):
            betas = train_data['betas'].to(device=cuda)
            body_pose = train_data['body_pose'].to(device=cuda)
            scan_posed = train_data['scan_cano_uni'].to(device=cuda)
            scan_tri = train_data['scan_tri_posed'].to(device=cuda)
            transl = train_data['transl'].to(device=cuda)
            f_ids = train_data['f_id'].to(device=cuda)
            smpl_data = train_data['smpl_data']
            global_orient = body_pose[:, :3]
            body_pose = body_pose[:, 3:]

            smpl_neutral = smpl_data['smpl_neutral'].cuda()
            smpl_cano = smpl_data['smpl_cano'].cuda()
            smpl_posed = smpl_data['smpl_posed'].cuda()
            smpl_n_posed = smpl_data['smpl_n_posed'].cuda()
            bmax = smpl_data['bmax'].cuda()
            bmin = smpl_data['bmin'].cuda()
            jT = smpl_data['jT'].cuda()
            inv_rootT = smpl_data['inv_rootT'].cuda()

            # Get rid of global rotation from posed scans
            scan_posed = torch.einsum('bst,bvt->bvs', inv_rootT, homogenize(scan_posed))[:, :, :3]

            reference_lbs_scan = compute_knn_feat(scan_posed, smpl_posed,
                                                  gt_lbs_smpl.expand(scan_posed.shape[0], -1, -1).permute(0, 2, 1))[:,
                                 :, 0].permute(0, 2, 1)
            scan_posed = scan_posed.permute(0, 2, 1)

            if opt['model']['inv_skin_net']['g_dim'] > 0:
                lat = lat_vecs_inv_skin(f_ids)  # (B, Z)
                inv_skin_net.set_global_feat(lat)

            feat3d_posed = None
            res_lbs_p, err_lbs_p, err_dict = inv_skin_net(feat3d_posed, smpl_posed.permute(0, 2, 1), gt_lbs_smpl,
                                                          scan=scan_posed, reference_lbs_scan=None, jT=jT,
                                                          bmin=bmin[:, :, None],
                                                          bmax=bmax[:, :, None])  # jT=jT, v_tri=scan_tri,

            feat3d_cano = None
            res_lbs_c, err_lbs_c, err_dict_lbs_c = fwd_skin_net(feat3d_cano, smpl_cano, gt_lbs_smpl,
                                                                scan=res_lbs_p['pred_scan_cano'].detach(),
                                                                reference_lbs_scan=reference_lbs_scan)  # , jT=jT, res_posed=res_lbs_p)

            # Back propagation
            err_dict.update(err_dict_lbs_c)
            err_dict['All-inv'] = err_lbs_p.item()
            err_dict['All-lbs'] = err_lbs_c.item()

            optimizer_lbs_p.zero_grad()
            err_lbs_p.backward()
            optimizer_lbs_p.step()

            optimizer_lbs_c.zero_grad()
            err_lbs_c.backward()
            optimizer_lbs_c.step()

            if n_iter % opt['training']['freq_plot'] == 0:
                err_txt = ''.join(['{}: {:.3f} '.format(k, v) for k, v in err_dict.items()])
                print('[%03d/%03d]:[%04d/%04d] %s' % (
                    epoch, opt['training']['num_epoch_pt1'], train_idx, len(train_data_loader), err_txt))
            n_iter += 1

    train_data_loader.dataset.is_train = False
    gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, train_data_loader,
              cuda, '_pt1', reference_body_v=reference_body_v)
    train_data_loader.dataset.is_train = True

    return optimizer_lbs_c, optimizer_lbs_p


def train_skinning_net(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin,
                       model, smpl_vitruvian, gt_lbs_smpl,
                       train_data_loader, test_data_loader,
                       cuda, reference_body_v=None, optimizers=None):
    if not optimizers == None:
        optimizer_lbs_c = optimizers[0]
        optimizer_lbs_p = optimizers[1]
    else:
        optimizer = torch.optim.Adam(list(fwd_skin_net.parameters()) + list(inv_skin_net.parameters()),
                                     lr=opt['training']['lr_pt2'])

    smpl_face = torch.LongTensor(model.faces[:, [0, 2, 1]].astype(np.int32))[None].to(cuda)

    o_cyc_smpl = fwd_skin_net.opt['lambda_cyc_smpl']
    o_cyc_scan = fwd_skin_net.opt['lambda_cyc_scan']
    n_iter = 0
    for epoch in range(opt['training']['num_epoch_pt2']):
        fwd_skin_net.train()
        inv_skin_net.train()
        if epoch % opt['training']['resample_every_n_epoch'] == 0:
            train_data_loader.dataset.resample_flag = True
        else:
            train_data_loader.dataset.resample_flag = False
        if epoch == opt['training']['num_epoch_pt2'] // 2 or epoch == 3 * (opt['training']['num_epoch_pt2'] // 4):
            fwd_skin_net.opt['lambda_cyc_smpl'] *= 10.0
            fwd_skin_net.opt['lambda_cyc_scan'] *= 10.0
            if not optimizers == None:
                for j, _ in enumerate(optimizer_lbs_c.param_groups):
                    optimizer_lbs_c.param_groups[j]['lr'] *= 0.1
                for j, _ in enumerate(optimizer_lbs_p.param_groups):
                    optimizer_lbs_p.param_groups[j]['lr'] *= 0.1
            else:
                for j, _ in enumerate(optimizer.param_groups):
                    optimizer.param_groups[j]['lr'] *= 0.1
        for train_idx, train_data in enumerate(train_data_loader):
            betas = train_data['betas'].to(device=cuda)
            body_pose = train_data['body_pose'].to(device=cuda)
            # scan_cano = train_data['scan_cano'].to(device=cuda).permute(0,2,1)
            scan_posed = train_data['scan_cano_uni'].to(device=cuda)
            scan_tri = train_data['scan_tri_posed'].to(device=cuda)
            w_tri = train_data['w_tri'].to(device=cuda)
            transl = train_data['transl'].to(device=cuda)
            f_ids = train_data['f_id'].to(device=cuda)
            smpl_data = train_data['smpl_data']
            global_orient = body_pose[:, :3]
            body_pose = body_pose[:, 3:]

            smpl_neutral = smpl_data['smpl_neutral'].cuda()
            smpl_cano = smpl_data['smpl_cano'].cuda()
            smpl_posed = smpl_data['smpl_posed'].cuda()
            smpl_n_posed = smpl_data['smpl_n_posed'].cuda()
            bmax = smpl_data['bmax'].cuda()
            bmin = smpl_data['bmin'].cuda()
            jT = smpl_data['jT'].cuda()
            inv_rootT = smpl_data['inv_rootT'].cuda()

            scan_posed = torch.einsum('bst,bvt->bsv', inv_rootT, homogenize(scan_posed))[:, :3,
                         :]  # remove root transform
            scan_tri = torch.einsum('bst,btv->bsv', inv_rootT, homogenize(scan_tri, 1))[:, :3, :]

            reference_lbs_scan = compute_knn_feat(scan_posed.permute(0, 2, 1), smpl_posed,
                                                  gt_lbs_smpl.expand(scan_posed.shape[0], -1, -1).permute(0, 2, 1))[:,
                                 :, 0].permute(0, 2, 1)

            if opt['model']['inv_skin_net']['g_dim'] > 0:
                lat = lat_vecs_inv_skin(f_ids)  # (B, Z)
                inv_skin_net.set_global_feat(lat)

            feat3d_posed = None
            res_lbs_p, err_lbs_p, err_dict = inv_skin_net(feat3d_posed, smpl_posed.permute(0, 2, 1), gt_lbs_smpl,
                                                          scan_posed, reference_lbs_scan=reference_lbs_scan, jT=jT,
                                                          v_tri=scan_tri, w_tri=w_tri, bmin=bmin[:, :, None],
                                                          bmax=bmax[:, :, None])

            feat3d_cano = None
            res_lbs_c, err_lbs_c, err_dict_lbs_c = fwd_skin_net(feat3d_cano, smpl_cano, gt_lbs_smpl,
                                                                res_lbs_p['pred_scan_cano'], jT=jT, res_posed=res_lbs_p)

            # Back propagation
            err_dict.update(err_dict_lbs_c)
            err = err_lbs_p + err_lbs_c
            err_dict['All'] = err.item()

            if not optimizers == None:
                optimizer_lbs_c.zero_grad()
                optimizer_lbs_p.zero_grad()
            else:
                optimizer.zero_grad()
            err.backward()
            if not optimizers == None:
                optimizer_lbs_c.step()
                optimizer_lbs_p.step()
            else:
                optimizer.step()

            if n_iter % opt['training']['freq_plot'] == 0:
                err_txt = ''.join(['{}: {:.3f} '.format(k, v) for k, v in err_dict.items()])
                print('[%03d/%03d]:[%04d/%04d] %s' % (
                    epoch, opt['training']['num_epoch_pt2'], train_idx, len(train_data_loader), err_txt))
            n_iter += 1

    fwd_skin_net.opt['lambda_cyc_smpl'] = o_cyc_smpl
    fwd_skin_net.opt['lambda_cyc_scan'] = o_cyc_scan

    train_data_loader.dataset.is_train = False
    gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, train_data_loader,
              cuda, '_pt2', reference_body_v=reference_body_v)
    train_data_loader.dataset.is_train = True


def train(opt, gen_cano_mesh=False, data_type='mvp'):
    cuda = torch.device('cuda:0')
    ckpt_dir = opt['experiment']['ckpt_dir']
    result_dir = opt['experiment']['result_dir']
    log_dir = opt['experiment']['log_dir']

    os.makedirs(result_dir, exist_ok=True)

    # Initialize vitruvian SMPL model
    if 'vitruvian_angle' not in opt['data']:
        opt['data']['vitruvian_angle'] = 25
        # opt['data']['vitruvian_angle'] = 0

    model = smpl.create(opt['data']['smpl_dir'], model_type='smpl_vitruvian',
                        gender=opt['data']['smpl_gender'], use_face_contour=False,
                        ext='npz').to(cuda)

    # Initialize dataset
    if data_type == 'mvp':
        train_dataset = MVPDataset_scan(opt['data'], smpl=model, device=cuda)
        test_dataset = MVPDataset_scan(opt['data'], smpl=model, device=cuda)
    else:
        train_dataset = THumanDataset_scan(opt['data'], smpl=model, device=cuda)
        test_dataset = THumanDataset_scan(opt['data'], smpl=model, device=cuda)

    reference_body_vs_train = train_dataset.Tpose_minimal_v
    reference_body_vs_test = test_dataset.Tpose_minimal_v

    smpl_vitruvian = model.initiate_vitruvian(device=cuda,
                                              body_neutral_v=train_dataset.Tpose_minimal_v,
                                              vitruvian_angle=opt['data']['vitruvian_angle'])

    train_data_loader = DataLoader(train_dataset, batch_size=opt['training']['batch_size'],
                                   num_workers=opt['training']['num_threads'], pin_memory=opt['training']['pin_memory'])
    test_data_loader = DataLoader(test_dataset)

    # All the hand, face joints are glued to body joints for SMPL
    gt_lbs_smpl = model.lbs_weights[:, :24].clone()
    root_idx = model.parents.cpu().numpy()
    idx_list = list(range(root_idx.shape[0]))
    for i in range(root_idx.shape[0]):
        if i > 23:
            root = idx_list[root_idx[i]]
            gt_lbs_smpl[:, root] += model.lbs_weights[:, i]
            idx_list[i] = root
    gt_lbs_smpl = gt_lbs_smpl[None].permute(0, 2, 1)

    smpl_vitruvian = model.initiate_vitruvian(device=cuda,
                                              body_neutral_v=train_dataset.Tpose_minimal_v,
                                              vitruvian_angle=opt['data']['vitruvian_angle'])

    # define bounding box
    bbox_smpl = (smpl_vitruvian[0].cpu().numpy().min(0).astype(np.float32),
                 smpl_vitruvian[0].cpu().numpy().max(0).astype(np.float32))
    bbox_center, bbox_size = 0.5 * (bbox_smpl[0] + bbox_smpl[1]), (bbox_smpl[1] - bbox_smpl[0])
    bbox_min = np.stack([bbox_center[0] - 0.55 * bbox_size[0], bbox_center[1] - 0.6 * bbox_size[1],
                         bbox_center[2] - 1.5 * bbox_size[2]], 0).astype(np.float32)
    bbox_max = np.stack([bbox_center[0] + 0.55 * bbox_size[0], bbox_center[1] + 0.6 * bbox_size[1],
                         bbox_center[2] + 1.5 * bbox_size[2]], 0).astype(np.float32)

    # Initialize networks
    pose_map = get_posemap(opt['model']['posemap_type'], 24, model.parents, opt['model']['n_traverse'],
                           opt['model']['normalize_posemap'])

    igr_net = IGRSDFNet(opt['model']['igr_net'], bbox_min, bbox_max, pose_map).to(cuda)
    fwd_skin_net = LBSNet(opt['model']['fwd_skin_net'], bbox_min, bbox_max, posed=False).to(cuda)
    inv_skin_net = LBSNet(opt['model']['inv_skin_net'], bbox_min, bbox_max, posed=True).to(cuda)

    lat_vecs_igr = nn.Embedding(1, opt['model']['igr_net']['g_dim']).to(cuda)
    lat_vecs_inv_skin = nn.Embedding(len(train_dataset), opt['model']['inv_skin_net']['g_dim']).to(cuda)

    torch.nn.init.constant_(lat_vecs_igr.weight.data, 0.0)
    torch.nn.init.normal_(lat_vecs_inv_skin.weight.data, 0.0, 1.0 / math.sqrt(opt['model']['inv_skin_net']['g_dim']))

    # print("igr_net:\n", igr_net)
    # print("fwd_skin_net:\n", fwd_skin_net)
    # print("inv_skin_net:\n", inv_skin_net)

    # Find checkpoints
    ckpt_dict = None
    if opt['experiment']['ckpt_file'] is not None:
        if os.path.isfile(opt['experiment']['ckpt_file']):
            logging.info('loading for ckpt...' + opt['experiment']['ckpt_file'])
            ckpt_dict = torch.load(opt['experiment']['ckpt_file'])
        else:
            logging.warning('ckpt does not exist [%s]' % opt['experiment']['ckpt_file'])
    elif opt['training']['continue_train']:
        model_path = '%s/ckpt_latest.pt' % ckpt_dir
        if os.path.isfile(model_path):
            logging.info('Resuming from ' + model_path)
            ckpt_dict = torch.load(model_path)
        else:
            logging.warning('ckpt does not exist [%s]' % model_path)
            opt['training']['use_trained_skin_nets'] = True
            model_path = '%s/ckpt_trained_skin_nets.pt' % ckpt_dir
            if os.path.isfile(model_path):
                logging.info('Resuming from ' + model_path)
                ckpt_dict = torch.load(model_path)
                logging.info('Pretrained model loaded.')
            else:
                logging.warning('ckpt does not exist [%s]' % model_path)
    elif opt['training']['use_trained_skin_nets']:
        model_path = '%s/ckpt_trained_skin_nets.pt' % ckpt_dir
        if os.path.isfile(model_path):
            logging.info('Resuming from ' + model_path)
            ckpt_dict = torch.load(model_path)
            logging.info('Pretrained model loaded.')
        else:
            logging.warning(
                'ckpt does not exist [%s] \n Failed to resume training, start training from beginning' % model_path)

    # Load checkpoints
    train_igr_start_epoch = 0
    if ckpt_dict is not None:
        if 'igr_net' in ckpt_dict:
            load_network(igr_net, ckpt_dict['igr_net'])
            if 'epoch' in ckpt_dict:
                train_igr_start_epoch = ckpt_dict['epoch']
        else:
            logging.warning("Couldn't find igr_net in checkpoints!")

        if 'fwd_skin_net' in ckpt_dict:
            load_network(fwd_skin_net, ckpt_dict['fwd_skin_net'])
        else:
            logging.warning("Couldn't find fwd_skin_net in checkpoints!")

        if 'inv_skin_net' in ckpt_dict:
            load_network(inv_skin_net, ckpt_dict['inv_skin_net'])
        else:
            logging.warning("Couldn't find inv_skin_net in checkpoints!")

        if 'lat_vecs_igr' in ckpt_dict:
            load_network(lat_vecs_igr, ckpt_dict['lat_vecs_igr'])
        else:
            logging.warning("Couldn't find lat_vecs_igr in checkpoints!")

        if 'lat_vecs_inv_skin' in ckpt_dict:
            load_network(lat_vecs_inv_skin, ckpt_dict['lat_vecs_inv_skin'])
        else:
            logging.warning("Couldn't find lat_vecs_inv_skin in checkpoints!")

    logging.info('train data size: %s' % str(len(train_dataset)))
    logging.info('test data size: %s' % str(len(test_dataset)))

    # Skip canonicalization
    if opt['training']['continue_train'] and os.path.isfile('%s/ckpt_trained_skin_nets.pt' % ckpt_dir):
        logging.info("Get fwd_skin_net, inv_skin_net and lat_vecs_inv_skin from trained skinning net!")
        trained_skin_nets_ckpt_dict = torch.load('%s/ckpt_trained_skin_nets.pt' % ckpt_dir)
        fwd_skin_net.load_state_dict(trained_skin_nets_ckpt_dict['fwd_skin_net'])
        inv_skin_net.load_state_dict(trained_skin_nets_ckpt_dict['inv_skin_net'])
        lat_vecs_inv_skin.load_state_dict(trained_skin_nets_ckpt_dict['lat_vecs_inv_skin'])

        opt['training']['skip_pt1'] = True
        opt['training']['skip_pt2'] = True

    # Pretrain fwd_skin_net and inv_skin_net net independently
    if not opt['training']['skip_pt1']:
        logging.info('start pretraining skinning nets (individual)')
        optimizers = pretrain_skinning_net(opt, log_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model,
                                           smpl_vitruvian, gt_lbs_smpl, train_data_loader, test_data_loader, cuda,
                                           reference_body_v=reference_body_vs_train)

    # Train fwd_skin_net and inv_skin_net jointly
    if not opt['training']['skip_pt2']:
        logging.info('start training skinning nets (joint)')
        train_skinning_net(opt, log_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian,
                           gt_lbs_smpl, train_data_loader, test_data_loader, cuda,
                           reference_body_v=reference_body_vs_train, optimizers=optimizers)

    if not opt['training']['skip_pt1'] and not opt['training']['skip_pt2']:
        ckpt_dict = {
            'opt': opt,
            'fwd_skin_net': fwd_skin_net.state_dict(),
            'inv_skin_net': inv_skin_net.state_dict(),
            'lat_vecs_inv_skin': lat_vecs_inv_skin.state_dict()
        }
        torch.save(ckpt_dict, '%s/ckpt_trained_skin_nets.pt' % ckpt_dir)

    if gen_cano_mesh:
        # get only valid triangles
        # train_data_loader.dataset.compute_valid_tri(inv_skin_net, model, lat_vecs_inv_skin, smpl_vitruvian)

        train_data_loader.dataset.is_train = False
        gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian,
                  train_data_loader,
                  cuda, '_pt3', reference_body_v=train_data_loader.dataset.Tpose_minimal_v, every_n_frame=1)
        train_data_loader.dataset.is_train = True



def trainWrapper_MVPHuman(subjects):
    parser = argparse.ArgumentParser(description='Train SCANimate.')
    parser.add_argument('--config', '-c', type=str, default='./lib/scanimate/config/example.yaml')
    parser.add_argument('--in_dir', '-i', type=str, default='/media/liaotingting/usb3/Dataset/data_sample')
    parser.add_argument('--out_dir', '-o', type=str, default='/media/liaotingting/usb3/Dataset/data_sample')
    parser.add_argument('--star', '-s', type=int, default=0)
    parser.add_argument('--inter', '-in', type=int, default=1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.star % 2 + 3)

    opt = load_config(args.config, 'lib/scanimate/config/default.yaml')

    # stage 0
    n = 0

    for i in range(args.star, len(subjects), args.inter):
        sid = subjects[i]
        opt['data']['data_dir'] = os.path.join(args.in_dir, sid)
        opt['experiment']['ckpt_dir'] = os.path.join(args.out_dir, sid, 'cano')
        opt['experiment']['result_dir'] = os.path.join(args.out_dir, sid, 'cano')
        opt['experiment']['log_dir'] = os.path.join(args.out_dir, sid, 'cano')

        if not os.path.exists(os.path.join(args.out_dir, sid, 'cano', 'da-pose.obj')):
            # try:
            #
            # except:
            #     continue
            train(opt, True)
        else:
            n += 1

    print(n)
    exit()


from lib.common.render import Render
import cv2
from lib.common.visual import image_grid

def render_one_item(gpu_id, sid):
    # obj_path = f'/media/liaotingting/usb/Params/{sid}/indoor/Action03/CANO/TPose.obj'
    # im_path = f'tmp/tpose/{sid}.png'
    # obj_path = f'./data_sample/{sid}/cano/tpose.obj'
    obj_path = '/media/liaotingting/usb3/Dataset/THuman2.0/scans/%04d/%04d.obj' % (sid, sid)
    im_path = f'/media/liaotingting/usb3/liaotingting/tmp/thuman/{sid}.png'
    os.makedirs(os.path.dirname(im_path), exist_ok=True)
    if os.path.exists(obj_path) and not os.path.exists(im_path):
        print(sid)
        render = Render(device=torch.device('cuda:%d' % (gpu_id % 2)))
        mesh = trimesh.load(obj_path)
        render.load_mesh(torch.from_numpy(mesh.vertices), torch.from_numpy(mesh.faces))
        image = render.get_image(cam_id=[2])
        cv2.imwrite(im_path, image * 255)
    elif not os.path.exists(obj_path):
        print('not exist')


def render_thuman(gpu_id, sid):
    from pytorch3d.io import load_objs_as_meshes
    obj_path = '/media/liaotingting/usb3/Dataset/THuman2.0/scans/%04d/%04d.obj' % (sid, sid)
    im_path = f'/media/liaotingting/usb3/liaotingting/tmp/thuman/{sid}.png'
    os.makedirs(os.path.dirname(im_path), exist_ok=True)
    if os.path.exists(obj_path) and not os.path.exists(im_path):
        print(sid)
        render = Render(device=torch.device('cuda:%d' % (gpu_id % 2)))

        mesh = load_objs_as_meshes([obj_path])

        render.load_mesh_with_text(mesh)
        image = render.get_image(cam_id=[2])
        cv2.imwrite(im_path, image * 255)
    if not os.path.exists(im_path):
        print(im_path)

def collect_images(subjects):
    images = []
    for sid in subjects:
        im = cv2.imread(f'tmp/tpose/{sid}.png')
        cv2.putText(im, sid, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, color=(0, 1, 0), thickness=4)
        images.append(im)
    from lib.common.visual import image_grid
    image_grid(images, rows=5, cols=10, save_path='tpose-test.png')


def collect_thuman():
    images = []
    for i in tqdm(range(525)):
        im = cv2.imread(f'/media/liaotingting/usb3/liaotingting/tmp/thuman/{i}.png')
        cv2.putText(im, str(i), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, color=(0, 1, 0), thickness=4)
        images.append(im)

        if i in [174, 349, 524]:
            image_grid(images, rows=12, cols=15, save_path=f'/media/liaotingting/usb3/liaotingting/tmp/{i}.png')
            images = []


def visual_results(subjects):
    import multiprocessing
    pool = multiprocessing.Pool(processes=2)
    try:
        r = [pool.apply_async(render_thuman, args=(i, sid)) for i, sid in enumerate(subjects)]
        pool.close()
        for item in r:
            item.wait(timeout=99999999)
    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.join()


def move_to_target_dir(subjects, target_dir):
    for sid in tqdm(subjects):
        # os.makedirs(os.path.join(target_dir, sid), exist_ok=True)
        # os.system(f'mv ./data_sample/{sid}/cano/da-pose.obj {target_dir}/{sid}')
        os.system(f'mv /media/liaotingting/usb3/Dataset/data_sample/{sid}/cano/da-pose.obj {target_dir}/{sid}')
        # os.system(f'mv ./data_sample/{sid}/cano/skin_weight.npz {target_dir}/{sid}')


def trainWrapper_THuman():
    parser = argparse.ArgumentParser(description='Train SCANimate.')
    parser.add_argument('--config', '-c', type=str, default='./lib/scanimate/config/example.yaml')
    parser.add_argument('--in_dir', '-i', type=str, default='/media/liaotingting/usb3/Dataset/THuman2.0/scans')
    parser.add_argument('--out_dir', '-o', type=str, default='/media/liaotingting/usb3/Dataset/THuman2.0/canon')
    parser.add_argument('--star', '-s', type=int, default=0)
    parser.add_argument('--inter', '-in', type=int, default=1)
    parser.add_argument('--n_gpu', '-ng', type=int, default=2)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.star % args.n_gpu)

    opt = load_config(args.config, 'lib/scanimate/config/default.yaml')

    # stage 0
    n = 0
    subjects = sorted(os.listdir(args.in_dir))
    for i in range(args.star, len(subjects), args.inter):
        sid = subjects[i]
        sid = '0020'
        opt['data']['data_dir'] = os.path.join(args.in_dir, sid)
        opt['experiment']['ckpt_dir'] = os.path.join(args.out_dir, sid)
        opt['experiment']['result_dir'] = os.path.join(args.out_dir, sid)
        opt['experiment']['log_dir'] = os.path.join(args.out_dir, sid)

        train(opt, True, 'thuman')
        exit()
        # use male for thuman
        opt['data']['smpl_gender'] = 'male'
        if not os.path.exists(os.path.join(args.out_dir, sid, 'tpose.obj')):
            train(opt, True, 'thuman')

        else:
            n += 1
        exit()

    print(n)


if __name__ == '__main__':
    train_sub = Path(
            '/media/liaotingting/usb2/projects/ARCH/data/mvphuman/train100.txt'
            # '/media/liaotingting/usb/eccv_challenge_body_train_subjects_200.txt'
             ).read_text().strip().split('\n')
    test_sub = sorted(Path('/media/liaotingting/usb/test.txt').read_text().strip().split('\n'))

    # move_to_target_dir(train_sub, '/media/liaotingting/usb3/Dataset/MVP-CANO200')
    # exit()
    # --- for thuman
    # visual_results(range(500))
    # render_thuman(0, 0)
    # exit()
    # collect_thuman()
    subjects = test_sub
    trainWrapper_MVPHuman(subjects)
    # trainWrapper_THuman()
    # visual_results(test_sub)
    # collect_images(test_sub)
    # move_to_target_dir(test_sub, target_dir='/media/liaotingting/usb2/projects/MVPHuman/MVPHumanTest')
