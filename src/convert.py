# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import os.path

import torch
import numpy as np
from os import path as osp
import tqdm

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel


# Not used here
# Joint order conversion will be handled in the downstream code

# h36m_regressor_reorder = [6, 5, 4,  # right leg
#                           1, 2, 3,  # left leg]
#                           0,  # root
#                           8, 7,  # neck, thorax
#                           9, 10,  # head
#                           16, 15, 14,  # right arm
#                           11, 12, 13,  # left arm
#                           ]


def batch_slice_indices(total, batch_size):
    starts = list(range(0, total, batch_size))
    ends = starts[1:]
    ends.append(total)
    indices = list(zip(starts, ends))
    return indices


def resample_to_lower_framerate(pose_sequence, src_frame_rate, target_frame_rate):
    assert target_frame_rate <= src_frame_rate
    T = 1 / src_frame_rate
    T_new = 1 / target_frame_rate
    duration = pose_sequence.shape[0] * T
    resampled_length = np.floor(duration / T_new)

    # Timestamps where to sample
    t = np.arange(resampled_length).astype(np.float64) * T_new
    # Fractional indices into old timestamps
    ts = t / T
    t_floor, t_ceil = np.floor(ts), np.ceil(ts)
    assert np.all(t_ceil < pose_sequence.shape[0])
    # Weighting for next pose, zero
    w2 = ts - t_floor
    # Weighting for previous pose
    w1 = 1. - w2
    # Generate resampled poses via weighted sum
    resampled_poses = (pose_sequence[t_floor.astype(np.int)] * w1[:, np.newaxis, np.newaxis]) + (
                pose_sequence[t_ceil.astype(np.int)] * w2[:, np.newaxis, np.newaxis])
    return resampled_poses


def convert_amass_to_3dkeypoint_sequence(path, body_models, joint_regressor_path, batch_size, use_dmpls=True,
                                         verbose=False):
    assert os.path.exists(path)
    required_fields = ['trans', 'gender', 'mocap_framerate', 'betas', 'poses']
    if use_dmpls:
        required_fields.append("dmpls")
    # Choose the device to run the body model on.
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bdata = np.load(path)
    complete = [k in bdata.keys() for k in required_fields]
    if not complete:
        print(f"{path} incomplete. Skipping ...")
        return None

    frame_rate = bdata["mocap_framerate"].item()

    subject_gender = bdata['gender']
    if "U" not in str(subject_gender.dtype):
        subject_gender = np.char.decode(bdata["gender"], encoding="utf-8")
    subject_gender = str(subject_gender)

    if verbose:
        print('Data keys available:%s' % list(bdata.keys()))
        print('The subject of the mocap sequence is  {}.'.format(subject_gender))
        print(f'The mocap sequence has {bdata["poses"].shape[0]} frames with framerate of {bdata["mocap_framerate"]}')

    with torch.no_grad():
        bm = body_models[subject_gender]
        regressor = np.load(joint_regressor_path)
        torch_regressor = torch.Tensor(regressor).to(comp_device)
        time_length = len(bdata['trans'])
        batch_slices = batch_slice_indices(total=time_length, batch_size=batch_size)
        # Output array
        keypoint_sequence = np.zeros((time_length, regressor.shape[0], 3), dtype=np.float32)

        # Process in fixed batches
        for start, end in batch_slices:
            # Frame-specific params
            body_parms = {
                'root_orient': torch.Tensor(bdata['poses'][start: end, :3]).to(comp_device),
                # controls the global root orientation
                'pose_body':   torch.Tensor(bdata['poses'][start: end, 3:66]).to(comp_device),  # controls the body
                'pose_hand':   torch.Tensor(bdata['poses'][start: end, 66:]).to(comp_device),
                # controls the finger articulation
                'betas':       torch.Tensor(
                    np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=(end - start), axis=0)).to(
                    comp_device),  # controls the body shape. Body shape is static
                'trans':       torch.Tensor(bdata['trans'][start:end]).to(comp_device),
                # controls the global body position
            }
            if use_dmpls:
                body_parms['dmpls'] = torch.Tensor(bdata['dmpls'][start: end, :num_dmpls]).to(
                    comp_device)  # controls soft tissue dynamics]

            if verbose:
                print('Body parameter vector shapes: \n{}'.format(
                    ' \n'.join(['{}: {}'.format(k, v.shape) for k, v in body_parms.items()])))

            # Create body model, shape it based on params in batch mode
            shaped_body_model = bm(return_dict=True, **{k: v for k, v in body_parms.items()})

            vertices = shaped_body_model["v"]
            keypoint_batch = torch.einsum('bik,ji->bjk', [vertices, torch_regressor])
            keypoint_batch = c2c(keypoint_batch)
            keypoint_sequence[start:end] = keypoint_batch
            del (shaped_body_model)

    return keypoint_sequence, frame_rate


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def expandpath(path):
    x = os.path.expanduser(path)
    x = os.path.realpath(x)
    x = os.path.abspath(x)
    return x


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert AMASS to H36m compatible sequences')
    parser.add_argument('--amass_dir', required=True,
                        metavar="/path/to/amass",
                        help='Directory of the AMASS datasets')
    parser.add_argument('--out_dir', required=True,
                        metavar="/path/to/output",
                        help='Output directory')
    parser.add_argument("--joint_regressor", required=True,
                        metavar="/path/to/joint_regressor.npy",
                        help='Joint regressor')
    parser.add_argument('--r', required=False,
                        metavar="framerate",
                        default=None,
                        help='Target frame rate for pose sequence resampling. If empty, retains the original frame rate for each pose sequence.',
                        type=float)
    parser.add_argument('--gpu_id', required=False,
                        default=None,
                        metavar="gpu_id",
                        help='GPU ID. If empty, use CPU',
                        type=str)
    parser.add_argument('--batch_size', required=False,
                        default=2048,
                        metavar="batchsize",
                        help='Batchsize for processing SMPL meshes in parallel.',
                        type=int)


    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu_id is None else str(args.gpu_id)
    # print(torch.cuda.device_count())

    USE_DMPLS = True
    SUPPORT_DIR = expandpath(os.path.join("..", "support_data"))
    # Choose the device to run the body model on.
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.amass_dir = expandpath(args.amass_dir)
    args.out_dir = expandpath(args.out_dir)
    mkdirs(args.out_dir)
    args.joint_regressor = expandpath(args.joint_regressor)

    num_betas = 16  # number of body parameters
    num_dmpls = 8 if USE_DMPLS is True else None  # number of DMPL parameters
    body_models = {}
    # Generate Body odels
    for gender in ["male", "female", "neutral"]:
        bm_fname = osp.join(SUPPORT_DIR, 'body_models/smplh/{}/model.npz'.format(gender))
        dmpl_fname = osp.join(SUPPORT_DIR, 'body_models/dmpls/{}/model.npz'.format(gender))
        if USE_DMPLS is False:
            dmpl_fname = None
        bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(
            comp_device)
        bm.eval()
        body_models[gender] = bm


    def list_sorted_dirs(path):
        return [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]


    def list_sorted_files(path):
        return [d for d in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, d))]


    datasets = list_sorted_dirs(args.amass_dir)

    for dataset in datasets:
        print(dataset)
        dataset_path = os.path.join(args.amass_dir, dataset)

        # Check for inner directory
        content = list_sorted_dirs(dataset_path)
        if len(content) == 1:
            subjects_dir = os.path.join(dataset_path, content[0])
        else:
            subjects_dir = dataset_path

        out_path = os.path.join(args.out_dir, f"{dataset}.npz")
        # if os.path.exists(out_path):
        #     continue
        output_dir = {}

        # Loop through subject dirs
        subjects = list_sorted_dirs(subjects_dir)
        with tqdm.tqdm(subjects) as t_subjects:
            for subject in t_subjects:
                t_subjects.set_description(subject, refresh=True)
                output_dir[subject] = {}
                subject_path = os.path.join(subjects_dir, subject)

                # Filter out shape files
                # Loop through sequences
                sequences = [s for s in list_sorted_files(subject_path) if s != "shape.npz"]
                for sequence in sequences:
                    sequence_path = os.path.join(subject_path, sequence)

                    # Convert sequence
                    converted, frame_rate = convert_amass_to_3dkeypoint_sequence(path=sequence_path,
                                                                                 body_models=body_models,
                                                                                 joint_regressor_path=args.joint_regressor,
                                                                                 batch_size=args.batch_size,
                                                                                 use_dmpls=True)
                    if args.r is not None:
                        converted = resample_to_lower_framerate(converted, src_frame_rate=frame_rate,
                                                                target_frame_rate=args.r)
                        frame_rate = args.r

                    if converted is not None:
                        output_dir[subject][os.path.splitext(sequence)[0]] = {"positions_3d": converted,
                                                                              "frame_rate":   frame_rate}

        print(f'Saving to {out_path} ...')
        np.savez_compressed(out_path, positions_3d=output_dir)
        print('... done.')
