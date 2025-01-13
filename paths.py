# -*- coding: utf-8 -*-
"""
Created on 13.01.20

"""
import os


class GeneralLoc:
    log_path = "/data/all-keypoints/logs"


class YTJumpLoc:
    base_path = '/data/jump_broadcast'  # '<path_to_download_of>/jump-broadcast'
    pretrained_model = "transformer/pretrained_weights/coco_arbitrary_limbs.pth.tar"

    frames_path = "/data/jump_broadcast/annotated_frames"  # os.path.join(base_path,
    # 'annotated_frames')
    annotation_dir = 'keypoints'
    annotation_path = os.path.join(base_path, annotation_dir)
    segmentation_path = os.path.join(base_path, "smplrx_results", 'segmentations')
    segmentation_images = segmentation_path
    segmentation_bbox_path = os.path.join(base_path, "smplrx_results", 'bboxes_segmentations_3d.csv')
    # segmentation_bbox_path = os.path.join(base_path, "bboxes_segmentations.csv")
    segmentation_endpoint_path = os.path.join(segmentation_path, 'endpoints_segmentations_{}{}.pkl')
    segmentation_anchor_path = os.path.join(segmentation_path, 'anchors_segmentations_{}.pkl')
    test_points = os.path.join(base_path, "test_points")
