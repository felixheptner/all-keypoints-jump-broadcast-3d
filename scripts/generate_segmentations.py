import csv
import json
import os
import glob

import numpy as np
import torch
import imagesize
import tqdm
import cv2
import trimesh
import smplx

from matplotlib import pyplot as plt

from datasets.general.csv_annotation_utils import read_csv_annotations
from datasets.general.csv_annotation_utils import read_csv_bboxes
from datasets.jump.jump_joint_order import JumpJointOrder
from datasets.jump.jump_bodypart_order import JumpBodypartOrder

SMPLX2JUMP_JOINTS = np.array([15, 12, 17, 19, 21, 73, 16, 18, 20, 68, 2, 5, 8, 65, 63, 1, 4, 7, 62, 60])
SMPLX2JUMP_BODYPARTS = [("spine", "spine1", "spine2", "hips"),
                        ("head", "eyeballs", "neck", "leftEye", "rightEye"),
                        ("rightHand", "rightHandIndex1"),
                        ("leftHand", "leftHandIndex1"),
                        ("leftFoot", "leftToeBase"),
                        ("rightFoot", "rightToeBase"),
                        ("rightUpLeg",),
                        ("leftUpLeg",),
                        ("rightLeg",),
                        ("leftLeg",),
                        ("leftArm", "leftShoulder"),
                        ("rightArm", "rightShoulder"),
                        ("leftForeArm",),
                        ("rightForeArm",)
                        ]


def perspective_projection(vertices, focal, princpt):
    # vertices: [N, 3]
    # cam_param: [3]
    fx, fy= focal
    cx, cy = princpt
    vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
    vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
    return vertices


def vis_keypoints(img, kps, alpha=1, radius=3, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    if color is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        if color is None:
            cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def load_smplrx_result(base_path: str, img_name: str):
    meta_path = os.path.join(base_path, "meta", img_name + ".json")
    result_path = os.path.join(base_path, "smplx", img_name + ".npz")
    mesh_path = os.path.join(base_path, "mesh", img_name + ".obj")
    with open(meta_path) as f:
        meta = json.load(f)
    result = np.load(result_path)
    mesh = trimesh.load(mesh_path)
    return meta, result, mesh

def keypoint_dif_sum(keypoints, joints):
    diff = 0
    for i in range(keypoints.shape[0]):
        diff += np.sqrt((keypoints[i][0] - joints[i][0])**2 + (keypoints[i][1] - joints[i][1])**2)
    return diff

def find_bounding_box(arr: np.ndarray, margin: float):
    """Finds the bounding box of non-zero elements in a numpy array.
    Args:
        arr: The input numpy array.
        margin: Percentage of margin added to the left, right, top and bottom border.
    Returns:
        A tuple (x_min, y_min, width, height) representing the bounding box coordinates.
    """

    rows, cols = np.nonzero(arr)

    if len(rows) == 0:
        return None  # No non-zero elements

    x_min, x_max = np.min(cols), np.max(cols)
    y_min, y_max = np.min(rows), np.max(rows)

    width = x_max - x_min
    height = y_max - y_min

    x_min = max(0, x_min - margin * width)
    y_min = max(0, y_min - margin * height)

    x_max = min(arr.shape[1], x_max + margin * width)
    y_max = min(arr.shape[0], y_max + margin * height)

    return x_min, y_min, x_max - x_min, y_max - y_min


if __name__ == "__main__":
    threshold = 500

    # Load jump broadcast keypoints
    yt_jump_annotation_header = ["event", "frame_num", "athlete", "slowmotion"] + [joint_name + suffix for joint_name in
                                                                                   JumpJointOrder.names() for suffix in
                                                                                   ["_x", "_y", "_s"]]
    offsets, keypoints = read_csv_annotations("/data/jump_broadcast/keypoints/train.csv", yt_jump_annotation_header, 20)
    bboxes = read_csv_bboxes("/data/jump_broadcast/bboxes_segmentations.csv")
    with open("/home/mmc-user/projektmodul/scripts/smplx_vert_segmentation.json") as f:
        segmentations = json.load(f)

    smplx_model = smplx.SMPLX(
        "/home/mmc-user/projektmodul/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz").to("cuda")

    img_names = [f"{offsets[i][0]}_({str(offsets[i][1]).zfill(5)})" for i in range(len(offsets))]
    jump_bodypart_indices = JumpJointOrder.bodypart_indices()
    base_result_path = "/data/jump_broadcast/smplrx_results"

    print("Start generating segmentation masks!\n")

    stats = {"body_parts_checked": 0,
             "masks_generated": 0,
             "images_skipped": 0}

    new_bbox_file = open(os.path.join(base_result_path, "bbox_val.csv"), newline="", mode="w")
    new_person_indices_file = open(os.path.join(base_result_path, "person_indices.csv"), newline="", mode="w")
    bbox_writer = csv.writer(new_bbox_file)
    bbox_writer.writerow(["image_id", "min_x", "min_y", "width", "height"])
    person_indices_writer = csv.writer(new_person_indices_file)
    person_indices_writer.writerow(["image_id", "person_index"])


    for i, img_name in tqdm.tqdm(enumerate(img_names)):
        # load smplrx results
        person_indices = glob.glob(os.path.join(base_result_path, "meta", img_name + "_**"))
        person_indices = [index.split("_")[-1].split(".")[0] for index in person_indices]
        min_index = 0
        min_diff = 10000000
        for person_index in person_indices:
            meta, result, mesh = load_smplrx_result(base_result_path, img_name + "_" + str(person_index))
            result_tensors = {key: torch.tensor(data).to("cuda") for key, data in result.items()}
            smplx_mesh = smplx_model.forward(betas=result_tensors["betas"],
                                             global_orient=result_tensors["global_orient"],
                                             body_pose=result_tensors["body_pose"].unsqueeze(dim=0),
                                             transl=result_tensors["transl"],
                                             expression=result_tensors["expression"],
                                             jaw_pose=result_tensors["jaw_pose"],
                                             # left_hand_pose=result_tensors["left_hand_pose"].unsqueeze(dim=0),
                                             # right_hand_pose=result_tensors["right_hand_pose"].unsqueeze(dim=0),
                                             # leye_pose=result_tensors["leye_pose"], reye_pose=result_tensors["reye_pose"]
                                             )
            joints = smplx_mesh.joints[0].detach().to("cpu").numpy()[SMPLX2JUMP_JOINTS]
            joints_2d = perspective_projection(joints, meta["focal"], meta["princpt"])
            diff = keypoint_dif_sum(keypoints[i], joints_2d)
            if diff < min_diff:
                min_diff = diff
                min_index = person_index

        if len(person_indices) < 1:
            print(f"\nNo smplx result data was found for {img_name}.\nMaybe no person was detected on the input image.")
            stats["images_skipped"] += 1
            continue
        meta, result, mesh = load_smplrx_result(base_result_path, img_name + "_" + str(min_index))
        result_tensors = {key: torch.tensor(data).to("cuda") for key, data in result.items()}
        smplx_mesh = smplx_model.forward(betas=result_tensors["betas"],
                                         global_orient=result_tensors["global_orient"],
                                         body_pose=result_tensors["body_pose"].unsqueeze(dim=0),
                                         transl=result_tensors["transl"],
                                         expression=result_tensors["expression"],
                                         jaw_pose=result_tensors["jaw_pose"],
                                         # left_hand_pose=result_tensors["left_hand_pose"].unsqueeze(dim=0),
                                         # right_hand_pose=result_tensors["right_hand_pose"].unsqueeze(dim=0),
                                         # leye_pose=result_tensors["leye_pose"], reye_pose=result_tensors["reye_pose"]
                                         )

        # extract and project vertices and faces
        vertices = mesh.vertices
        vertices_2d = perspective_projection(vertices, meta['focal'], meta['princpt']).astype(np.int32)[:, :2]
        faces = mesh.faces

        joints = smplx_mesh.joints[0].detach().to("cpu").numpy()[SMPLX2JUMP_JOINTS]
        joints_2d = perspective_projection(joints, meta["focal"], meta["princpt"])

        dims = imagesize.get(meta["img_path"])
        img = plt.imread(meta["img_path"])
        img_plot = plt.imshow(img)
        plt.scatter(joints_2d[:, 0], joints_2d[:, 1], c="r", s=2)
        plt.scatter(keypoints[i, :, 0], keypoints[i, :, 1], s=2)
        for joint, label in enumerate(JumpJointOrder.names()):
            plt.annotate(label, (joints_2d[joint, 0], joints_2d[joint, 1]), size="xx-small")
        plt.savefig(os.path.join(base_result_path, "vis", img_name + ".png"))
        plt.clf()

        if img_name.split("_")[0] == "12":
            bbox_name = img_name + ".jpg"
        else:
            bbox_name = img_name
        bbox = bboxes[bbox_name]

        masks = []

        for j, smplx_segs in enumerate(SMPLX2JUMP_BODYPARTS):
            bodypart_name = JumpBodypartOrder.names()[j + 1]
            mask = np.zeros(shape=dims[::-1], dtype=np.uint8)
            joint_indices = np.array(JumpJointOrder.bodypart_indices()[j])
            jump_keypoints = keypoints[i, joint_indices, :2]
            smplx_joints = joints_2d[joint_indices, :2]

            joint_distance = keypoint_dif_sum(jump_keypoints, smplx_joints)

            stats["body_parts_checked"] += 1
            # generate all masks, worry about quality later
            """if joint_distance > threshold:
                continue"""
            for seg in smplx_segs:
                seg_verts = vertices_2d[segmentations[seg]]
                seg_faces = np.array([face for face in faces if face[0] in segmentations[seg] or
                                      face[1] in segmentations[seg] or face[2] in segmentations[seg]])
                for face in seg_faces:
                    cv2.fillPoly(mask, [vertices_2d[face]], (255))

            masks.append(mask)

        # find bounding box
        compound_mask = np.max(np.stack(masks), axis=0)
        new_bbox = find_bounding_box(arr=compound_mask, margin=0.05)
        cv2.imwrite("test.png", compound_mask[int(new_bbox[1]): int(new_bbox[1] + new_bbox[3]), int(new_bbox[0]): int(new_bbox[0] + new_bbox[2])])

        # crop and write masks for bodyparts
        for j, mask in enumerate(masks):
            bodypart_name = JumpBodypartOrder.names()[j + 1]
            mask = mask[int(np.floor(new_bbox[1] + 0.5)): int(np.floor(new_bbox[1] + new_bbox[3] + 0.5)),
                   int(np.floor(new_bbox[0] + 0.5)): int(np.floor(new_bbox[0] + new_bbox[2] + 0.5))]
            if new_bbox[2] < 1 or new_bbox[3] < 1:
                print(f"\nBounding box with non positive size for image {img_name}.")
                stats["images_skipped"] += 1
                break
            os.makedirs(os.path.join(base_result_path, "segmentations", f"{img_name}"), exist_ok=True)
            cv2.imwrite(os.path.join(base_result_path, "segmentations", f"{img_name}", bodypart_name + ".png"), mask)

            stats["masks_generated"] += 1

        bbox_writer.writerow([img_name, new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]])
        person_indices_writer.writerow([img_name, min_index])


    print("\nFinished generating segmentation masks!")
    print(f"Masks generated: {stats['masks_generated']}")
    print(f"Total body parts checked: {stats['body_parts_checked']}")
    print(f"Images skipped: {stats['images_skipped']}")

    new_bbox_file.close()
    new_person_indices_file.close()
