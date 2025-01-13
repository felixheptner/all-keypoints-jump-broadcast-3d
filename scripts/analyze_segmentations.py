import numpy as np
import csv


def generate_histogram(data, num_bins):
    """Generates a histogram from a 1D NumPy array.

    Args:
    data: The 1D NumPy array to create the histogram from.
    num_bins: The number of bins in the histogram.

    Returns:
    A tuple containing the bin edges and the histogram values.
    """

    # Calculate bin edges based on the minimum and maximum values
    min_val = np.min(data)
    max_val = np.max(data)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Create the histogram
    hist, _ = np.histogram(data, bins=bin_edges)

    return bin_edges, hist

def read_person_indices_csv(filename: str) -> dict:
    result_dict = {}
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            key = row[0]
            value = int(row[1])
            result_dict[key] = value
    return result_dict


if __name__ == "__main__":
    import shutil
    import glob
    import json
    import os
    import smplx
    from datasets.general.csv_annotation_utils import read_csv_annotations
    from datasets.jump.jump_joint_order import JumpJointOrder
    from generate_segmentations import load_smplrx_result, SMPLX2JUMP_JOINTS, perspective_projection
    import tqdm
    import matplotlib.pyplot as plt

    import torch

    threshold = 500

    # Load jump broadcast keypoints
    yt_jump_annotation_header = ["event", "frame_num", "athlete", "slowmotion"] + [joint_name + suffix for
                                                                                   joint_name in
                                                                                   JumpJointOrder.names() for suffix
                                                                                   in
                                                                                   ["_x", "_y", "_s"]]
    offsets, keypoints = read_csv_annotations("/data/jump_broadcast/keypoints/train.csv", yt_jump_annotation_header,
                                              20)
    with open("/home/mmc-user/projektmodul/scripts/smplx_vert_segmentation.json") as f:
        segmentations = json.load(f)

    smplx_model = smplx.SMPLX(
        "/home/mmc-user/projektmodul/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz").to("cuda")

    img_names = [f"{offsets[i][0]}_({str(offsets[i][1]).zfill(5)})" for i in range(len(offsets))]
    jump_bodypart_indices = JumpJointOrder.bodypart_indices()
    base_result_path = "/data/jump_broadcast/smplrx_results"
    person_indices = read_person_indices_csv(os.path.join(base_result_path, "person_indices.csv"))
    images_skipped = 0

    distances = []
    mean_distances = []
    names = []

    for i, img_name in tqdm.tqdm(enumerate(img_names)):

        try:
            person_index = person_indices[img_name]
        except Exception as e:
            print(f"Key {img_name} not in person_indices... that shouln't happen")
            images_skipped += 1
            continue
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
        smplx_joints = smplx_mesh.joints[0].to("cpu").detach().numpy()[SMPLX2JUMP_JOINTS]
        smplx_joints = perspective_projection(smplx_joints, meta["focal"], meta["princpt"])[:, :2]
        jump_keypoints = keypoints[i][:, :2]
        distance = np.abs(np.linalg.norm(smplx_joints - jump_keypoints, axis=1))
        distances.append(distance)
        mean_distances.append(np.mean(distance))
        names.append(img_name)

    distances = np.array(distances)
    print(distances)

    num_bins = 30  # Desired number of bins
    os.makedirs(os.path.join(base_result_path, "statistics", "histograms"), exist_ok=True)
    os.makedirs(os.path.join(base_result_path, "statistics", "boxplots"), exist_ok=True)

    for i, name in enumerate(JumpJointOrder.names()):
        # Plot the histograms
        bin_edges, hist = generate_histogram(distances[:, i], num_bins)
        plt.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0])
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.title(f"{name} histogram")
        plt.savefig(os.path.join(base_result_path, "statistics", "histograms", f"{name}_hist.png"))
        plt.clf()

        # Plotting boxplots ... makes more sense than boxing plotboxes
        plt.boxplot(distances[:, i])
        plt.xticks(rotation=45)
        plt.xlabel(name)
        plt.ylabel("Distance")
        plt.title(f"{name} boxplot")
        plt.savefig(os.path.join(base_result_path, "statistics", "boxplots", f"{name}_bplot.png"))
        plt.clf()

    # Plot Boxplot
    plt.boxplot(distances, labels=JumpJointOrder.names())
    plt.xticks(rotation=45)
    plt.xlabel("Bodypart")
    plt.ylabel("Distance")
    plt.title(f"Box plot")
    plt.savefig(os.path.join(base_result_path, "statistics", "boxplots", f"bplot.png"))
    plt.clf()

    # copy worst fits to inspect
    l = zip(mean_distances, names)
    l = sorted(l)
    distances, names = zip(*l)
    worst_100 = names[-360: -1]
    best_100 = names[:100]
    vis_base_path = os.path.join(base_result_path, "vis")
    img_base_path = os.path.join(base_result_path, "img")
    worst_target_path = os.path.join(base_result_path, "worst")
    best_target_path = os.path.join(base_result_path, "best")
    segmentation_path = os.path.join(base_result_path, "segmentations")
    unused_path = os.path.join(segmentation_path, "unused")

    for name in worst_100:
        shutil.copyfile(os.path.join(vis_base_path, name + ".png"), os.path.join(worst_target_path, name + ".png"))
        shutil.copyfile(os.path.join(img_base_path, name + ".jpg"), os.path.join(worst_target_path, name + ".jpg"))
        shutil.move(os.path.join(segmentation_path, name), os.path.join(unused_path, name))
    for name in best_100:
        shutil.copyfile(os.path.join(vis_base_path, name + ".png"), os.path.join(best_target_path, name + ".png"))
        shutil.copyfile(os.path.join(img_base_path, name + ".jpg"), os.path.join(best_target_path, name + ".jpg"))

    print(f"Images skipped: {images_skipped}")