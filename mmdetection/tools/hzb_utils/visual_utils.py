import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

DOTA_CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter')

PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
           (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
           (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
           (255, 255, 0), (147, 116, 116), (0, 0, 255)]

def draw_line(img, bbox8, color, thickness_=2):
    points = [(int(bbox8[0]), int(bbox8[1])), (int(bbox8[2]), int(bbox8[3])),
              (int(bbox8[4]), int(bbox8[5])), (int(bbox8[6]), int(bbox8[7]))]
    cv2.line(img, points[0], points[1], color=color, thickness=thickness_)
    cv2.line(img, points[1], points[2], color=color, thickness=thickness_)
    cv2.line(img, points[2], points[3], color=color, thickness=thickness_)
    cv2.line(img, points[3], points[0], color=color, thickness=thickness_)
    return img

def visual_gt_DOTA(img_path, ann_path, output_path, num_rank=1000, show=False):
    if not osp.exists(output_path):
        os.mkdir(output_path)
    file_names = os.listdir(img_path)
    for i, file_name in enumerate(file_names):
        if i >= num_rank:
            break
        img = cv2.imread(osp.join(img_path, file_name))
        ann_file_name = file_name.split('.')[0] + '.txt'
        with open(osp.join(ann_path, ann_file_name), 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                continue
            for line in lines:
                line = line.split()
                bbox8 = [float(_) for _ in line[:8]]
                category = line[8]
                cv2.putText(img, category, (int(bbox8[0]), int(bbox8[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            PALETTE[DOTA_CLASSES.index(category)], 2)
                img = draw_line(img, bbox8, PALETTE[DOTA_CLASSES.index(category)])

        cv2.imwrite(osp.join(output_path, file_name), img)
        if show:
            plt.imshow(img)
            plt.show()

def visual_dist():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define function to generate points with a random shape
    def generate_random_shape(center_x, center_y, num_points, scale_range):
        # Generate points in a random shape
        x = np.random.normal(0, 1, num_points) * np.random.uniform(*scale_range, num_points)
        y = np.random.normal(0, 1, num_points) * np.random.uniform(*scale_range, num_points)
        x += center_x
        y += center_y
        return x, y

    # Generate random positions for surrounding clusters with no overlap
    def generate_cluster_centers(num_clusters, min_dist, max_dist, radius, existing_centers):
        centers = []
        angles = np.linspace(0, 2 * np.pi, num_clusters, endpoint=False)  # Equally spaced angles
        for angle in angles:
            # Randomize distance and add some offset to the angle
            r = np.random.uniform(min_dist, max_dist)
            angle_offset = np.random.uniform(-np.pi / 16, np.pi / 16)  # Small random offset
            center_x = r * np.cos(angle + angle_offset)
            center_y = r * np.sin(angle + angle_offset)
            # Ensure new cluster center doesn't overlap with existing ones
            if all(np.linalg.norm([center_x - xc, center_y - yc]) > 2 * radius for xc, yc in existing_centers):
                centers.append((center_x, center_y))
        return centers

    # Create figure
    plt.figure(figsize=(8, 8))

    # Colors for the 9 regions
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'cyan', 'pink', 'lime']

    # Generate central cluster (Region 0) with the most points and random shape
    x0, y0 = generate_random_shape(0, 0, 1000, (0.5, 3))
    plt.scatter(x0, y0, c=colors[0], label='0', alpha=0.6, s=10)

    # Generate 8 surrounding clusters with random positions and random shapes
    existing_centers = [(0, 0)]  # Start with the central cluster position
    radius = 6  # Maximum effective radius of each cluster
    cluster_centers = generate_cluster_centers(8, min_dist=15, max_dist=20, radius=radius,
                                               existing_centers=existing_centers)

    # Generate points for each cluster and apply color changes
    all_points = []
    for i, (center_x, center_y) in enumerate(cluster_centers):
        x, y = generate_random_shape(center_x, center_y, 800, (1, 4))  # Random shape for each cluster
        all_points.append((x, y, colors[i + 1]))

    # Apply 10% color change to every region
    change_prob = 0.1  # 10% probability
    for i, (x, y, original_color) in enumerate(all_points):
        # Determine which points change color
        change_mask = np.random.rand(len(x)) < change_prob

        # Randomly assign new colors to the selected points, excluding the original color
        other_colors = [c for c in colors if c != original_color]
        new_colors = np.random.choice(other_colors, size=len(x))
        final_colors = np.where(change_mask, new_colors, original_color)  # Apply color change based on probability

        # Re-plot the region with updated colors
        plt.scatter(x, y, c=final_colors, label=f'{i + 1} (10% color change)', alpha=0.6, s=5)

    # Customize plot
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.xlabel('Dim_1')
    plt.ylabel('Dim_2')
    plt.title('随机形状均匀分布的环形散点图（每区域颜色变换）')
    plt.legend(title="Regions", loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Save the plot as a PDF file
    output_file = "/data1/detection_data/thesis_visual/scatter_plot.svg"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")

    # Show plot
    plt.show()

    print(f"The scatter plot has been saved as '{output_file}'.")

if __name__=="__main__":
    # dota_path = '/data1/detection_data/DOTA_v1/DOTA1_1024_hbb/val/images/'
    # dota_ann_path = '/data1/detection_data/DOTA_v1/DOTA1_1024_hbb/val/annfiles/'
    # output_path = '/data1/detection_data/thesis_visual/DOTA_label'
    # visual_gt_DOTA(dota_path, dota_ann_path, output_path, show=False)
    visual_dist()