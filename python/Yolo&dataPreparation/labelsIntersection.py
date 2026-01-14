# from shapely.geometry import Polygon
# import os
# import matplotlib.pyplot as plt

# def read_coordinates(file_path):
#     with open(file_path, 'r') as file:
#         coordinates = list(map(float, file.read().split()))
#     return [(coordinates[i], coordinates[i+1]) for i in range(1, len(coordinates), 2)]

# def compute_intersection_over_union(poly1, poly2):
#     polygon1 = Polygon(poly1)
#     polygon2 = Polygon(poly2)
#     intersection = polygon1.intersection(polygon2)
#     intersection_area = intersection.area
#     union_area = polygon1.area + polygon2.area - intersection_area
#     iou = intersection_area / union_area
#     return iou

# def compute_average_precision(iou_thresholds, ground_truth_polygon, detected_polygon):
#     ap_sum = 0
#     for threshold in iou_thresholds:
#         if threshold == 1.0:
#             ap_sum += compute_precision(ground_truth_polygon, detected_polygon)
#         else:
#             ap_sum += compute_interpolated_precision(ground_truth_polygon, detected_polygon, threshold)
#     return ap_sum / len(iou_thresholds)

# def compute_precision(ground_truth_polygon, detected_polygon):
#     iou = compute_intersection_over_union(ground_truth_polygon, detected_polygon)
#     return 1 if iou >= 0.5 else 0

# def compute_interpolated_precision(ground_truth_polygon, detected_polygon, threshold):
#     iou = compute_intersection_over_union(ground_truth_polygon, detected_polygon)
#     return 1 if iou >= threshold else 0

# def main():
#     file1_folder = "C:/Users/blagn771/Desktop/FishDataset/Fish3/labelsByHand"  # Path to the first file containing polygon coordinates
#     file2_folder = "C:/Users/blagn771/Desktop/FishDataset/Fish3/labels"  # Path to the second file containing polygon coordinates

#     iou_sum = 0
#     count = 0
#     iou_thresholds = [0.5 + i * 0.05 for i in range(10)]
#     iou95_sum = 0

#     for file in os.listdir(file2_folder):
#         file1_path = os.path.join(file1_folder, file)
#         file2_path = os.path.join(file2_folder, file)
#         polygon1_coords = read_coordinates(file1_path)
#         polygon2_coords = read_coordinates(file2_path)

#         iou = compute_intersection_over_union(polygon1_coords, polygon2_coords)
#         mAP50_95 = compute_average_precision(iou_thresholds, polygon1_coords, polygon2_coords)

#         iou_sum += iou
#         iou95_sum += mAP50_95
        
#         count += 1

#     print("mean Intersection over Union (IoU):", iou_sum/count)
#     print("mAP50-95::", iou95_sum/count)

# if __name__ == "__main__":
#     main()

from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        coordinates = list(map(float, file.read().split()))
    return [(coordinates[i], coordinates[i+1]) for i in range(1, len(coordinates), 2)]

def compute_intersection(poly1, poly2):
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    intersection = polygon1.intersection(polygon2)
    return intersection.area

def compute_area(polygon_coords):
    polygon = Polygon(polygon_coords)
    return polygon.area

def plot_polygon(polygon_coords, label):
    x, y = zip(*polygon_coords)
    plt.plot(x, y, label=label)
    plt.fill(x, y, alpha=0.3)
    plt.axis([min(x)-0.01, max(x)+0.01, max(y)+0.01, min(y)-0.01])

def main():
    file1_path = "C:/Users/blagn771/Desktop/FishDataset/Fish3/labelsByHand/Fish3_frame-0114.txt"  # Path to the first file containing polygon coordinates
    file2_path = "C:/Users/blagn771/Desktop/FishDataset/Fish3/labels/Fish3_frame-0114.txt"  # Path to the second file containing polygon coordinates

    polygon1_coords = read_coordinates(file1_path)
    polygon2_coords = read_coordinates(file2_path)

    intersection_area = compute_intersection(polygon1_coords, polygon2_coords)
    polygon1_area = compute_area(polygon1_coords)
    percentage_variation = ((intersection_area - polygon1_area) / polygon1_area) * 100

    print("Intersection area:", intersection_area)
    print("Polygon 1 area:", polygon1_area)
    print("Percentage variation:", percentage_variation, "%")

    plot_polygon(polygon1_coords, label="Label manuel")
    plot_polygon(polygon2_coords, label="Label OpenCV")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Intersection des labels')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()