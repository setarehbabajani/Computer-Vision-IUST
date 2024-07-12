def non_maximum_suppression(bboxes, scores, iou_threshold):
    # Step 1: Sort the bounding boxes by their confidence scores in descending order
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    selected_indices = []

    while indices:
        # Step 2: Select the box with the highest score
        current_index = indices.pop(0)
        selected_indices.append(current_index)

        # Step 3: Suppress all overlapping boxes with IoU above the threshold
        filtered_indices = []
        for i in indices:
            if compute_iou(bboxes[current_index], bboxes[i]) < iou_threshold:
                filtered_indices.append(i)
        indices = filtered_indices

    return selected_indices

def compute_iou(box1, box2):
    # Compute intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Compute union
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    return inter_area / union_area
