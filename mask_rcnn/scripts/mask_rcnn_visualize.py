import numpy as np
import cv2

from mrcnn.visualize import random_colors, apply_mask

# Modified "display_instances" function from mrcnn library that returns masked image
def display_instances(image, boxes, masks, class_ids, class_names, scores=None):

    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    masked_image = image.copy()

    N = boxes.shape[0]

    # Generate random colors
    colors = random_colors(N)

    for i in range(N):
        if not np.any(boxes[i]):
            continue

        # Mask
        mask = masks[:, :, i]
        color = colors[i]
        masked_image = apply_mask(masked_image, mask, color)

        # Bounding Box
        y1, x1, y2, x2 = boxes[i]
        masked_image = cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, 2)

        #Label
        label = class_names[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = "{} {:.3f}".format(label, score) if score else label
        masked_image = cv2.putText(masked_image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    return masked_image