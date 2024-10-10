from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue, GaussNoise, Blur,
    HorizontalFlip, VerticalFlip, Rotate
)

def apply_augmentation(image, annotations):
    aug = Compose([
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(p=0.5),
        GaussNoise(p=0.3),
        Blur(blur_limit=7, p=0.3),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.3),
        Rotate(limit=30, p=0.5)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    bboxes = [[anno[2], anno[3], anno[2]+anno[4], anno[3]+anno[5]] for anno in annotations]
    labels = [anno[1] for anno in annotations]

    try:
        augmented = aug(image=image, bboxes=bboxes, labels=labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_annotations = []
        for i, bbox in enumerate(augmented_bboxes):
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            augmented_annotations.append([annotations[i][0], labels[i], int(x), int(y), int(w), int(h)] + annotations[i][6:])
        return augmented_image, augmented_annotations
    except ValueError as e:
        print(f"Skipping augmentation due to error: {e}")
        return image, annotations