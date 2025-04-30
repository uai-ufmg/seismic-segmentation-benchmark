import os
import json
import torch
import numpy as np
from datetime import datetime
from PIL import Image


def tensor_to_image(output):
    # Defining color palettes
    color_map = np.array([
        (68, 1, 84),
        (72, 40, 120),
        (62, 73, 137),
        (49, 104, 142),
        (38, 130, 142),
        (31, 158, 137),
        (53, 183, 121),
        (110, 206, 88),
        (181, 222, 43),
        (253, 231, 37)
    ], dtype=np.uint8)
    
    _, height, width = output.shape

    # Finding the class with the highest probability for each pixel
    output = np.argmax(output, axis=0)

    # Mapping the class indices to colors and mirroring
    image = color_map[output]
    image = np.flip(image.reshape((height, width, 3)), axis=0)

    image = Image.fromarray(image)
    image = image.rotate(270, expand=True)

    return image


def save_images(preds, preds_path):    
    for idx, pred in preds.items():
        # pred = tensor_to_image(pred.cpu())
        pred = np.argmax(pred.cpu(), axis=0)
        np.save(os.path.join(preds_path, f'pred_{idx}.npy'), pred)


def store_results(args, results):
    # Creating the results folder if it does not exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    results_folder = os.path.join(args.results_path, datetime.now().isoformat('#'))
    os.makedirs(results_folder)

    images_folder = os.path.join(results_folder, 'preds')
    os.makedirs(images_folder)

    # Storing metadata
    with open(os.path.join(results_folder, f'metadata.json'), 'w') as json_buffer:
        json.dump(vars(args), json_buffer, indent=4)
    
    for fold_number in range(len(results)):
        suffix = f'_fold_{fold_number + 1}' if args.cross_validation else ''

        model = results[fold_number]['model']
        scores = {
            key: value for key, value in results[fold_number].items() if key not in ['model', 'preds']
        }

        # Storing model weights
        if args.train:
            torch.save(model.state_dict(), os.path.join(results_folder, 'model' + suffix + '.pt'))

        # Storing metric scores
        with open(os.path.join(results_folder, 'scores' + suffix + '.json'), 'w') as json_buffer:
            json.dump(scores, json_buffer, indent=4)

        # Storing model outputs as images
        save_images(results[fold_number]['preds'], images_folder)
    
    print(f'\nResults saved in {results_folder}')