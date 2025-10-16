import os

import numpy as np
import torch
import torch.nn.functional as f
from .Classification_Labels_Mean_file import DEFAULT_PATH_TO_LIST_LABELS_MEAN, load_labels_means, PATH_TO_SETS_FOLDER
from .Classification.PANN_files.inference import ast



def which_motor(audio):
    query_embedding = ast.encode(audio)

    # Get a list of subfolder names in the folder
    lables_folder = os.listdir(PATH_TO_SETS_FOLDER)
    lables_scores = []

    labels_means = load_labels_means()

    # Iterate over the subfolder names
    for subfolder_label in lables_folder:
        # dist = torch.cdist(torch.from_numpy(query_embedding), torch.from_numpy(labels_means[subfolder_label]))
        dist = np.linalg.norm(query_embedding - labels_means[subfolder_label]['mean'])
        score = -dist
        lables_scores.append(score)

    lables_scores_tensor = torch.tensor(lables_scores, dtype=torch.float32)

    # Apply softmax
    lables_scores_tensor = f.softmax(lables_scores_tensor, dim=0)

    # Convert the result back to a Python list if needed
    lables_scores = lables_scores_tensor.tolist()

    dict_scores = {}
    for i, label in enumerate(lables_folder):
        dict_scores[label] = round(lables_scores[i], 2)

    return lables_folder[np.argmax(lables_scores)], dict_scores
