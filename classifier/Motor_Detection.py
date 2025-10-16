from .motor_config import labels, motors_sounds, other_sounds
from .Classification.PANN_files.inference import ast


def is_motor(wav_file, threshold=0.01):

    """Analyze and print the dominant sound category (motors or others) for each second.

    Args:
      motors_sounds: List of motor sound event labels
      other_sounds: List of other sound event labels
      threshold: Threshold for including an event in the aggregation
    """
    # Mapping labels to their indices
    label_to_index = {label: index for index, label in enumerate(labels)}
    clipwise_output = ast.inference(wav_file)
    # Aggregate probabilities for motor and other sounds
    motor_prob = sum(clipwise_output[label_to_index[label]] for label in motors_sounds if
                     clipwise_output[label_to_index[label]] > threshold)
    other_prob = sum(clipwise_output[label_to_index[label]] for label in other_sounds if
                     clipwise_output[label_to_index[label]] > threshold)

    # Determine dominant category
    dominant_category = "Motors" if motor_prob > other_prob else "Others"
    result = (motor_prob > other_prob)
    dominant_prob = max(motor_prob, other_prob)
    motore_score = motor_prob
    other_score = other_prob
    total_score = motore_score + other_score
    if total_score == 0:
        print("nothing :(")
        return False, None
    else:
        score_ratio = dominant_prob / total_score

    return result, score_ratio
