import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

def generate_offset_str(csv_full_path):
    angle_list = []
    offset_list = []
    
    with open(csv_full_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            angle_list.append(int(row['angle']))
            offset_list.append(float(row['offset']))
    offset_str = ""
    for angle, offset in zip(angle_list, offset_list):
        corrected_angle = '{0:.2f}'.format(angle-offset)
        offset_str += f"{angle}:{corrected_angle},"
    print(offset_str)
    
    
if __name__ == "__main__":
    f_name = "D:/acoustic/code/offsets.csv"
    generate_offset_str(f_name)
    