## Find adjacent red and green pixels. If none, find 20 nearest pairs of red and green pixels.

import json
from PIL import Image
import numpy as np
from scipy.spatial import KDTree
import os
import cv2

import tqdm

def find_nearest_pairs(img, num_pairs=10, red_coords=None, green_coords=None):
    # # Load the image
    # img = Image.open(image_path)
    # pixels = np.array(img)

    # # Prepare lists to store coordinates
    # red_coords = []
    # green_coords = []

    # # Identify coordinates of blue and green pixels
    # for y in range(pixels.shape[0]):
    #     for x in range(pixels.shape[1]):
    #         if (pixels[y, x] == [255, 0, 0]).all():
    #             red_coords.append((x, y))
    #         elif (pixels[y, x] == [0, 255, 0]).all():
    #             green_coords.append((x, y))

    # Calculate distances and store them with pairs
    distances = []
    for bx, by in red_coords:
        for gx, gy in green_coords:
            distance = np.sqrt((bx - gx) ** 2 + (by - gy) ** 2)
            distances.append((distance, ((bx, by), (gx, gy))))

    # Sort distances and select the nearest pairs
    distances.sort()
    nearest_pairs = distances[:num_pairs]

    return nearest_pairs


def find_nearest_pairs_kdtree(img, num_pairs=20, red_coords=None, green_coords=None):
    # # Load the image
    # img = Image.open(image_path)
    # pixels = np.array(img)

    # Create KD-Trees
    tree_green = KDTree(green_coords)

    # Find nearest green pixels for each red pixel
    distances = []
    for red_coord in red_coords:
        distance, index = tree_green.query(red_coord)
        green_coord = green_coords[index]
        distances.append((distance, (red_coord, green_coord)))

    # Sort by distance and return the closest pairs
    distances.sort()
    nearest_pairs = distances[:num_pairs]

    return nearest_pairs

def remove_pixels_with_adjacent(red_coords, green_coords):
    # Prepare list to store adjacent pairs
    cleaned_red_coords = red_coords.copy()
    cleaned_green_coords = green_coords.copy()
    for bx, by in red_coords:
        for gx, gy in green_coords:
            if abs(bx - gx) <= 1 and abs(by - gy) <= 1:
                if (bx, by) in cleaned_red_coords:
                    cleaned_red_coords.remove((bx, by))
                if (gx, gy) in cleaned_green_coords:    
                    cleaned_green_coords.remove((gx, gy))

    return cleaned_red_coords, cleaned_green_coords

# Find adjacent red and green pixels, maximum of one neighbor per pixel, no repeats
def find_adjacent_pairs(img):
    # Load the image
    # img = Image.open(image_path)
    pixels = np.array(img)

    # Prepare lists to store coordinates
    red_coords = []
    green_coords = []

    # Identify coordinates of blue and green pixels
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            if (pixels[y, x] == [255, 0, 0]).all():
                red_coords.append((x, y))
            elif (pixels[y, x] == [0, 255, 0]).all():
                green_coords.append((x, y))
    
    # Prepare list to store adjacent pairs
    adjacent_pairs = []
    for bx, by in red_coords:
        for gx, gy in green_coords:
            if abs(bx - gx) <= 1 and abs(by - gy) <= 1:
                adjacent_pairs.append(((bx, by), (gx, gy)))
                red_coords.remove((bx, by))
                green_coords.remove((gx, gy))
                break

    return adjacent_pairs, red_coords, green_coords

# Find 20 nearest red and green pixels
def find_all_pairs(img, dist_threshold=1.0):
    ## Dict to store output info
    #  Fields:
    #  'all_pairs': list of tuples (distance, (red_pixel_coordinates, green_pixel_coordinates)
    #  'adjacent_pairs': list of tuples (distance, (red_pixel_coordinates, green_pixel_coordinates)
    #  'num_adjacent_pairs': int
    #  'nearest_pairs': list of tuples (distance, (red_pixel_coordinates, green_pixel_coordinates)) 
    out_dict = {}

    # Find adjacent pairs
    adjacent_pairs, red_coords, green_coords = find_adjacent_pairs(img)
    adjacent_pairs = [(1.0, (pair[0], pair[1])) for pair in adjacent_pairs]
    out_dict['adjacent_pairs'] = adjacent_pairs
    out_dict['num_adjacent_pairs'] = len(adjacent_pairs)
    # print('Adjacent pairs:', adjacent_pairs)
    # print('Number of adjacent pairs:', len(adjacent_pairs))

    # Remove adjacent pairs from red and green coordinates
    red_coords, green_coords = remove_pixels_with_adjacent(red_coords, green_coords)

    # # Find nearest pairs if there are less than 20 adjacent pairs
    # if len(adjacent_pairs) < 20:
    #     nearest_pairs = find_nearest_pairs(image_path, 20, red_coords, green_coords)
    #     out_dict['nearest_pairs'] = nearest_pairs
    #     print('Nearest pairs:', nearest_pairs)

    # Find nearest pairs using KD-Tree
    if len(adjacent_pairs) < 20 and len(red_coords) > 0 and len(green_coords) > 0:
        nearest_pairs = find_nearest_pairs_kdtree(img, 20 - out_dict['num_adjacent_pairs'], red_coords, green_coords)
        out_dict['nearest_pairs'] = nearest_pairs
        # print('Nearest pairs:', nearest_pairs)
        # print(len(nearest_pairs))
    else :
        nearest_pairs = []

    # Store adjacent and nearest pairs in the output dictionary
    out_dict['all_pairs'] = adjacent_pairs + nearest_pairs
    # print('Total pairs:', out_dict['all_pairs'])

    # Remove pairs with distance greater than the threshold
    # out_dict['all_pairs'] = [pair for pair in out_dict['all_pairs'] if pair[0] <= dist_threshold]

    return out_dict

# Convert pairs dictionary to a list of x,y coordinates
def pairs_dict_to_list(pairs_dict):
    pairs_list = []
    for pair in pairs_dict:
        pairs_list.append((pair[1][1], pair[1][0]))
    return pairs_list



def __main__():
    # Find all pairs for all images in the output directory, write to a json file
    mask_path = '../outputs/inference_results_mask/'
    output_path = '../outputs/near_pixel_pairs/'

    output = {}
    for masks in tqdm.tqdm(os.listdir(mask_path), desc='Processing images'):
        image_path = mask_path + masks
        input_image = Image.open(image_path)
        out_dict = find_all_pairs(input_image)
        # output[masks] = out_dict
        out_list = pairs_dict_to_list(out_dict['all_pairs'])
        output[masks] = out_list
        with open(output_path + masks[:-4], 'w+') as f:
            f.write(str(out_list))
    with open(output_path + 'near_pixel_pairs.json', 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    __main__()















# for tqdm, masks in enumerate(os.listdir(mask_path)):
#     image_path = mask_path + masks
#     out_dict = find_all_pairs(image_path)
#     output[masks] = out_dict   
    








