import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import string


def get_rectangle(bb, edcolor='r'):
    """
    Create a rectangle patch for a given bounding box.

    This function creates and returns a rectangle patch object using the specified bounding box coordinates. The rectangle
    is defined by the center point (bb[0], bb[1]), width bb[2], and height bb[3]. The edge color of the rectangle can be
    customized using the optional 'edcolor' parameter.

    Parameters:
    bb (tuple): The bounding box coordinates in the format (center_x, center_y, width, height).
    edcolor (str, optional): The edge color of the rectangle (default is 'r').

    Returns:
    patches.Rectangle: A rectangle patch object representing the bounding box.

    Rectangle(xy=(0, 0), width=100, height=100, angle=0)

    Note:
    - The function assumes the bounding box coordinates are in the form (center_x, center_y, width, height).
    - The default edge color is 'r' (red).
    - The face color of the rectangle is set to 'none', resulting in an empty rectangle.
    """
    return patches.Rectangle((bb[0] - bb[2] / 2, bb[1] - bb[3] / 2), bb[2], bb[3], linewidth=2, edgecolor=edcolor,
                             facecolor='none')


def get_corners_coord(bb):
    x0 = bb[0] - bb[-2] / 2
    y0 = bb[1] - bb[-1] / 2
    x1 = bb[0] + bb[-2] / 2
    y1 = bb[1] + bb[-1] / 2

    top_left = (x0, y0)
    top_right = (x1, y0)
    bottom_left = (x0, y1)
    bottom_right = (x1, y1)

    return [top_left, top_right, bottom_left, bottom_right]


def get_interested_angles(bb):
    """
    Get the list of interested angles for a given bounding box.

    This function calculates and returns a list of angles of interest for a given bounding box. The angles include
    0.0, 90.0, -90.0, and 180.0, representing horizontal, vertical, and diagonal orientations.

    Parameters:
    bb (tuple): The bounding box coordinates in the format (center_x, center_y, width, height).

    Returns:
    list: A list of angles representing the interested angles for the bounding box.

    Example:
    # >>> bounding_box = (50, 50, 100, 100)
    # >>> angles = get_interested_angles(bounding_box)
    # >>> print(angles)
    [0.0, 90.0, -90.0, 180.0, 45.0, -45.0, 135.0, -135.0]

    Note:
    - The function assumes the bounding box coordinates are in the form (center_x, center_y, width, height).
    - The additional diagonal angles are calculated based on the corners of the bounding box.
    """
    angles = [0.0, 90.0, -90.0, 180.0]
    corners = get_corners_coord(bb)

    boxes = [(coord[0], coord[1], 0.1 * bb[2], 0.1 * bb[3]) for coord in corners]
    diag_angles = [calculate_angle(box, bb) for box in boxes]

    # print('diag_angles : ', diag_angles)

    return angles + diag_angles


def calculate_angle(bb1, bb2):
    c2 = np.array((bb2[0], bb2[1]))
    c1 = np.array((bb1[0], bb1[1]))

    x = c2 - c1

    return np.angle(x[0] + 1j * x[1], deg=True)


def  generate_equally_spaced_points(bounding_box):
    """
    Generate equally spaced points within a bounding box in clockwise manner.

    This function calculates and generates equally spaced points within a given bounding box. The points are distributed
    based on different angles of interest within the box.

    Parameters:
    bounding_box (tuple): The bounding box coordinates in the format (center_x, center_y, width, height).

    Returns:
    list: A list of (x, y) coordinates representing the equally spaced points within the bounding box.
    """
    center_x, center_y, width, height = bounding_box
    points = []

    # Calculate spacing between points
    spacing_horizontal = width / 8
    spacing_vertical = height / 8

    diagonal_length = math.sqrt(width ** 2 + height ** 2)
    spacing_diagonal = diagonal_length / 8

    angles = get_interested_angles(bounding_box)
    angles.sort()

    pos_angles = [x for x in angles if x >= 0.0]
    neg_angles = [x for x in angles if x < 0.0]

    # ajustar angulos positivos para ordem crescente
    pos_angles.sort()
    # ajustar angulos negativos para ordem crescente
    neg_angles.sort()

    angles = pos_angles + neg_angles
    # para cada bb das direções, contando apartir do centro (1, 2 , 3 bb)
    for i in range(1, 4):

        # percorrendo os angulos de forma horária (pois está organizado : 0* -> 90 -> 180* e 180* -> -90 -> 0*)

        for angle in angles:

            x_signal = 1.0 if angle > 90.0 or angle < -90.0 else -1.0
            y_signal = 1.0 if angle > 0.0 else -1.0

            # horizontal
            if angle in [0.0, 180.0]:
                x = center_x + (i * spacing_horizontal * x_signal)
                y = center_y

            # diagonal
            if angle not in [0.0, 90.0, -90.0, 180.0]:
                x = center_x - i * spacing_diagonal * math.cos(math.radians(angle))
                y = center_y - i * spacing_diagonal * math.sin(math.radians(angle))

            # vertical
            if angle in [90.0, -90.0]:
                x = center_x
                y = center_y - (i * spacing_vertical * y_signal)

            points.append((x, y))
            

    return points


def plot_comparisons_diagonals(bb, boxes):
    """
    Plot comparisons of bounding boxes in relation to a main bounding box.

    This function generates a matplotlib figure with four subplots to visualize the comparisons of bounding boxes with
    respect to a main bounding box. The center_box and the main bounding box are plotted on each subplot for comparison.

    Parameters:
    bb (tuple): The bounding box coordinates of the main box in the format (x, y, width, height).
    boxes (list): A list of bounding box coordinates in the format [(x1, y1, width1, height1), (x2, y2, width2, height2), ...].

    Returns:
    None

    Example:
    bb = (10, 20, 50, 30)
    boxes = [(15, 25, 40, 20), (20, 30, 35, 25), (25, 35, 30, 30)]
    plot_comparisons_diagonals(bb, boxes)
    """
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # inSideLeftAltAbove, inSideLeftAltBelow, inSideRightAltAbove, inSideRightAltBelow
    name = list(string.ascii_uppercase)[:len(boxes)]

    # Plot center box and bb for comparision

    center_box = boxes[0]

    for ax in [ax1, ax2, ax3, ax4]:
        ax.add_patch(get_rectangle(bb, 'r'))
        ax.add_patch(get_rectangle(center_box, 'g'))
        ax.text(center_box[0], center_box[1], f'box : {name[0]}', ha='center', va='center', fontsize=12, color='blue')
        ax.invert_yaxis()
        ax.autoscale()

    # Add rectangles using patches.Rectangle()

    rect_inSideLeftAbove = get_rectangle(boxes[2], 'b')
    ax1.add_patch(rect_inSideLeftAbove)
    ax1.set_title(f'inSideLeftAltAbove : {inSideLeftAltAbove(boxes[2], center_box)}')
    ax1.text(boxes[2][0], boxes[2][1], f'box : {name[2]}', ha='center', va='center', fontsize=12, color='blue')

    rect_inSideRightAltAbove = get_rectangle(boxes[4], 'b')
    ax2.add_patch(rect_inSideRightAltAbove)
    ax2.set_title(f'inSideRightAltAbove : {inSideRightAltAbove(boxes[4], center_box)}')
    ax2.text(boxes[4][0], boxes[4][1], f'box : {name[4]}', ha='center', va='center', fontsize=12, color='blue')

    rect_inSideRightAltBelow = get_rectangle(boxes[6], 'b')
    ax3.add_patch(rect_inSideRightAltBelow)
    ax3.set_title(f'inSideRightAltBelow : {inSideRightAltBelow(boxes[6], center_box)}')
    ax3.text(boxes[6][0], boxes[6][1], f'box : {name[6]}', ha='center', va='center', fontsize=12, color='blue')

    rect_inSideLeftAltBelow = get_rectangle(boxes[8], 'b')
    ax4.add_patch(rect_inSideLeftAltBelow)
    ax4.set_title(f'inSideLeftAltBelow : {inSideLeftAltBelow(boxes[8], center_box)}')
    ax4.text(boxes[8][0], boxes[8][1], f'box : {name[8]}', ha='center', va='center', fontsize=15, color='blue')

    fig.tight_layout()
    plt.show()


def generate_eq_spaced_bbs_in_(bb, dictionary_with_letters=True):
    """
    Function to generate bounding boxes inside of a bounding box 'bb' in clockwise direction.

    Parameters:
    bb (list): list of bounding boxes in YOLO Format ([x_center, y_center, width, height]).
    dictionary_with_letters (bool): if boxes will be referred as Alphabetic letter ( A to Z ).

    Returns:
    bbs (list) : list of resulting bounding boxes

    The first element of the 'boxes' list is considered as the center_box. The subsequent bounding boxes are arranged
    in a clockwise manner starting from 0 degrees and progressing to 90 degrees, 180 degrees, -90 degrees, and the
    diagonal angle before 0 degrees.

    """

    # print(f'x : {bb[0]}, y : {bb[1]}')

    angles = get_interested_angles(bb)
    angles.sort()

    ## get boxes 
    center_coord = generate_equally_spaced_points(bb)
    # print('center coordinates : ', center_coord)

    boxes = [[coord[0], coord[1], 0.1 * bb[2], 0.1 * bb[3]] for coord in center_coord]

    center_box = [bb[0], bb[1], 0.1 * bb[2], 0.1 * bb[3]]

    bbs = [center_box] + boxes


    if dictionary_with_letters:
        letters = list(string.ascii_uppercase)[:len(bbs) + 1]
        result_bbs = {letter: box for letter, box in zip(letters, bbs)}
        greater_box = [bb[0], bb[1], 2 * (result_bbs['N'][0] - bb[0]), 2 * (result_bbs['P'][1] - bb[1])]
        
        result_bbs[letters[-1]] = greater_box
        
        return result_bbs
    else:
        return bbs


def plot_all_bb_with_names(bb, boxes, name=list(string.ascii_uppercase)):
    """
    Plot bounding boxes with corresponding names on a matplotlib figure.

    Parameters:
    bb (tuple): The bounding box coordinates of the main box (x, y, width, height).
    boxes (list): A list of bounding box coordinates (x, y, width, height).
    name (list, optional): A list of names corresponding to each box. Default is a list of uppercase letters.

    Returns:
    None

    Example:
    bb = (10, 20, 50, 30)
    boxes = [(15, 25, 40, 20), (20, 30, 35, 25), (25, 35, 30, 30)]
    plot_all_bb_with_names(bb, boxes)
    """
    name = list(string.ascii_uppercase)[:len(boxes)]
    print(f'len_boxes : {len(boxes)}, len_names : {len(name)},name : {name}')

    fig, ax = plt.subplots()
    ax.add_patch(get_rectangle(bb))
    ax.set_title('Boxes Generated')

    if isinstance(boxes, dict):
        boxes_keys = list(boxes.keys())
        
        for i in range(0, len(boxes)):
            
            key_at_index = boxes_keys[i]
            box = boxes[key_at_index]
            
            
            if i == 0:
                ed = 'orange'
            elif (i % 2 == 0 and not (i == '0') and not (i == len(boxes) - 1)):
                ed = 'red'
            elif i == len( boxes ) - 1 :
                ed = 'green'
            else:
                ed = 'magenta'

            ax.add_patch(get_rectangle(box, edcolor=ed))
            ax.text(box[0] + 0.3 * box[2], box[1] - 0.55 * box[3], f'box : {name[i]}', ha='center', va='center', fontsize=12,
                    color='blue')
    else:
      for i in range(0, len(boxes)):
        if i == 0:
            ed = 'orange'
        elif (i % 2 == 0 and not (i == '0') and not (i == len(boxes) - 1)):
            ed = 'red'
        elif i == len( boxes ) - 1 :
            ed = 'green'
        else:
            ed = 'magenta'

        ax.add_patch(get_rectangle(boxes[i], edcolor=ed))
        ax.text(boxes[i][0], boxes[i][1] - 0.10, f'box : {name[i]}', ha='center', va='center', fontsize=12,
                color='blue')  
        
    

    ax.legend(handles=[patches.Patch(color='red', label='diagonals'),
                       patches.Patch(color='magenta', label='horizontal and vertical'),
                       patches.Patch(color='orange', label='center_box'),
                       patches.Patch(color='green', label='greater_box')])
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    bb1 = [0.5, 0.5, 0.2, 0.6]  # Bounding box 

    boxes = generate_eq_spaced_bbs_in_(bb1)
    print('box : ', boxes['Z'])
    plot_all_bb_with_names(bb1, boxes)
    # plot_comparisons_diagonals(bb1, boxes)

    # print(boxes['A'])

#%%
