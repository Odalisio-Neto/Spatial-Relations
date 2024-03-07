import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_coordinates( bb ):
    x0 = bb[ 0 ]
    y0 = bb[ 1 ]
    x1 = bb[ 0 ] + bb[ -2 ]
    y1 = bb[ 1 ] + bb[ -1 ]
    return [ x0, y0, x1, y1 ]


def calculate_angle( bb1, bb2 ):
    c2 = np.array( (bb2[ 0 ], bb2[ 1 ]) )
    c1 = np.array( (bb1[ 0 ], bb1[ 1 ]) )
    
    x = c2 - c1
    
    return np.angle( x[ 0 ] + 1j * x[ 1 ], deg = True )


def translate_bbs( bb1: tf.Tensor, bb2: tf.Tensor, is_class_first = False,
                   is_yolo_format = False ):
    """
    Translate bounding box coordinates based on specified conditions.

    Parameters:
        bb1 (tensor): The coordinates of the first bounding box.
        bb2 (tensor): The coordinates of the second bounding box.
        is_class_first (bool, optional): If True, assumes the class is the first element in each bounding box. Default is False.
        is_yolo_format (bool, optional): If True, assumes the bounding boxes are in YOLO format. Default is False.

    Returns:
        tuple: Translated coordinates of the first and second bounding boxes in the format (x, y, w, h).

    Example:
        bb1 = tf.constant([0.3, 0.3, 0.2, 0.2])
        bb2 = tf.constant([0.3, 0.3, 0.1, 0.1])
        translated_bb1, translated_bb2 = translate_bbs(bb1, bb2, is_class_first=True, is_yolo_format=True)
    """
    if is_class_first:
        indices = (1, 2, 3, 4) if is_yolo_format else (0, 1, 2, 3)
    else:
        indices = (0, 1, 2, 3) if is_yolo_format else (0, 1, 2, 3)
    
    x1, y1, w1, h1 = bb1[ ..., indices ]
    x2, y2, w2, h2 = bb2[ ..., indices ]
    
    if is_yolo_format:
        x1, y1 = x1 - w1 / 2.0, y1 - h1 / 2.0
        x2, y2 = x2 - w2 / 2.0, y2 - h2 / 2.0
    
    return (x1, y1, w1, h1), (x2, y2, w2, h2)


@tf.function
def O( bb1: tf.Tensor, bb2: tf.Tensor, is_yolo_format = True, is_class_first = False ):
    """
    Check if two bounding boxes overlap.

    Parameters:
        bb1 (tensor): The coordinates of the first bounding box.
        bb2 (tensor): The coordinates of the second bounding box.
        is_class_first (bool, optional): If True, assumes the class is the first element in each bounding box. Default is False.
        is_yolo_format (bool, optional): If True, assumes the bounding boxes are in YOLO format. Default is False.
        
    Returns:
        bool: True if the bounding boxes overlap, False otherwise.
    
    Example:
        bb1 = tf.constant( [ 0.3, 0.3, 0.2, 0.2 ] )
        bb2 = tf.constant( [ 0.3, 0.3, 0.1, 0.1 ] )
    
        print( 'Check for O (True): ', O( bbs = (bb1, bb2), is_yolo_format = True, is_class_first = False ) )
    """
    (x1, y1, w1, h1), (x2, y2, w2, h2) = translate_bbs( bb1, bb2 )
    
    condition1 = tf.logical_and( tf.greater_equal( x2, x1 ), tf.less_equal( x2 + w2, x1 + w1 ) )
    condition2 = tf.logical_and( tf.greater( y2, y1 ), tf.less( y2 + h2, y1 + h1 ) )
    
    return tf.logical_or( condition1, condition2 )


@tf.function
# def PO(bbs, is_yolo_format=True):
def PO( bb1: tf.Tensor, bb2: tf.Tensor, is_yolo_format = True, is_class_first = True ):
    """
    Check if two bounding boxes partially overlap.

    Parameters:
        bb1 (tensor): The coordinates of the first bounding box.
        bb2 (tensor): The coordinates of the second bounding box.
        is_class_first (bool, optional): If True, assumes the class is the first element in each bounding box. Default is False.
        is_yolo_format (bool, optional): If True, assumes the bounding boxes are in YOLO format. Default is False.
        
    Returns:
        bool: True if the bounding boxes partially overlap, False otherwise.

    Note: (x1,y1) and (x2,y2) ordered pair ar located in lowest-left corner of the bounding box
    """
    (x1, y1, w1, h1), (x2, y2, w2, h2) = translate_bbs( bb1, bb2, is_yolo_format, is_class_first )
    '''
    mesmo que:
                # PO 1
            ((x2 < x1) and ((x2 + w2) > x1) and
             (((y2 < y1) and ((y2 + h2) > y1)) or ((y2 + h2) > (y1 + h1))))

            # PO 2
            or (((x2 < x1) and ((x2 + w2) > x1) and ((x2 + w2) < (x1 + w1))) and
                ((((y2 + h2) > (y1 + h1)) and (y2 < (y1 + h1))) or ((y2 < y1) and ((y2 + h2) > y1))))

            # PO 3
            or (((x2 > x1) and ((x2 + h2) < (x1 + h1))) and
                ((((y2 + h2) > (y1 + h1)) and (y2 < (y1 + h1))) or ((y2 < y1) and ((y2 + h2) > y1))))

            # PO 4
            or (((x2 > x1) and (x2 < (x1 + w1)) and ((x2 + w2) > (x1 + w1))) and
                ((((y2 + h2) < (y1 + h1)) and ((y2 + h2) > y1)) or ((y2 > y1) and (y2 < (y1 + h1)))))
    Essas condições são para elementos iteraveis (list), mas aqui é adaptado para uso em tf.Tensor()
    '''
    
    # Condition 1
    condition1 = tf.logical_and(
            tf.logical_and( x2 < x1, x2 + w2 > x1 ),
            tf.logical_or(
                    tf.logical_and( y2 < y1, y2 + h2 > y1 ),
                    y2 + h2 > y1 + h1
            )
    )
    
    # Condition 2
    condition2 = tf.logical_and(
            tf.logical_and( x2 < x1, x2 + w2 > x1 ),
            tf.logical_and(
                    x2 + w2 < x1 + w1,
                    tf.logical_or(
                            tf.logical_and( y2 + h2 > y1 + h1, y2 < y1 + h1 ),
                            tf.logical_and( y2 < y1, y2 + h2 > y1 )
                    )
            )
    )
    
    # Condition 3
    condition3 = tf.logical_and(
            tf.logical_and( x2 > x1, x2 + h2 < x1 + h1 ),
            tf.logical_or(
                    tf.logical_and( y2 + h2 > y1 + h1, y2 < y1 + h1 ),
                    tf.logical_and( y2 < y1, y2 + h2 > y1 )
            )
    )
    
    # Condition 4
    condition4 = tf.logical_and(
            tf.logical_and( x2 > x1, tf.logical_and( x2 < x1 + w1, x2 + w2 > x1 + w1 ) ),
            tf.logical_or(
                    tf.logical_and( y2 + h2 < y1 + h1, y2 + h2 > y1 ),
                    tf.logical_and( y2 > y1, tf.logical_and( y2 < y1 + h1, y2 < y1 + h1 ) )
            )
    )
    
    # Final condition, combining all the above conditions with logical OR
    final_condition = tf.logical_or(
            tf.logical_or( condition1, condition2 ),
            tf.logical_or( condition3, condition4 )
    )
    return final_condition


@tf.function
def D( bb1: tf.Tensor, bb2: tf.Tensor, is_yolo_format = True, is_class_first = False ):
    """
    Check if two bounding boxes are disjoint.

    Parameters:
        bb1 (tuple): The coordinates of the first bounding box in the format (x1, y1, w1, h1).
        bb2 (tuple): The coordinates of the second bounding box in the format (x2, y2, w2, h2).
        is_class_first (bool, optional): If True, assumes the class is the first element in each bounding box. Default is False.
        is_yolo_format (bool, optional): If True, assumes the bounding boxes are in YOLO format. Default is False.
        
    Returns:
        bool: True if the bounding boxes are disjoint, False otherwise.
    """
    (x1, y1, w1, h1), (x2, y2, w2, h2) = translate_bbs( bb1, bb2, is_yolo_format, is_class_first )
    
    condition1 = tf.less( x2 + w2, x1 )
    condition2 = tf.greater( x2, x1 + w1 )
    condition3 = tf.greater( y2, y1 + h1 )
    condition4 = tf.less( y2 + h2, y1 )
    
    bool_tensor = tf.logical_or( tf.logical_or( condition1, condition2 ),
                                 tf.logical_or( condition3, condition4 ) )
    float_tensor = tf.cast( bool_tensor, dtype = tf.float32 )
    
    return float_tensor


import matplotlib.pyplot as plt


def plot_bounding_boxes( bb1: tf.Tensor, bb2: tf.Tensor, is_class_first = False,
                         is_yolo_format = False ):
    """
    Plot two bounding boxes on a 2D plane.

    Parameters:
        bb1 (tensor): The coordinates of the first bounding box.
        bb2 (tensor): The coordinates of the second bounding box.
        is_class_first (bool, optional): If True, assumes the class is the first element in each bounding box. Default is False.
        is_yolo_format (bool, optional): If True, assumes the bounding boxes are in YOLO format. Default is False.

    Example:
        bb1 = tf.constant([0.3, 0.3, 0.2, 0.2])
        bb2 = tf.constant([0.3, 0.3, 0.1, 0.1])
        plot_bounding_boxes(bb1, bb2, is_class_first=True, is_yolo_format=True)
    """
    (x1, y1, w1, h1), (x2, y2, w2, h2) = translate_bbs( bb1, bb2, is_class_first, is_yolo_format )
    
    plt.figure( )
    plt.plot( [ x1 - w1 / 2, x1 + w1 / 2, x1 + w1 / 2, x1 - w1 / 2, x1 - w1 / 2 ],
              [ y1 - h1 / 2, y1 - h1 / 2, y1 + h1 / 2, y1 + h1 / 2, y1 - h1 / 2 ],
              'r' )
    
    plt.plot( [ x2 - w2 / 2, x2 + w2 / 2, x2 + w2 / 2, x2 - w2 / 2, x2 - w2 / 2 ],
              [ y2 - h2 / 2, y2 - h2 / 2, y2 + h2 / 2, y2 + h2 / 2, y2 - h2 / 2 ],
              'b' )
    
    plt.xlim( 0, 1 )
    plt.ylim( 0, 1 )
    plt.gca( ).invert_yaxis( )
    plt.show( )


## TODO: UPDATE EXAMPLES FOR TEST CASES
if __name__ == '__main__':
    # bb1 tem contido bb2
    # O(bb1,bb2)
    ## Exemplo True
    bb1 = tf.constant( [ 0.3, 0.3, 0.2, 0.2 ] )
    bb2 = tf.constant( [ 0.3, 0.3, 0.1, 0.1 ] )
    
    print( 'Check for O (True): ',
           O( bbs = (bb1, bb2), is_yolo_format = True, is_class_first = False ) )
    plot_bounding_boxes( bb1, bb2 )
    
    ## Exemplo False
    bb1 = tf.constant( [ 0.7, 0.6, 0.2, 0.3 ] )
    bb2 = tf.constant( [ 0.8, 0.7, 0.2, 0.3 ] )
    
    print( 'Check for O (False): ',
           O( bbs = (bb1, bb2), is_yolo_format = True, is_class_first = False ) )
    plot_bounding_boxes( bb1, bb2 )
    #
    # PO(bb1,bb2)
    ## Exemplo True
    bb1 = tf.constant( [ 0.5, 0.5, 0.2, 0.3 ] )
    bb2 = tf.constant( [ 0.4, 0.4, 0.25, 0.35 ] )
    
    print( 'Check for PO (True): ', PO( bb1, bb2, is_yolo_format = True, is_class_first = False ) )
    plot_bounding_boxes( bb1, bb2 )
    
    ## Exemplo False
    bb1 = tf.constant( [ 0.1, 0.1, 0.2, 0.3 ] )
    bb2 = tf.constant( [ 0.8, 0.7, 0.2, 0.3 ] )
    
    print( 'Check for PO (False): ',
           PO( bb1, bb2, is_yolo_format = True, is_class_first = False ) )
    plot_bounding_boxes( bb1, bb2 )
    
    # D(bb1,bb2)
    ## Exemplo True
    bb1 = tf.constant( [ 0.5, 0.5, 0.2, 0.3 ] )
    bb2 = tf.constant( [ 0.8, 0.7, 0.2, 0.3 ] )
    
    print( 'Check for D (True): ', D( bb1, bb2, is_yolo_format = True, is_class_first = False ) )
    plot_bounding_boxes( bb1, bb2 )
    
    ## Exemplo False
    bb1 = tf.constant( [ 0.5, 0.4, 0.2, 0.3 ] )
    bb2 = tf.constant( [ 0.5, 0.6, 0.2, 0.2 ] )
    
    print( 'Check for D (False): ', D( bb1, bb2, is_yolo_format = True, is_class_first = False ) )
    plot_bounding_boxes( bb1, bb2 )

# %%
