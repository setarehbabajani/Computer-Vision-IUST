
import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage import io
import functools

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')


    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(padded[i:i+Hk, j:j+Wk] * kernel)

    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian filter_values formula,
    and creates a filter_values matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp

    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate filter_values

    Returns:
        filter_values: numpy array of shape (size, size)
    """

    filter_values = np.zeros((size, size))
    delta = (size-1) / 2

    ### YOUR CODE HERE
    for i in range(size):
        for j in range(size):
            x = i - delta
            y = j - delta
            filter_values[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    ### END YOUR CODE

    return filter_values

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE
    Dx = np.array([[-1, 0, 1]]) * 0.5
    out = conv(img, Dx)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE
    Dy = np.array([[-1], [0], [1]]) * 0.5
    out = conv(img, Dy)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = partial_x(img)
    Gy = partial_y(img)
    G = np.sqrt(Gx**2 + Gy**2)

    # Convert radians to degrees and normalize to [0, 360)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 360
    ### END YOUR CODE

    return G, theta

def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    for i in range(1, H-1):
        for j in range(1, W-1):
            direction = theta[i, j]  # Direction of gradient
            if direction >= 0 and direction < 45 or direction >= 315 and direction <= 360:
              dx1, dy1 = 1, 0
              dx2, dy2 = -1, 0
            elif direction >= 45 and direction < 135:
              dx1, dy1 = 1, 1
              dx2, dy2 = -1, -1
            elif direction >= 135 and direction < 225:
              dx1, dy1 = 0, 1
              dx2, dy2 = 0, -1
            else:
              dx1, dy1 = -1, 1
              dx2, dy2 = 1, -1
            # Preserve the value if it's a local maximum
            if (G[i, j] >= G[i + dy1, j + dx1]) and (G[i, j] >= G[i + dy2, j + dx2]):
              out[i, j] = G[i, j]
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE
    strong_edges = img > high
    weak_edges = (img <= high) & (img > low)
    ### END YOUR CODE

    return strong_edges, weak_edges

def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    ### YOUR CODE HERE
    for i in range(max(0, y-1), min(y+2, H)):
        for j in range(max(0, x-1), min(x+2, W)):
            if (i, j) != (y, x):
                neighbors.append((i, j))
    ### END YOUR CODE

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))
    ### YOUR CODE HERE
    # Initialize queue with the positions of all strong edges
    queue = list(indices)

    while queue:
       # Pop the first element in the queue
        y, x = queue.pop(0)
        # If the edge at (y, x) hasn't been visited
        if not edges[y, x]:
            # Mark this edge as visited (linked)
            edges[y, x] = True
            # Iterate through all 8 neighbors
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    # Calculate neighbor's coordinates
                    ny, nx = y + dy, x + dx
                    # If neighbor is within bounds, is a weak edge, and hasn't been added yet
                    if 0 <= ny < H and 0 <= nx < W and weak_edges[ny, nx] and not edges[ny, nx]:
                        # Add it to the queue for further exploration
                        queue.append((ny, nx))
    ### END YOUR CODE

    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    ### YOUR CODE HERE
    # Step 1: Smooth the image by convolving it with a Gaussian kernel.
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed_img = conv(img, kernel)

    # Step 2: Find gradients of the smoothed image.
    G, theta = gradient(smoothed_img)

    # Step 3: Apply non-maximum suppression to thin out the edges.
    nms_img = non_maximum_suppression(G, theta)

    # Step 4: Use double thresholding to identify strong, weak, and non-edges.
    strong_edges, weak_edges = double_thresholding(nms_img, high, low)

    # Step 5: Link edges based on the edge tracking by hysteresis.
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge