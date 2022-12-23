import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Global variables for the colors
BLACK = [0, 0, 0]
PINK = [239, 214, 189]
RED = [255, 0, 0]
BROWN = [40, 26, 16]


def fill_spot(x_start, y_start, surface, new_color, border=False):
    """
    Function to fill the mole
    -------------------------
    Arguments:
        x_start, y_start: coordinates of the point from which the filling starts;
        surface: the image to fill;
        new_color: the color with which the spot has to be filled;
        border: if True, mark as RED the the pixels of the border.
    """
    
    # The original color to fill, in the point (x_start, y_start)
    orig_value = surface[x_start, y_start].copy()
    # A stack to avoid recursion
    stack = [(x_start, y_start)]
    
    # Emulating the recursion
    while stack:
        x, y = stack.pop()
        # Every pixel which has the same color of the starting point has to be filled...
        if (surface[x, y] == orig_value).all():
            surface[x, y] = new_color
            # Insert in the stack the pixel to the top, bottom, left and right: 
            stack.append((x - 1, y))
            stack.append((x + 1, y))
            stack.append((x, y - 1))
            stack.append((x, y + 1))
        # ...until we reach a different color and it means we are at the border
        elif (surface[x, y] != new_color).all() and border:
            # and we set pixel color to red
            surface[x, y] = RED


def fill_ext(x_start, y_start, surface, new_color):
    """
    Function to fill the outside of the mole
    ----------------------------------------
    Arguments:
        x_start, y_start: coordinates of the point from which the filling starts;
        surface: the image to fill;
        new_color: the color with which the spot has to be filled;
    """
    # The limits of the picture
    x_max = surface.shape[0] - 1
    y_max = surface.shape[1] - 1
    # A stack to avoid recursion
    stack = [(x_start, y_start)]
    
    # Emulating the recursion
    while stack:
        x, y = stack.pop()
        # Every pixel which is neither black nor pink has to be filled
        if (surface[x, y] != BLACK).all() and (surface[x, y] != PINK).all():
            surface[x, y] = new_color
            # Insert in the stack the pixel to the top, bottom, left and right,
            # taking care not to exit from the picture:
            if x > 0:
                stack.append((x - 1, y))
            if x < x_max:
                stack.append((x + 1, y))
            if y > 0:
                stack.append((x, y - 1))
            if y < y_max:
                stack.append((x, y + 1))


if __name__ == '__main__':
    
    # File to be analyzed
    mole_name = 'low_risk_4'
    im = mpimg.imread('./moles/' + mole_name + '.jpg')
    
    # Show the original image
    plt.figure()
    plt.imshow(im)
    plt.title('Original image')
    plt.show()
    
    #%% Reshape the image from 3D to 2D
    N1, N2, N3 = im.shape # note: N3 is 3, the number of elementary colors, i.e. red, green, blue
    # im_or(i,j,1) stores the amount of red for the pixel in position i,j
    # im_or(i,j,2) stores the amount of green for the pixel in position i,j
    # im_or(i,j,3) stores the amount of blue for the pixel in position i,j
    im_2D = im.reshape((N1 * N2, N3)) # im_2D has N1*N2 rows and N3 columns
    # pixel in position i.j goes to position k=(i-1)*N2+j
    # im_2D(k,1) stores the amount of red of pixel k 
    # im_2D(k,2) stores the amount of green of pixel k 
    # im_2D(k,3) stores the amount of blue of pixel k 
    # im_2D is a sequence of colors, that can take 2^24 different values
    
    #%% Get a simplified image with only Ncluster colors
    # number of clusters/quantized colors we want to have in the simplified image
    Ncluster = 3
    # instantiate the object K-means:
    kmeans = KMeans(n_clusters=Ncluster, random_state=0)
    # run K-means:
    kmeans.fit(im_2D)
    # get the centroids (i.e. the 3 colors). Note that the centroids
    # take real values, we must convert these values to uint8
    # to properly see the quantized image
    centroids = kmeans.cluster_centers_.astype('uint8')
    # copy im_2D into im_2D_quant
    im_2D_quant = im_2D.copy()
    for kc in range(Ncluster):
        # find the indexes of the pixels that belong to cluster kc
        ind = (kmeans.labels_ == kc)
        # set the quantized color to these pixels
        im_2D_quant[ind, :] = centroids[kc, :]
        
    # Reshape the 'quantized' image from 2D to 3D
    im_quant = im_2D_quant.reshape((N1, N2, N3))
    # Show the 'quantized' image
    plt.figure()
    plt.imshow(im_quant, interpolation=None)
    plt.title('Image with quantized colors')
    plt.show()
    
    #%% Preliminary steps to find the contour after the clustering
    # Ask the user to write the coordinates of an internal point of the mole:
    x_spot = int(input('Insert the X of an internal point of the mole? '))
    y_spot = int(input('Insert the Y of an internal point of the mole? '))
    
    # Fill the mole with black, starting from the point the user wrote:
    fill_spot(y_spot, x_spot, im_quant, BLACK)
    
    # Find the pixels belonging to the mole (the black ones):
    spot = np.argwhere(im_quant == BLACK)
    # Find the coordinates of the limits of the sub-image containing the mole, 
    # with a 5 pixels frame:
    xmin = spot[:, 0].min() - 5
    xmax = spot[:, 0].max() + 5
    ymin = spot[:, 1].min() - 5
    ymax = spot[:, 1].max() + 5
    
    # Subset is the sub-image that includes the mole:
    subset = im_quant[xmin:xmax, ymin:ymax]
    
    # Fill with pink the outside of the mole (all the pixels that are not black),
    # starting from the top-left corner (0, 0):
    fill_ext(0, 0, subset, PINK)
    
    # Clean the internal light spots of the mole (due to reflections, etc. in the picture).
    # We want a picture with two colors: black for the mole, pink for the outside of the mole.
    # So, every pixel which is not pink belongs to internal unwanted spots, and is set to black.
    subset[subset[:, :, 0] != 239, :] = BLACK
    
    #%% Find the contour of the mole
    fill_spot(subset.shape[0] // 2, subset.shape[1] // 2, subset, BROWN, border=True)
    
    # Show the image with border
    plt.figure()
    plt.imshow(subset, interpolation=None)
    plt.title('Mole with border')
    plt.show()
    
    # The perimeter of the mole consists of red pixels:
    mole_perim = (subset[:,:,0] == 255).sum()
    # The area of the mole consists of brown pixels:
    mole_area = (subset[:,:,0] == 40).sum()
    # The radius of the circle having the same area of the mole:
    r = np.sqrt(mole_area / np.pi)
    # The perimeter of the circle having the same area of the mole:
    circ_perim = 2 * r * np.pi
    # The ratio between the perimeter of the mole and the circle:
    ratio = mole_perim / circ_perim
    print("Ratio: ", ratio)
    
    #%% Save the ratios 
    data = pd.read_csv('moles_ratios.txt', index_col=0)
    if mole_name.split('_')[0] == 'low':
        category = 0
    elif mole_name.split('_')[0] == 'medium':
        category = 1
    elif mole_name.split('_')[0] == 'melanoma':
        category = 2
    data.loc[mole_name] = [ratio, category]
    data.to_csv('moles_ratios.txt')
    
    # The ratios of the low risk moles:
    x1 = list(data.loc[data.loc[:,'category'] == 0]['ratio'])
    # The ratios of the medium risk moles:
    x2 = list(data.loc[data.loc[:,'category'] == 1]['ratio'])
    # The ratios of the melanomas:
    x3 = list(data.loc[data.loc[:,'category'] == 2]['ratio'])
    
    # Plot the histogram of the ratios grouped by category
    labels = ['low risk', 'medium risk', 'melanoma']
    colors = ['#ffbb99', '#cc4400', '#662200']
    plt.grid(zorder=0)
    plt.hist([x1, x2, x3], label=labels, color=colors, edgecolor='white',zorder=3) 
    plt.legend()
    plt.xlabel('Ratio')
    plt.ylabel('Frequency')
    plt.title('Histogram of moles grouped by category')
    plt.show()
    

