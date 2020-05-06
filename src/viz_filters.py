"""viz_filters.py 

Generates basis filters of the form in polar coordinates:

R_alpha(r)e^{i*alpha*phi}

Here, R_alpha is a Gaussian centred at beta.

Usage:
  viz_filters.py [--ksize=<n>]
  viz_filters.py (-h | --help)
  viz_filters.py --version

Options:
  -h --help      Show this string.
  --version      Show version.
  --ksize=<n>    Kernel size to display. [default: 7]
"""


from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


def get_filter_info(k_size):
    """
    Get the filter parameters for a given kernel size

    Args:
        k_size (int): input kernel size
    
    Returns:
        alpha_list: list of alpha values
        beta_list:  list of beta values
        bl_list:    used to bandlimit high frequency filters in get_basis_filters()
    """
    if k_size == 5:
        alpha_list = [0, 1, 2]
        beta_list = [0, 1, 2]
        bl_list = [0, 2, 2]
    elif k_size == 7:
        alpha_list = [0, 1, 2, 3]
        beta_list = [0, 1, 2, 3]
        bl_list = [0, 2, 3, 2]
    elif k_size == 9:
        alpha_list = [0, 1, 2, 3, 4]
        beta_list = [0, 1, 2, 3, 4]
        bl_list = [0, 3, 4, 4, 3]
    elif k_size == 11:
        alpha_list = [0, 1, 2, 3, 4]
        beta_list = [1, 2, 3, 4]
        bl_list = [0, 3, 4, 4, 3]
    
    return alpha_list, beta_list, bl_list


def get_basis_filters(alpha_list, beta_list, bl_list, k_size, eps=10**-8):
    """
    Gets the atomic basis filters

    Args:
        alpha_list:
        beta_list:
        bl_list:
        k_size (int): kernel size of basis filters
        eps=10**-8: epsilon used to prevent division by 0
    
    Returns:
        filter_list_bl: list of filters, with bandlimiting (bl) to reduce aliasing
        alpha_list_bl:  corresponding list of alpha used in bandlimited filters
        beta_list_bl:   corresponding list of beta used in bandlimited filters
    """

    filter_list_bl = []
    alpha_list_bl = []
    beta_list_bl = []
    for alpha in alpha_list:
        for beta in beta_list:
            if np.abs(alpha) <= bl_list[beta]:
                his = k_size//2  # half image size
                y_index, x_index = np.mgrid[-his:(his+1), -his:(his+1)]
                y_index *= -1
                z_index = x_index + 1j*y_index

                # convert z to natural coordinates and add eps to avoid division by zero
                z = (z_index + eps)
                r = np.abs(z)

                if beta == beta_list[-1]:
                    sigma = 0.6
                else:
                    sigma = 0.6
                rad_prof = np.exp(-(r-beta)**2/(2*(sigma**2)))
                c_image = rad_prof * (z/r)**alpha

                # add filter to list
                filter_list_bl.append(c_image)
                # add frequency of filter to list (needed for phase manipulation)
                alpha_list_bl.append(alpha)
                beta_list_bl.append(beta)

    return filter_list_bl, alpha_list_bl, beta_list_bl


def plot_filters(filter_list, alpha_list, beta_list):
    """
    Plot the real and imaginary parts of the basis filters.

    Args:
        filter_list: list of basis filters
        alpha_list:  alpha of each basis filter
        beta_list:   beta of each basis filter
    """

    count = 1
    plt.figure(figsize=(25, 11))
    nr_filts = 2*len(filter_list)
    len_x = math.ceil(np.sqrt(nr_filts))
    len_y = math.ceil(np.sqrt(nr_filts))
    len_x = 5
    len_y = 10

    for i in range(len(filter_list)):
        filt_real = filter_list[i].real
        filt_imag = filter_list[i].imag
        plt.subplot(len_x, len_y, count)
        plt.imshow(filt_real, vmin=-1, vmax=1, cmap=cm.gist_gray)
        plt.axis('off')
        plt.title('Real: $alpha$= %s, $beta$= %s' %
                    (alpha_list[i], beta_list[i]), fontsize=8)
        plt.subplot(len_x, len_y, count+1)
        plt.imshow(filt_imag, vmin=-1, vmax=1, cmap=cm.gist_gray)
        plt.axis('off')
        plt.title('Imaginary: $alpha$= %s, $beta$= %s' %
                    (alpha_list[i], beta_list[i]), fontsize=8)
        count += 2
    plt.tight_layout()
    plt.show()


#####
if __name__ == '__main__':
    args = docopt(__doc__)
    
    ksize = int(args['--ksize'])

    if ksize not in [5,7,9,11]:
        raise Exception(
            'Select ksize to be either 5,7,9 or 11')

    info = get_filter_info(ksize)

    filter_list, alpha_list, beta_list = get_basis_filters(
        info[0], info[1], info[2], ksize)
        
    plot_filters(filter_list, alpha_list, beta_list)
