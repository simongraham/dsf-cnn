import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

"""
This script can be used to display the atomic basis filters
"""
#-------------------------------------------------------------------------------------------

# Kernel size to visualise
k_size = 7

if k_size == 5:
    alpha_list = [0,1,2]
    beta_list = [0,1,2]
    bl_list = [0,2,2]
elif k_size == 7:
    alpha_list = [0,1,2,3]
    beta_list = [0,1,2,3]
    bl_list = [0,2,3,2]
elif k_size == 9:
    alpha_list = [0,1,2,3,4]
    beta_list = [0,1,2,3,4]
    bl_list = [0,3,4,4,3]
elif k_size == 11:
    alpha_list = [0,1,2,3,4]
    beta_list = [1,2,3,4]
    bl_list = [0,3,4,4,3]
elif k_size == 13:
    alpha_list = [0,1,2,3,4,5]
    beta_list = [1,2,3,4]
    bl_list = [0,3,4,4,3]

#-------------------------------------------------------------------------------------------
def get_basis_filters(alpha_list, beta_list, bl_list, k_size, eps = 10**-8):
    filter_list = []
    freq_list = []
    radius_list = []
    for alpha in alpha_list:
        for beta in beta_list:
            if np.abs(alpha) <= bl_list[beta]:
                his = k_size//2 # half image size
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
                filter_list.append(c_image)
                # add frequency of filter to list (needed for phase manipulation)
                freq_list.append(alpha)
                radius_list.append(beta)
    
    return filter_list, freq_list, radius_list

#-------------------------------------------------------------------------------------------

# Generate and plot basis filters 

filts, freqs, rads = get_basis_filters(alpha_list, beta_list, bl_list, k_size)

count = 1
plt.figure(figsize=(25,11))
nr_filts = 2*len(filts)
len_x = math.ceil(np.sqrt(nr_filts))
len_y = math.ceil(np.sqrt(nr_filts))
len_x = 5
len_y = 10

for i in range(len(filts)):
        filt_real = filts[i].real
        filt_imag = filts[i].imag
        plt.subplot(len_x, len_y, count)
        plt.imshow(filt_real, vmin=-1,vmax=1, cmap=cm.gist_gray)
        plt.axis('off')
        plt.title('Real: $alpha$= %s, $beta$= %s' %(freqs[i], rads[i]), fontsize=8)
        plt.subplot(len_x, len_y, count+1)
        plt.imshow(filt_imag, vmin=-1, vmax=1, cmap=cm.gist_gray)
        plt.axis('off')
        plt.title('Imaginary: $alpha$= %s, $beta$= %s' %(freqs[i], rads[i]), fontsize=8)
        count += 2
plt.tight_layout()
plt.show()




