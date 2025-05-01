import numpy as np
from skimage import io
from skimage.feature import graycomatrix, graycoprops

def extract_GLCM_features(masked_image, dist, angl, norm):

    GLCM = graycomatrix(masked_image, dist, angl, normed = norm) #for 1 distance and 1 angle, size is 256 by 256
    #shape=(256, 256, 2, 1), dtype=uint32),where the 3rd dimension is the distance and the 4th dimension is the angle

    prop_1 = graycoprops(GLCM, 'contrast')
    prop_2 = graycoprops(GLCM, 'energy')
    prop_3 = graycoprops(GLCM, 'entropy')
    prop_4 = graycoprops(GLCM, 'correlation')
    prop_5 = graycoprops(GLCM, 'dissimilarity')
    prop_6 = graycoprops(GLCM, 'homogeneity')
    prop_7 = graycoprops(GLCM, 'ASM')
    prop_8 = graycoprops(GLCM, 'mean')
    prop_9 = graycoprops(GLCM, 'variance')
    prop_10 = graycoprops(GLCM, 'std')
            
    return_features = np.concatenate((prop_1.flatten(), prop_2.flatten(), prop_3.flatten(), 
                                        prop_4.flatten(), prop_5.flatten(), prop_6.flatten(),
                                        prop_7.flatten(), prop_8.flatten(), prop_9.flatten(),
                                        prop_10.flatten()))
    return_glcm = GLCM

    return return_features, return_glcm