
import cv2

# Template Sysnthesis definitions
#
# all numpy arrays

original_sample = cv2.imread("args.sample_path")   #-  original image, the source pattern to be compared and reproduced

# sample - original_sample in gray scale, converted to floating point and 
#          normalize to the range [0., 1.]
#          used to template matching search and copy fragments to window

# window  - the result window in gray scale of template synthesis algorithm. 
           # Dimensions defined in parameters
           # pixels copied from sample 
           # started with one randomic seed from sample 
           # the template grows here
           # base to pattern reconstruction
           # base to template matching comparsion 

# result_window - the result window in rgb color of template synthesis. 
           # same dimensions defined in parameters
           # pixels copied from rgb original sample 
           # same pixels coords defined in window seach


# mask  - information about pixels already synthesized, 
           # same dimension of window and result_window
           


# sample_semantic_mask - diferent regions of patterns of input image 
                   # 0 salt,
                   # 1 litofacies,
                   # 2 frontier
                 # oringinal sample image size

# generat_mask  -  mask to generate, same result windows size 


# patch mask - fragment to syntehtize

# control mask - mask to control the synthesis

# control mask expansion - new control mask expansion with no interruptions

