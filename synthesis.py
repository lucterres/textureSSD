'''
This module is a Python implementation of:

    A. Efros and T. Leung, "Texture Synthesis by Non-parametric Sampling,"
    Proceedings of the Seventh IEEE International Conference on Computer
    Vision, September 1999.

Specifically, this module implements texture synthesis by growing a 3x3 texture patch 
pixel-by-pixel. Please see the authors' project page for additional algorithm details: 

    https://people.eecs.berkeley.edu/~efros/research/EfrosLeung.html

Example:

    Generate a 50x50 texture patch from a texture available at the input path and save it to
    the output path. Also, visualize the synthesis process:

        $ python synthesis.py --sample_path=[input path] --out_path=[output path] --visualize

'''

__author__ = 'Maxwell Goldberg'

INSPECT  = False

import argparse
import cv2
import numpy as np
import time
import uuid
import pandas as pd
import suport.patchesMethods as pm

EIGHT_CONNECTED_NEIGHBOR_KERNEL = np.array([[1., 1., 1.],
                                            [1., 0., 1.],
                                            [1., 1., 1.]], dtype=np.float64)
SIGMA_COEFF = 6.4      # The denominator for a 2D Gaussian sigma used in the reference implementation.
ERROR_THRESHOLD = 0.1  # The default error threshold for synthesis acceptance in the reference implementation.


def normalized_ssd(sample, window, mask):
    wh, ww = window.shape
    sh, sw = sample.shape

    # Get sliding window views of the sample, window, and mask.
    strided_sample = np.lib.stride_tricks.as_strided(sample, shape=((sh-wh+1), (sw-ww+1), wh, ww), 
                        strides=(sample.strides[0], sample.strides[1], sample.strides[0], sample.strides[1]))
    strided_sample = strided_sample.reshape(-1, wh, ww)

    # Note that the window and mask views have the same shape as the strided sample, but the kernel is fixed
    # rather than sliding for each of these components.
    strided_window = np.lib.stride_tricks.as_strided(window, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, window.strides[0], window.strides[1]))
    strided_window = strided_window.reshape(-1, wh, ww)

    strided_mask = np.lib.stride_tricks.as_strided(mask, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, mask.strides[0], mask.strides[1]))
    strided_mask = strided_mask.reshape(-1, wh, ww)

    # Form a 2D Gaussian weight matrix from symmetric linearly separable Gaussian kernels and generate a 
    # strided view over this matrix.
    sigma = wh / SIGMA_COEFF
    kernel = cv2.getGaussianKernel(ksize=wh, sigma=sigma)
    kernel_2d = kernel * kernel.T

    strided_kernel = np.lib.stride_tricks.as_strided(kernel_2d, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, kernel_2d.strides[0], kernel_2d.strides[1]))
    strided_kernel = strided_kernel.reshape(-1, wh, ww)

    # Take the sum of squared differences over all sliding sample windows and weight it so that only existing neighbors
    # contribute to error. Use the Gaussian kernel to weight central values more strongly than distant neighbors.
    squared_differences = ((strided_sample - strided_window)**2) * strided_kernel * strided_mask
    ssd = np.sum(squared_differences, axis=(1,2))
    ssd = ssd.reshape(sh-wh+1, sw-ww+1)

    # Normalize the SSD by the maximum possible contribution.
    total_ssd = np.sum(mask * kernel_2d)
    normalized_ssd = ssd / total_ssd

    return normalized_ssd

def get_candidate_indices(normalized_ssd, error_threshold=ERROR_THRESHOLD):
    min_ssd = np.min(normalized_ssd)
    min_threshold = min_ssd * (1. + error_threshold)
    indices = np.where(normalized_ssd <= min_threshold)
    return indices

def select_pixel_index(normalized_ssd, indices, method='uniform'):
    N = indices[0].shape[0]

    if method == 'uniform':
        weights = np.ones(N) / float(N)
    else:
        weights = normalized_ssd[indices]
        weights = weights / np.sum(weights)

    # Select a random pixel index from the index list.
    selection = np.random.choice(np.arange(N), size=1, p=weights)
    selected_index = (indices[0][selection], indices[1][selection])
    return selected_index

def get_neighboring_pixel_indices(pixel_mask):
    # Taking the difference between the dilated mask and the initial mask
    # gives only the 8-connected neighbors of the mask frontier.
    kernel = np.ones((3,3))
    dilated_mask = cv2.dilate(pixel_mask, kernel, iterations=1)
    neighbors = dilated_mask - pixel_mask

    # Recover the indices of the mask frontier.
    neighbor_indices = np.nonzero(neighbors)

    return neighbor_indices

def permute_neighbors(pixel_mask, neighbors):
    N = neighbors[0].shape[0]

    # Generate a permutation of the neigboring indices
    permuted_indices = np.random.permutation(np.arange(N))
    permuted_neighbors = (neighbors[0][permuted_indices], neighbors[1][permuted_indices])

    # Use convolution to count the number of existing neighbors for all entries in the mask.
    neighbor_count = cv2.filter2D(pixel_mask, ddepth=-1, kernel=EIGHT_CONNECTED_NEIGHBOR_KERNEL, borderType=cv2.BORDER_CONSTANT)

    # Sort the permuted neighboring indices by quantity of existing neighbors descending.
    permuted_neighbor_counts = neighbor_count[permuted_neighbors]

    sorted_order = np.argsort(permuted_neighbor_counts)[::-1]
    permuted_neighbors = (permuted_neighbors[0][sorted_order], permuted_neighbors[1][sorted_order])

    return permuted_neighbors

def nCompletePix(mask):
    # The texture can be synthesized while the mask has unfilled entries.
    num_completed = np.count_nonzero(mask)
    return num_completed

# Convert original to sample representation.
def update(original_sample):
    sample =  cv2.cvtColor(original_sample, cv2.COLOR_BGR2GRAY)
    # Convert sample to floating point and normalize to the range [0., 1.]
    return sample.astype(np.float64)/ 255.
    
def initialize(original_sample, window_size, kernel_size, controlMask):
    sample = update(original_sample)

    # Generate window
    window = np.zeros(window_size, dtype=np.float64) # dtype=np.float64)

    # Generate output window
    if original_sample.ndim == 2:
        result_window = np.zeros_like(window, dtype=np.uint8)
    else:
        result_window = np.zeros(window_size + (3,), dtype=np.uint8)

    # Initialize window with random seed from sample     
    sh, sw = original_sample.shape[:2]
    #ih = np.random.randint(sh-3+1)
    #iw = np.random.randint(sw-3+1)
    # get seed from center of sample
    ih = (sh//2)-3+1
    iw = (sw//2)-3+1
    seed = sample[ih:ih+3, iw:iw+3]

    # Generate window mask
    h, w = window.shape
    mask = np.zeros((h, w), dtype=np.float64)
    
    inside = True
    if inside: # Place seed inside mask target zone
        #random choose one pixel of controlMAsk == 1
        ph, pw = findInsideMaskPixel(controlMask)
        window[ph:ph+3, pw:pw+3] = seed
        mask[ph:ph+3, pw:pw+3] = 1
        result_window[ph:ph+3, pw:pw+3] = original_sample[ih:ih+3, iw:iw+3]
    else:
        # Place seed in center of window 
        # ph, pw = (h//2)-1, (w//2)-1
        # Place seed inside edge zone
        ph,pw = 62,50
        window[ph:ph+3, pw:pw+3] = seed
        mask[ph:ph+3, pw:pw+3] = 1
        result_window[ph:ph+3, pw:pw+3] = original_sample[ih:ih+3, iw:iw+3]

    # Obtain padded versions of window and mask
    win = kernel_size//2
    padded_window = cv2.copyMakeBorder(window, 
                                       top=win, bottom=win, left=win, right=win, borderType=cv2.BORDER_CONSTANT, value=0.)
    padded_mask = cv2.copyMakeBorder(mask,
                                     top=win, bottom=win, left=win, right=win, borderType=cv2.BORDER_CONSTANT, value=0.)
    
    # Obtain views of the padded window and mask
    window = padded_window[win:-win, win:-win]
    mask = padded_mask[win:-win, win:-win]

    return sample, window, mask, padded_window, padded_mask, result_window

def findInsideMaskPixel(controlMask):
    sh, sw = controlMask.shape[:2]
    ph = np.random.randint(sh)
    pw = np.random.randint(sw)
    while controlMask[ph,pw] == 0.0:
        ph = np.random.randint(sh)
        pw = np.random.randint(sw)
    return ph,pw

def synthesize(origRGBSample, semantic_mask, generat_mask, window_size, kernel_size, visualize):
    # Convert original to sample representation.

    def fillSample():
        # Synthesize texture until all pixels in the window are filled.
        while nCompletePix(controlMask)>nCompletePix(doneWindow):
            # Get neighboring indices
            neighboring_indices = get_neighboring_pixel_indices(doneWindow)
            # Permute and sort neighboring indices by quantity of 8-connected neighbors.
            neighboring_indices = permute_neighbors(doneWindow, neighboring_indices)
            # Iterate over neighboring indices.      
            for ch, cw in zip(neighboring_indices[0], neighboring_indices[1]):
                if (controlMask[ch, cw] > 0.0):
                    """
                    if generat_mask[ch, cw]==1.0: # 
                        sample=sampleZone0
                    if generat_mask[ch, cw]==2.0: # 
                        sample=sampleEdge
                    if generat_mask[ch, cw]==3.0: # 
                        sample=sampleZone1     
                    """                                   
                    windowPatchSlice = padded_window[ch:ch+kernel_size, cw:cw+kernel_size]
                    maskPatchSlice = padded_mask[ch:ch+kernel_size, cw:cw+kernel_size]

                    # Compute SSD for the current pixel neighborhood and select an index with low error.
                    ssd = normalized_ssd(sampleGray, windowPatchSlice, maskPatchSlice)
                    indices = get_candidate_indices(ssd)
                    selected_index = select_pixel_index(ssd, indices)

                    # Translate index to accommodate padding.
                    selected_index = (selected_index[0] + kernel_size // 2, selected_index[1] + kernel_size // 2)

                    # Set windows and mask.
                    resultGrayW[ch, cw] = sampleGray[selected_index]
                    doneWindow[ch, cw] = 1
                    resultRGBW[ch, cw] = origRGBSample[selected_index[0], selected_index[1]]
                    
                    if visualize:
                        showResult(resultRGBW)
                        key = cv2.waitKey(1) 
                        if key == 27:
                            cv2.destroyAllWindows()
                            return resultRGBW
                        
    # discover generation segments and angles 
    genSegments,l = pm.probHough(generat_mask, generat_mask, tresh = 10, minPoints=15, maxGap=10, sort=False)
    #iterate over patches and angle segments
    if genSegments is None: # if patches not null makePatchMask
        print("Error: no patches found for this image")
        return None


    # build patches database to find the nearest patch according to angle
    samplesPatchesDB = pm.loadDataBase(600,2000)    
    # create the interface edge dilated 
    dilated_edge, zone0, zone1, fullmask = pm.create_Masks(generat_mask)
    zones = [dilated_edge, zone0, zone1, fullmask]
    inspect(fullmask, "fullmask")
    inspect(dilated_edge, "edge")
    inspect(zone0, "zone0")
    inspect(zone1, "zone1" )

    completeMask = generat_mask.copy() #later we will complete the generation mask
    original_sample = origRGBSample.copy()
    
    sampleGray=0
    resultGrayW=0
    doneWindow=0
    padded_window=0
    padded_mask=0
    resultRGBW=0
    start = True
    rotate = False
    
    # first step is to generate the edge zone 
    generat_mask = dilated_edge
    controlMask = np.zeros(generat_mask.shape)
    for p in (genSegments):
        # Iterate over patches and angle segments
        origRGBSample = pm.searchNearestKey(samplesPatchesDB, p.angle)
        inspect(origRGBSample, "origRGBSample")
        x1,y1,x2,y2 = p.line
        patchMask = pm.makePatchMask(generat_mask, x1, x2)
        controlMask = controlMask + patchMask
        inspect(patchMask, "patchMask")
        inspect(controlMask, "controlMask")
        # Verifica pixels marcados na janela de síntese.
        y_indices, x_indices = np.where(controlMask > 0)
        # Encontre o valor mínimo de X
        min_x = np.min(x_indices)
        max_x = np.max(x_indices)
        controlMask = pm.makePatchMask(generat_mask, min_x, max_x)
        inspect(controlMask, "controlMask expanded")
        if start:
                (sampleGray, resultGrayW, doneWindow, padded_window, 
                padded_mask, resultRGBW) = initialize(origRGBSample, window_size, kernel_size,controlMask)
                start = False
        else:
            if rotate:
                sampleGray = pm.rotateImage(sampleGray,57)
            else: 
                sampleGray=update(origRGBSample)
        fillSample()


    inspect(original_sample, "Sample")
    inspect(semantic_mask, "semantic_mask")
    sampleEdge, sampleZ1, sampleZ0 = pm.sampleBreak(original_sample, semantic_mask)
    inspect(sampleZ1, "Sample Z1")

    # second step is to generate the zone1
    origRGBSample = sampleZ1
    sampleGray=origRGBSample.astype(np.float64)/ 255.
    controlMask = controlMask + zone1
    inspect(controlMask, "controlMask + zone expanded")
    inspect(sampleGray, "sampleGray")
    fillSample()

    # third step is to generate the zone0
    origRGBSample = sampleZ0
    inspect(sampleZ0, "Sample Z0")
    sampleGray=origRGBSample.astype(np.float64)/ 255.
    controlMask = controlMask + zone0
    inspect(controlMask, "controlMask + zone expanded")
    
    inspect(sampleGray, "sampleGray")
    fillSample()

    # fourth step is to complete the full mask
    controlMask = controlMask + fullmask
    inspect(controlMask, "controlMask + fullmask expanded")
    fillSample

    cv2.waitKey(0)
    if visualize:
        inspect(resultRGBW, "Sinthesys" )

    return resultRGBW

def inspect(img,title=None):  
    if INSPECT:
        print(img.shape)
        showResult(img,title)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
     
def showResult(resultRGBWindow, title=None):
    img = cv2.resize(resultRGBWindow, (0, 0), fx=8, fy=8)
    cv2.imshow(title, img)
    cv2.moveWindow(title, 300, 200)

def validate_args(args):
    wh, ww = args.window_height, args.window_width
    if wh < 3 or ww < 3:
        raise ValueError('window_size must be greater than or equal to (3,3).')

    if args.kernel_size <= 1:
        raise ValueError('kernel size must be greater than 1.')

    if args.kernel_size % 2 == 0:
        raise ValueError('kernel size must be odd.')

    if args.kernel_size > min(wh, ww):
        raise ValueError('kernel size must be less than or equal to the smaller window_size dimension.')

def parse_args():
    parser = argparse.ArgumentParser(description='Perform texture synthesis')
    parser.add_argument('--sample_path', type=str, required=True, help='Path to texture sample')
    parser.add_argument('--sample_semantic_mask_path', type=str, required=False, help='Path to semantic mask')
    parser.add_argument('--generat_mask_path', type=str, required=False, help='Path to geracional mask')
    parser.add_argument('--out_path', type=str, required=False, help='Output path for synthesized texture')
    parser.add_argument('--window_height', type=int,  required=False, default=50, help='Height of the synthesis window')
    parser.add_argument('--window_width', type=int, required=False, default=50, help='Width of the synthesis window')
    parser.add_argument('--kernel_size', type=int, required=False, default=11, help='One dimension of the square synthesis kernel')
    parser.add_argument('--visualize', required=False, action='store_true', help='Visualize the synthesis process')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    sample = cv2.imread(args.sample_path)
    if sample is None:
        raise ValueError('Unable to read image from sample_path.')
    
    if args.sample_semantic_mask_path != "":
        sample_semantic_mask = cv2.imread(args.sample_semantic_mask_path)
        sample_semantic_mask = cv2.cvtColor(sample_semantic_mask, cv2.COLOR_BGR2GRAY) 
        if sample_semantic_mask is None:
            raise ValueError('Unable to read image from sample_path.')
    else:
        sample_semantic_mask = sample
            
    if args.generat_mask_path != "none":
        generat_mask = cv2.imread(args.generat_mask_path)
        generat_mask = cv2.cvtColor(generat_mask, cv2.COLOR_BGR2GRAY) 
        if generat_mask is None:
            raise ValueError('Unable to read image from sample_path.')
    else:
        idim = args.window_height
        jdim = args.window_width
        generat_mask = np.ones((idim,jdim),dtype=int)

    validate_args(args)

    tic = time.time() 
    synthesized_texture= synthesize(origRGBSample=sample, 
                                    semantic_mask = sample_semantic_mask,
                                    generat_mask = generat_mask,
                                    window_size=(args.window_height, args.window_width), 
                                    kernel_size=args.kernel_size, 
                                    visualize=args.visualize)
    toc = time.time()
    print ("Tempo de processamento:" , toc - tic);

    # save result
    filename = "result/" + str(uuid.uuid4())[:8] + ".jpg"
    cv2.imwrite(filename, synthesized_texture)

if __name__ == '__main__':
    main()