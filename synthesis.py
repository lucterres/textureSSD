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
# based on original work of 'Maxwell Goldberg'
__author__ = 'Luciano Terres' 


INSPECT  = False

import argparse
import cv2
import largestinteriorrectangle as lir
import numpy as np
import time 
import uuid
import pandas as pd 
import suport.patchesMethods as pm
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from datetime import datetime
import traceback


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

def analizeMetrics(original_sample, resultRGBW):
    # Calculate metrics
    from metrics import mse, dssim, lbp_tile_distance
    # compute the MSE between the two images
    m = mse(original_sample, resultRGBW)
    # compute the SSIM between the two images
    s = dssim(original_sample, resultRGBW)
    # compute euclidean distance
    euclidean_distance = lbp_tile_distance(original_sample, resultRGBW)

    # Create a DataFrame to display the results
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'DLBP' , 'DSSIM'],
        'Value': [m, euclidean_distance, s]
    })
    print(metrics_df)
    # Return raw values for programmatic usage
    return m, euclidean_distance, s
    
def synthesize(origRGBSample, semantic_mask, generat_mask, window_size, kernel_size, visualize, patches_db=None):
    # Convert original to sample representation.
    print ("Starting texture synthesis...")

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
                        showResult(resultRGBW, "generation",50,100)
                        #showResult(doneWindow, "done Window",900,100)
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

    # use provided patches_db (loaded once) or fallback
    if patches_db is None:
        try:
            cache_path = globals().get('runtime_args').patches_cache_path if 'runtime_args' in globals() else 'result/patches_db_cache.npz'
            rebuild_flag = globals().get('runtime_args').rebuild_patches_db if 'runtime_args' in globals() else False
            if cache_path == 'none':
                cache_path = None
            samplesPatchesDB = pm.loadDataBase(cache_path=cache_path, rebuild=rebuild_flag)
        except Exception:
            samplesPatchesDB = pm.loadDataBase()
    else:
        samplesPatchesDB = patches_db
    # create the interface edge dilated 
    dilated_edge, zone0, zone1, fullmask = pm.create_Masks(generat_mask)
    zones = [dilated_edge, zone0, zone1, fullmask]
    inspect(fullmask, "fullmask")
    inspect(generat_mask, "generat_mask")
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
    for s in (genSegments):
        # Iterate over patches and angle segments
        #origRGBSample = pm.searchNearestPatchByAngle(samplesPatchesDB, s.angle)
        origRGBSample = pm.searchNearestPatchByAngleAndHistogram(samplesPatchesDB, s.angle, origRGBSample)
        inspect(origRGBSample, "origRGBSample")
        x1,y1,x2,y2 = s.line
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
        
    # complete edge zone with the last sample patch
    controlMask = dilated_edge
    inspect(controlMask, "controlMask")
    inspect(sampleGray, "sampleGray")
    fillSample()

    inspect(original_sample, "Sample")
    inspect(semantic_mask, "semantic_mask")
    sampleEdge, sampleZ0, sampleZ1 = pm.sampleBreak(original_sample, semantic_mask)
    inspect(sampleZ0, "sample Z0")
    inspect(sampleZ1, "Sample Z1")
    # second step is to generate the zone1
    origRGBSample = extractBiggestSquare(sampleZ1)
    inspect(origRGBSample, "Biggest Square in Z1")
    sampleGray=origRGBSample.astype(np.float64)/ 255.
    controlMask = controlMask + zone1
    inspect(controlMask, "controlMask + zone expanded")
    inspect(sampleGray, "sampleGray")
    fillSample()

    # third step is to generate the zone0

    # Find the biggest square filled in sampleZ0
    origRGBSample = extractBiggestSquare(sampleZ0)
    sampleGray=origRGBSample.astype(np.float64)/ 255.
    controlMask = controlMask + zone0
    inspect(sampleZ0, "Sample Z0")
    inspect(origRGBSample, "Biggest Square in Z0")
    inspect(controlMask, "controlMask + zone expanded")
    inspect(sampleGray, "sampleGray")
    fillSample()

    # fourth step is to complete the full mask
    controlMask = controlMask + fullmask
    inspect(controlMask, "controlMask + fullmask expanded")
    fillSample()

    #cv2.waitKey(0)
    if visualize:
        inspect(controlMask, "final controlMask")
        inspect(resultRGBW, "Sinthesys" )
   

    return resultRGBW

def extractBiggestSquare(sampleZ0):
    _, mask = cv2.threshold(sampleZ0, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = np.array([contours[0][:, 0, :]])
    inner_bb = lir.lir(contour)

    cropped_img = sampleZ0[inner_bb[1]:inner_bb[1] + inner_bb[3],
                  inner_bb[0]:inner_bb[0] + inner_bb[2]]

    return cropped_img

def inspect(img,title=None):  
    if INSPECT:
        print(img.shape)
        showResult(img,title,70,200)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
     
def showResult(resultRGBWindow, title=None,coluna=50,linha=200):
    #coluna no segundo monitor
    img = cv2.resize(resultRGBWindow, (0, 0), fx=8, fy=8)
    cv2.imshow(title, img)
    cv2.moveWindow(title, coluna, linha)

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
    parser.add_argument('--patches_cache_path', type=str, required=False, default='result/patches_db_cache.npz',
                        help='Path to cache the patches database (.npz). Use "none" to disable caching.')
    parser.add_argument('--rebuild_patches_db', action='store_true', help='Force rebuild of patches DB even if cache exists')
    parser.add_argument('--iterations', type=int, required=False, default=50, help='Number of synthesis iterations (default: 50)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # store globally for synthesize() optional cache usage
    globals()['runtime_args'] = args
    sample = cv2.imread(args.sample_path)
    if sample is None:
        raise ValueError('Unable to read image from sample_path.')
    
    if args.sample_semantic_mask_path != "":
        sample_semantic_mask = cv2.imread(args.sample_semantic_mask_path)
        sample_semantic_mask = cv2.cvtColor(sample_semantic_mask, cv2.COLOR_BGR2GRAY) 
        if sample_semantic_mask is None:
            raise ValueError('Unable to read image from sample_semantic_mask.')
    else:
        sample_semantic_mask = sample
            
    if args.generat_mask_path != "none":
        generat_mask = cv2.imread(args.generat_mask_path)
        generat_mask = cv2.cvtColor(generat_mask, cv2.COLOR_BGR2GRAY) 
        if generat_mask is None:
            raise ValueError('Unable to read image from generat_mask.')
    else:
        idim = args.window_height
        jdim = args.window_width
        generat_mask = np.ones((idim,jdim),dtype=int)

    validate_args(args)

    # Repeat synthesis n times and record timings
    os.makedirs("result", exist_ok=True)
    # Create a unique subfolder for this run 
    # run_id = nome do sample
    run_id = os.path.basename(args.sample_path).split('.')[0]
    run_dir = os.path.join('result', f'run_{run_id}')
    # se existir o diretorio rundir  acrescenta um numero no final
    if os.path.exists(run_dir):
        i = 1
        while os.path.exists(run_dir):
            run_dir = os.path.join('result', f'run_{run_id}_{i}')
            i += 1
    os.makedirs(run_dir, exist_ok=False)
    print ("*****************")
    print(f"Resultados desta execução serão salvos em: {run_dir}")
    print ("*****************")
    durations = []
    metrics_rows = []  # collect per-iteration metrics
    n = args.iterations  # number of synthesis iterations
    metrics_csv_path = None
    i = -1  # track iteration for error reporting
    # Load / build patches DB once before loop
    patches_db = None
    try:
        cache_path = args.patches_cache_path
        rebuild_flag = args.rebuild_patches_db
        if cache_path == 'none':
            cache_path = None
        patches_db = pm.loadDataBase(cache_path=cache_path, rebuild=rebuild_flag)
        print(f"Patches DB carregado: {len(patches_db)} patches.")
    except Exception as e:
        print(f"Falha ao carregar patches DB com cache (fallback para rebuild in-memory): {e}")
        try:
            patches_db = pm.loadDataBase()
        except Exception as ee:
            print(f"Falha ao construir patches DB sem cache: {ee}")
            patches_db = None
    print ("*****************")
    try:
        for i in range(n):
            tic = time.time()
            print ("*****************")
            synthesized_texture = synthesize(origRGBSample=sample,
                                             semantic_mask=sample_semantic_mask,
                                             generat_mask=generat_mask,
                                             window_size=(args.window_height, args.window_width),
                                             kernel_size=args.kernel_size,
                                             visualize=args.visualize,
                                             patches_db=patches_db)
            toc = time.time()
            dur = toc - tic
            durations.append(dur)
            print(f"Iteração {i+1}/{n} - Tempo de processamento: {dur:.3f}s")

            # save result of this iteration
            randomName = str(uuid.uuid4())[:8]
            filename = os.path.join(run_dir, f"{randomName}.jpg")
            cv2.imwrite(filename, synthesized_texture)
            print(f'Iteração {i+1}: textura sintetizada salva em {filename}')

            # Analyze metrics for this iteration (optional: could skip for speed)
            graysample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            synthesized_gray = cv2.cvtColor(synthesized_texture, cv2.COLOR_BGR2GRAY)
            m, lbp_dist, dssim_val = analizeMetrics(graysample, synthesized_gray)
            metrics_rows.append({
                'iteration': i+1,
                'output_file': filename,
                'time_sec': dur,
                'mse': m,
                'dssim': dssim_val,
                'lbp_distance': lbp_dist
            })
    except Exception as e:
        print(f"Erro durante a iteração {i+1} (loop interrompido): {e}")
        traceback.print_exc()
    finally:
        if durations:
            print(f"Tempo médio ({len(durations)} execuções bem-sucedidas): {np.mean(durations):.3f}s | Desvio padrão: {np.std(durations):.3f}s")
        else:
            print("Nenhuma execução bem-sucedida para calcular estatísticas.")

        # Persist metrics to CSV (even if partial)
        # calcule as estatísticas min, max, mean, stddev, q125, median, q75

        metrics_df = pd.DataFrame(metrics_rows)
        #troca randomname por nome da sample
        sampleName = os.path.splitext(os.path.basename(args.sample_path))[0]
        metrics_csv_path = os.path.join(run_dir, f"run_metrics_{sampleName}.csv")
        run_ts = datetime.now().isoformat(timespec='seconds')
        with open(metrics_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"# run_dir;{run_dir}\n")
            f.write(f"# run_timestamp;{run_ts}\n")
            f.write(f"# sample_path;{args.sample_path}\n")
            f.write(f"# sample_semantic_mask_path;{args.sample_semantic_mask_path or 'N/A'}\n")
            f.write(f"# generat_mask_path;{args.generat_mask_path or 'N/A'}\n")
            f.write(f"# window_height;{args.window_height}\n")
            f.write(f"# window_width;{args.window_width}\n")
            f.write(f"# kernel_size;{args.kernel_size}\n")
            f.write(f"# visualize;{args.visualize}\n")
            f.write(f"# iterations_requested;{args.iterations}\n")
            f.write(f"# iterations_completed;{len(durations)}\n")
            if i >= 0 and len(durations) < n:
                f.write(f"# interrupted_iteration;{i+1}\n")
            #TODO gerar estatísticas descritivas
            f.write("# --- métricas por iteração ---\n")
            metrics_df.to_csv(f, sep=';', index=False, float_format='%.6f')
        print(f"Metric results (parciais ou completos) salvos em {metrics_csv_path}")

if __name__ == '__main__':
    main()