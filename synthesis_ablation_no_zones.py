'''
Ablation Study: Texture Synthesis WITHOUT Zone Separation

This is a simplified version of synthesis.py that removes the context-oriented
zone separation (salt, rock, boundary) to evaluate its importance.

Key differences from the original:
- NO division into edge, zone0, zone1
- NO angle-based patch selection from database
- Uses complete original image as single texture source
- Single-pass synthesis using only the basic Efros-Leung algorithm
'''

__author__ = 'Luciano Terres'

INSPECT = False

import argparse
import cv2
import largestinteriorrectangle as lir
import numpy as np
import time
import uuid
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from datetime import datetime
import traceback

EIGHT_CONNECTED_NEIGHBOR_KERNEL = np.array([[1., 1., 1.],
                                            [1., 0., 1.],
                                            [1., 1., 1.]], dtype=np.float64)
SIGMA_COEFF = 6.4
ERROR_THRESHOLD = 0.1


def log_output(enabled, *args, **kwargs):
    if enabled:
        print(*args, **kwargs)


def normalized_ssd(sample, window, mask):
    wh, ww = window.shape
    sh, sw = sample.shape

    strided_sample = np.lib.stride_tricks.as_strided(sample, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(sample.strides[0], sample.strides[1], sample.strides[0], sample.strides[1]))
    strided_sample = strided_sample.reshape(-1, wh, ww)

    strided_window = np.lib.stride_tricks.as_strided(window, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, window.strides[0], window.strides[1]))
    strided_window = strided_window.reshape(-1, wh, ww)

    strided_mask = np.lib.stride_tricks.as_strided(mask, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, mask.strides[0], mask.strides[1]))
    strided_mask = strided_mask.reshape(-1, wh, ww)

    sigma = wh / SIGMA_COEFF
    kernel = cv2.getGaussianKernel(ksize=wh, sigma=sigma)
    kernel_2d = kernel * kernel.T

    strided_kernel = np.lib.stride_tricks.as_strided(kernel_2d, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, kernel_2d.strides[0], kernel_2d.strides[1]))
    strided_kernel = strided_kernel.reshape(-1, wh, ww)

    squared_differences = ((strided_sample - strided_window)**2) * strided_kernel * strided_mask
    ssd = np.sum(squared_differences, axis=(1,2))
    ssd = ssd.reshape(sh-wh+1, sw-ww+1)

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

    if method == 'best':
        # Deterministic: always pick the lowest-error candidate.
        candidate_ssd = normalized_ssd[indices]
        best_pos = np.argmin(candidate_ssd)
        return (
            np.array([indices[0][best_pos]]),
            np.array([indices[1][best_pos]])
        )

    if method == 'uniform':
        weights = np.ones(N) / float(N)
    else:
        # Weighted random choice favoring lower SSD values.
        weights = 1.0 / (normalized_ssd[indices] + 1e-12)
        weights = weights / np.sum(weights)

    selection = np.random.choice(np.arange(N), size=1, p=weights)
    selected_index = (indices[0][selection], indices[1][selection])
    return selected_index


def get_neighboring_pixel_indices(pixel_mask):
    kernel = np.ones((3,3))
    dilated_mask = cv2.dilate(pixel_mask, kernel, iterations=1)
    neighbors = dilated_mask - pixel_mask
    neighbor_indices = np.nonzero(neighbors)
    return neighbor_indices


def permute_neighbors(pixel_mask, neighbors):
    N = neighbors[0].shape[0]
    permuted_indices = np.random.permutation(np.arange(N))
    permuted_neighbors = (neighbors[0][permuted_indices], neighbors[1][permuted_indices])

    neighbor_count = cv2.filter2D(pixel_mask, ddepth=-1, kernel=EIGHT_CONNECTED_NEIGHBOR_KERNEL, borderType=cv2.BORDER_CONSTANT)
    permuted_neighbor_counts = neighbor_count[permuted_neighbors]

    sorted_order = np.argsort(permuted_neighbor_counts)[::-1]
    permuted_neighbors = (permuted_neighbors[0][sorted_order], permuted_neighbors[1][sorted_order])

    return permuted_neighbors


def texture_can_be_synthesized(mask):
    # The texture can be synthesized while the mask has unfilled entries.
    mh, mw = mask.shape[:2]
    num_completed = np.count_nonzero(mask)
    num_incomplete = (mh * mw) - num_completed
    
    return num_incomplete > 0


def update(original_sample):
    sample = cv2.cvtColor(original_sample, cv2.COLOR_BGR2GRAY)
    return sample.astype(np.float64) / 255.


def initialize(
    original_sample, window_size, kernel_size, controlMask, seed_mode='random'
):
    sample = update(original_sample)

    window = np.zeros(window_size, dtype=np.float64)

    if original_sample.ndim == 2:
        result_window = np.zeros_like(window, dtype=np.uint8)
    else:
        result_window = np.zeros(window_size + (3,), dtype=np.uint8)

    sh, sw = original_sample.shape[:2]
    if seed_mode == 'center':
        ih = max((sh // 2) - 1, 0)
        iw = max((sw // 2) - 1, 0)
    else:
        ih = np.random.randint(sh-3+1)
        iw = np.random.randint(sw-3+1)
    seed = sample[ih:ih+3, iw:iw+3]

    h, w = window.shape
    mask = np.zeros((h, w), dtype=np.float64)

    # Place seed in center of window
    ph, pw = (h//2)-1, (w//2)-1
    window[ph:ph+3, pw:pw+3] = seed
    mask[ph:ph+3, pw:pw+3] = 1
    result_window[ph:ph+3, pw:pw+3] = original_sample[ih:ih+3, iw:iw+3]

    win = kernel_size//2
    padded_window = cv2.copyMakeBorder(window,
                                       top=win, bottom=win, left=win, right=win, borderType=cv2.BORDER_CONSTANT, value=0.)
    padded_mask = cv2.copyMakeBorder(mask,
                                     top=win, bottom=win, left=win, right=win, borderType=cv2.BORDER_CONSTANT, value=0.)

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
    return ph, pw


def analizeMetrics(original_sample, resultRGBW, verbose=False):
    from metrics import mse, dssim, lbp_tile_distance
    m = mse(original_sample, resultRGBW)
    s = dssim(original_sample, resultRGBW)
    euclidean_distance = lbp_tile_distance(original_sample, resultRGBW)

    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'DLBP', 'DSSIM'],
        'Value': [m, euclidean_distance, s]
    })
    log_output(verbose, metrics_df)
    return m, euclidean_distance, s


def synthesize_ablated(
    original_sample,
    window_size,
    kernel_size,
    visualize,
    verbose=False,
    selection_method='uniform',
    seed_mode='random',
    error_threshold=ERROR_THRESHOLD
):
    """
    ABLATED VERSION: Single-pass synthesis without zone separation.
    Uses the complete original image as texture source.
    """
    log_output(
        verbose, "Starting ABLATED texture synthesis (no zone separation)..."
    )

    # Use complete original image as texture source
    sample = update(original_sample)

    # Initialize synthesis
    _, window, mask, padded_window, padded_mask, result_window = initialize(
        original_sample, window_size, kernel_size, None, seed_mode=seed_mode
    )

    # Single-pass synthesis: fill entire window
    while texture_can_be_synthesized(mask):
        neighboring_indices = get_neighboring_pixel_indices(mask)
        neighboring_indices = permute_neighbors(mask, neighboring_indices)

        for ch, cw in zip(neighboring_indices[0], neighboring_indices[1]):
            window_slice = padded_window[ch:ch+kernel_size, cw:cw+kernel_size]
            mask_slice = padded_mask[ch:ch+kernel_size, cw:cw+kernel_size]

            # Compute SSD and select best matching pixel
            ssd = normalized_ssd(sample, window_slice, mask_slice)
            indices = get_candidate_indices(
                ssd, error_threshold=error_threshold
            )
            selected_index = select_pixel_index(
                ssd, indices, method=selection_method
            )

            # Translate index to accommodate padding
            selected_index = (selected_index[0] + kernel_size // 2, selected_index[1] + kernel_size // 2)

            # Set windows and mask
            window[ch, cw] = sample[selected_index]
            mask[ch, cw] = 1
            result_window[ch, cw] = original_sample[selected_index[0], selected_index[1]]

            if visualize:
                cv2.imshow('ablated synthesis window', result_window)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    return result_window

    if visualize:
        cv2.imshow('ablated synthesis window', result_window)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_window


def inspect(img, title=None, verbose=False):
    if INSPECT:
        log_output(verbose, img.shape)
        showResult(img, title, 70, 200)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def showResult(resultRGBWindow, title=None, coluna=50, linha=200):
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
    parser = argparse.ArgumentParser(description='Ablation Study: Texture synthesis WITHOUT zone separation')
    parser.add_argument('--sample_path', type=str, required=True, help='Path to texture sample')
    parser.add_argument('--generat_mask_path', type=str, required=False, help='Path to generation mask')
    parser.add_argument('--out_path', type=str, required=False, help='Output path for synthesized texture')
    parser.add_argument('--window_height', type=int, required=False, default=50, help='Height of the synthesis window')
    parser.add_argument('--window_width', type=int, required=False, default=50, help='Width of the synthesis window')
    parser.add_argument('--kernel_size', type=int, required=False, default=11, help='One dimension of the square synthesis kernel')
    parser.add_argument('--visualize', required=False, action='store_true', help='Visualize the synthesis process')
    parser.add_argument('--iterations', type=int, required=False, default=50, help='Number of synthesis iterations (default: 50)')
    parser.add_argument(
        '--seed',
        type=int,
        required=False,
        default=None,
        help='Random seed for reproducible synthesis runs'
    )
    parser.add_argument(
        '--selection_method',
        type=str,
        required=False,
        default='uniform',
        choices=['uniform', 'weighted', 'best'],
        help='Pixel candidate selection strategy'
    )
    parser.add_argument(
        '--seed_mode',
        type=str,
        required=False,
        default='random',
        choices=['random', 'center'],
        help='How the initial 3x3 seed is chosen from sample'
    )
    parser.add_argument(
        '--error_threshold',
        type=float,
        required=False,
        default=ERROR_THRESHOLD,
        help='Candidate acceptance threshold relative to min SSD'
    )
    parser.add_argument(
        '--verbose',
        required=False,
        action='store_true',
        help='Enable console output during execution'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    sample = cv2.imread(args.sample_path)
    if sample is None:
        raise ValueError('Unable to read image from sample_path.')

    if args.generat_mask_path and args.generat_mask_path != "none":
        generat_mask = cv2.imread(args.generat_mask_path)
        generat_mask = cv2.cvtColor(generat_mask, cv2.COLOR_BGR2GRAY)
        if generat_mask is None:
            raise ValueError('Unable to read image from generat_mask.')
    else:
        idim = args.window_height
        jdim = args.window_width
        generat_mask = np.ones((idim, jdim), dtype=int)

    validate_args(args)

    # Create output directory
    os.makedirs("result", exist_ok=True)
    run_id = os.path.basename(args.sample_path).split('.')[0]
    run_dir_prefix = 'abl_nz'
    run_dir = os.path.join('result', f'{run_dir_prefix}_{run_id}')

    if os.path.exists(run_dir):
        i = 1
        while os.path.exists(run_dir):
            run_dir = os.path.join('result', f'{run_dir_prefix}_{run_id}_{i}')
            i += 1
    os.makedirs(run_dir, exist_ok=False)
    stats_dir = os.path.join(run_dir, 'data')
    os.makedirs(stats_dir, exist_ok=False)
    compare_base_dir = stats_dir
    if args.out_path:
        out_parent = os.path.dirname(os.path.abspath(args.out_path))
        out_grandparent = os.path.dirname(out_parent)
        out_parent_name = os.path.basename(out_parent)
        if out_parent_name.startswith('sample_'):
            compare_base_dir = out_grandparent
            os.makedirs(compare_base_dir, exist_ok=True)
        elif (
            out_parent_name.startswith('compara')
            and out_parent_name[7:].isdigit()
        ):
            compare_base_dir = out_parent
            os.makedirs(compare_base_dir, exist_ok=True)

    log_output(args.verbose, "*****************")
    log_output(args.verbose, "ABLATION STUDY: No Zone Separation")
    log_output(args.verbose, f"Results will be saved to: {run_dir}")
    log_output(args.verbose, "*****************")

    durations = []
    metrics_rows = []
    n = args.iterations
    i = -1

    try:
        for i in range(n):
            tic = time.time()
            log_output(args.verbose, "*****************")
            synthesized_texture = synthesize_ablated(
                original_sample=sample,
                window_size=(args.window_height, args.window_width),
                kernel_size=args.kernel_size,
                visualize=args.visualize,
                verbose=args.verbose,
                selection_method=args.selection_method,
                seed_mode=args.seed_mode,
                error_threshold=args.error_threshold
            )
            toc = time.time()
            dur = toc - tic
            durations.append(dur)
            log_output(
                args.verbose,
                f"Iteration {i+1}/{n} - Processing time: {dur:.3f}s"
            )

            # Save result
            randomName = str(uuid.uuid4())[:8]
            filename = os.path.join(run_dir, f"{randomName}.jpg")
            cv2.imwrite(filename, synthesized_texture)
            if args.verbose:
                log_output(
                    args.verbose,
                    f'Iteration {i+1}: synthesized texture saved to {filename}'
                )
            else:
                print('.', end='', flush=True)

            # Analyze metrics
            graysample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            synthesized_gray = cv2.cvtColor(synthesized_texture, cv2.COLOR_BGR2GRAY)
            m, lbp_dist, dssim_val = analizeMetrics(
                graysample, synthesized_gray, verbose=args.verbose
            )
            metrics_rows.append({
                'iteration': i+1,
                'output_file': filename,
                'time_sec': dur,
                'mse': m,
                'dssim': dssim_val,
                'lbp_distance': lbp_dist
            })
    except Exception as e:
        if not args.verbose and i >= 0:
            print()
        log_output(
            args.verbose,
            f"Error during iteration {i+1} (loop interrupted): {e}"
        )
        if args.verbose:
            traceback.print_exc()
    finally:
        if not args.verbose and durations:
            print()
        if durations:
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            average_time_message = (
                f"Average time ({len(durations)} successful runs): "
                f"{mean_duration:.3f}s | Std dev: {std_duration:.3f}s"
            )
            log_output(
                args.verbose,
                average_time_message
            )
        else:
            log_output(
                args.verbose, "No successful runs to calculate statistics."
            )

        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_rows)
        sampleName = os.path.splitext(os.path.basename(args.sample_path))[0]
        metrics_csv_path = os.path.join(
            stats_dir, f"metrics_{sampleName}.csv"
        )
        metrics_stats_csv_path = os.path.join(
            compare_base_dir, f"stats_{sampleName}.csv"
        )
        metadata_path = os.path.join(
            stats_dir, f"metadata_{sampleName}.txt"
        )
        run_ts = datetime.now().isoformat(timespec='seconds')

        stats_columns = ['mse', 'dssim', 'lbp_distance']
        available_stats_columns = [
            column for column in stats_columns if column in metrics_df.columns
        ]
        if available_stats_columns:
            metrics_stats_df = metrics_df[available_stats_columns].agg(
                ['min', 'mean', 'median', 'max', 'std']
            ).transpose().reset_index()
            metrics_stats_df = metrics_stats_df.rename(
                columns={'index': 'metric'}
            )
        else:
            metrics_stats_df = pd.DataFrame(
                columns=['metric', 'min', 'mean', 'median', 'max', 'std']
            )

        with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
            f.write("# ABLATION STUDY: No Zone Separation\n")
            f.write(f"# run_dir;{run_dir}\n")
            f.write(f"# run_timestamp;{run_ts}\n")
            f.write(f"# sample_path;{args.sample_path}\n")
            f.write(f"# generat_mask_path;{args.generat_mask_path or 'N/A'}\n")
            f.write(f"# window_height;{args.window_height}\n")
            f.write(f"# window_width;{args.window_width}\n")
            f.write(f"# kernel_size;{args.kernel_size}\n")
            f.write(f"# visualize;{args.visualize}\n")
            f.write(f"# iterations_requested;{args.iterations}\n")
            f.write(f"# iterations_completed;{len(durations)}\n")
            f.write(f"# random_seed;{args.seed}\n")
            f.write(f"# selection_method;{args.selection_method}\n")
            f.write(f"# seed_mode;{args.seed_mode}\n")
            f.write(f"# error_threshold;{args.error_threshold}\n")
            if i >= 0 and len(durations) < n:
                f.write(f"# interrupted_iteration;{i+1}\n")
        metrics_df.to_csv(
            metrics_csv_path, sep=';', index=False, float_format='%.6f'
        )
        metrics_stats_df.to_csv(
            metrics_stats_csv_path, sep=';', index=False, float_format='%.2f'
        )
        log_output(args.verbose, f"Metrics saved to {metrics_csv_path}")
        log_output(
            args.verbose, f"Metrics stats saved to {metrics_stats_csv_path}"
        )
        log_output(args.verbose, f"Run info saved to {metadata_path}")


if __name__ == '__main__':
    main()
