'''
Ablation Study Comparison Script

Executes both methods (complete with zones vs ablated without zones) and
generates comparative analysis of results.
'''

__author__ = 'Luciano Terres'

import argparse
import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import subprocess
import sys

# Import synthesis functions from both scripts
import synthesis
import synthesis_ablation_no_zones as ablation


def run_complete_method(sample_path, semantic_mask_path, generat_mask_path, 
                       window_size, kernel_size, iterations, output_dir):
    """Run the complete method with zone separation."""
    print("\n" + "="*80)
    print("RUNNING COMPLETE METHOD (with zone separation)")
    print("="*80 + "\n")
    
    sample = cv2.imread(sample_path)
    semantic_mask = cv2.imread(semantic_mask_path, cv2.IMREAD_GRAYSCALE)
    generat_mask = cv2.imread(generat_mask_path, cv2.IMREAD_GRAYSCALE)
    
    results = []
    durations = []
    
    # Load patches DB once
    try:
        import suport.patchesMethods as pm
        patches_db = pm.loadDataBase(cache_path='result/patches_db_cache.npz', rebuild=False)
        print(f"Patches DB loaded: {len(patches_db)} patches")
    except Exception as e:
        print(f"Warning: Could not load patches DB: {e}")
        patches_db = None
    
    for i in range(iterations):
        print(f"\n--- Complete Method: Iteration {i+1}/{iterations} ---")
        tic = time.time()
        
        try:
            synthesized = synthesis.synthesize(
                origRGBSample=sample,
                semantic_mask=semantic_mask,
                generat_mask=generat_mask,
                window_size=window_size,
                kernel_size=kernel_size,
                visualize=False,
                patches_db=patches_db
            )
            
            toc = time.time()
            dur = toc - tic
            durations.append(dur)
            
            # Save result
            filename = os.path.join(output_dir, f"complete_{i+1:03d}.jpg")
            cv2.imwrite(filename, synthesized)
            
            # Calculate metrics
            from metrics import mse, dssim, lbp_tile_distance
            gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            gray_synth = cv2.cvtColor(synthesized, cv2.COLOR_BGR2GRAY)
            
            m = mse(gray_sample, gray_synth)
            s = dssim(gray_sample, gray_synth)
            lbp = lbp_tile_distance(gray_sample, gray_synth)
            
            results.append({
                'iteration': i+1,
                'method': 'complete',
                'filename': filename,
                'time_sec': dur,
                'mse': m,
                'dssim': s,
                'lbp_distance': lbp
            })
            
            print(f"Time: {dur:.3f}s | MSE: {m:.6f} | DSSIM: {s:.6f} | LBP: {lbp:.6f}")
            
        except Exception as e:
            print(f"ERROR in iteration {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    return pd.DataFrame(results), durations


def run_ablated_method(sample_path, generat_mask_path, window_size, kernel_size, 
                      iterations, output_dir):
    """Run the ablated method without zone separation."""
    print("\n" + "="*80)
    print("RUNNING ABLATED METHOD (without zone separation)")
    print("="*80 + "\n")
    
    sample = cv2.imread(sample_path)
    generat_mask = cv2.imread(generat_mask_path, cv2.IMREAD_GRAYSCALE)
    
    results = []
    durations = []
    
    for i in range(iterations):
        print(f"\n--- Ablated Method: Iteration {i+1}/{iterations} ---")
        tic = time.time()
        
        try:
            synthesized = ablation.synthesize_ablated(
                origRGBSample=sample,
                generat_mask=generat_mask,
                window_size=window_size,
                kernel_size=kernel_size,
                visualize=False
            )
            
            toc = time.time()
            dur = toc - tic
            durations.append(dur)
            
            # Save result
            filename = os.path.join(output_dir, f"ablated_{i+1:03d}.jpg")
            cv2.imwrite(filename, synthesized)
            
            # Calculate metrics
            from metrics import mse, dssim, lbp_tile_distance
            gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            gray_synth = cv2.cvtColor(synthesized, cv2.COLOR_BGR2GRAY)
            
            m = mse(gray_sample, gray_synth)
            s = dssim(gray_sample, gray_synth)
            lbp = lbp_tile_distance(gray_sample, gray_synth)
            
            results.append({
                'iteration': i+1,
                'method': 'ablated',
                'filename': filename,
                'time_sec': dur,
                'mse': m,
                'dssim': s,
                'lbp_distance': lbp
            })
            
            print(f"Time: {dur:.3f}s | MSE: {m:.6f} | DSSIM: {s:.6f} | LBP: {lbp:.6f}")
            
        except Exception as e:
            print(f"ERROR in iteration {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    return pd.DataFrame(results), durations


def generate_statistics(df, metric_name):
    """Generate descriptive statistics for a metric."""
    stats = df.groupby('method')[metric_name].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('q25', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('q75', lambda x: x.quantile(0.75)),
        ('max', 'max')
    ])
    return stats


def save_comparison_report(complete_df, ablated_df, output_dir, args):
    """Generate and save comprehensive comparison report."""
    
    # Combine dataframes
    combined_df = pd.concat([complete_df, ablated_df], ignore_index=True)
    
    # Save raw data
    raw_csv = os.path.join(output_dir, 'raw_results.csv')
    combined_df.to_csv(raw_csv, sep=';', index=False, float_format='%.6f')
    print(f"\nRaw results saved to: {raw_csv}")
    
    # Generate statistics for each metric
    metrics = ['time_sec', 'mse', 'dssim', 'lbp_distance']
    
    stats_file = os.path.join(output_dir, 'comparative_statistics.csv')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"# Ablation Study: Zone Separation Comparison\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Sample: {args.sample_path}\n")
        f.write(f"# Iterations: {args.iterations}\n")
        f.write(f"# Window size: {args.window_height}x{args.window_width}\n")
        f.write(f"# Kernel size: {args.kernel_size}\n")
        f.write("#\n")
        
        for metric in metrics:
            f.write(f"\n# --- {metric.upper()} Statistics ---\n")
            stats = generate_statistics(combined_df, metric)
            stats.to_csv(f, sep=';', float_format='%.6f')
            f.write("\n")
    
    print(f"Comparative statistics saved to: {stats_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        stats = generate_statistics(combined_df, metric)
        print(stats.to_string())
        
        # Calculate percentage difference
        if len(stats) == 2:
            complete_mean = stats.loc['complete', 'mean']
            ablated_mean = stats.loc['ablated', 'mean']
            pct_diff = ((ablated_mean - complete_mean) / complete_mean) * 100
            print(f"  → Ablated vs Complete: {pct_diff:+.2f}% change")
    
    print("\n" + "="*80)


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation Study: Compare methods with/without zone separation')
    parser.add_argument('--sample_path', type=str, required=True, help='Path to texture sample')
    parser.add_argument('--sample_semantic_mask_path', type=str, required=True, help='Path to semantic mask')
    parser.add_argument('--generat_mask_path', type=str, required=True, help='Path to generation mask')
    parser.add_argument('--window_height', type=int, default=101, help='Height of synthesis window')
    parser.add_argument('--window_width', type=int, default=101, help='Width of synthesis window')
    parser.add_argument('--kernel_size', type=int, default=11, help='Synthesis kernel size')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations per method')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (auto-generated if not specified)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Create output directory
    if args.output_dir is None:
        sample_name = os.path.splitext(os.path.basename(args.sample_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join('result', f'ablation_comparison_{sample_name}_{timestamp}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    complete_dir = os.path.join(args.output_dir, 'complete_method')
    ablated_dir = os.path.join(args.output_dir, 'ablated_method')
    os.makedirs(complete_dir, exist_ok=True)
    os.makedirs(ablated_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("ABLATION STUDY: Zone Separation Comparison")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Iterations per method: {args.iterations}")
    print(f"Window size: {args.window_height}x{args.window_width}")
    print(f"Kernel size: {args.kernel_size}")
    
    # Run both methods
    window_size = (args.window_height, args.window_width)
    
    complete_df, complete_times = run_complete_method(
        args.sample_path,
        args.sample_semantic_mask_path,
        args.generat_mask_path,
        window_size,
        args.kernel_size,
        args.iterations,
        complete_dir
    )
    
    ablated_df, ablated_times = run_ablated_method(
        args.sample_path,
        args.generat_mask_path,
        window_size,
        args.kernel_size,
        args.iterations,
        ablated_dir
    )
    
    # Generate comparison report
    save_comparison_report(complete_df, ablated_df, args.output_dir, args)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Complete method images: {complete_dir}")
    print(f"  - Ablated method images: {ablated_dir}")
    print(f"  - Raw results: {os.path.join(args.output_dir, 'raw_results.csv')}")
    print(f"  - Statistics: {os.path.join(args.output_dir, 'comparative_statistics.csv')}")


if __name__ == '__main__':
    main()
