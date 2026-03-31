# Ablation Study: No Zone Separation - Usage Guide

## Overview

This ablation study evaluates the importance of zone separation in texture synthesis by comparing two methods:
- **Complete Method**: Original synthesis with zone separation (edge, zone0, zone1)
- **Ablated Method**: Simplified synthesis without zone separation (single-pass)

## Files Created

1. **`synthesis_ablation_no_zones.py`** - Ablated synthesis script (no zone separation)
2. **`ablation_compare_no_zones.py`** - Comparison script that runs both methods
3. **`todo/ablation_no_zones_results.md`** - Results documentation template

## Quick Start

### Option 1: Run Comparison (Recommended)

Run both methods and generate comparative analysis:

```bash
python ablation_compare_no_zones.py \
  --sample_path tgs_salt/0bdd44d530.png \
  --sample_semantic_mask_path tgs_salt/0bdd44d530Mask.png \
  --generat_mask_path tgs_salt/0bdd44d530Mask.png \
  --window_height 101 \
  --window_width 101 \
  --kernel_size 11 \
  --iterations 10
```

**Output Structure:**
```
result/ablation_comparison_[sample]_[timestamp]/
├── complete_method/
│   ├── complete_001.jpg
│   ├── complete_002.jpg
│   └── ...
├── ablated_method/
│   ├── ablated_001.jpg
│   ├── ablated_002.jpg
│   └── ...
├── raw_results.csv
└── comparative_statistics.csv
```

### Option 2: Run Ablated Method Only

Run only the ablated version (without zone separation):

```bash
python synthesis_ablation_no_zones.py \
  --sample_path tgs_salt/0bdd44d530.png \
  --generat_mask_path tgs_salt/0bdd44d530Mask.png \
  --window_height 101 \
  --window_width 101 \
  --kernel_size 11 \
  --iterations 50
```

### Option 3: Run Complete Method Only

Run the original method with zone separation:

```bash
python synthesis.py \
  --sample_path tgs_salt/0bdd44d530.png \
  --sample_semantic_mask_path tgs_salt/0bdd44d530Mask.png \
  --generat_mask_path tgs_salt/0bdd44d530Mask.png \
  --window_height 101 \
  --window_width 101 \
  --kernel_size 11 \
  --iterations 50 \
  --patches_cache_path result/patches_db_cache.npz
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--sample_path` | Path to input texture sample | Required |
| `--sample_semantic_mask_path` | Path to semantic mask (complete method only) | Required for complete |
| `--generat_mask_path` | Path to generation mask | Required |
| `--window_height` | Height of synthesis window | 50 |
| `--window_width` | Width of synthesis window | 50 |
| `--kernel_size` | Size of synthesis kernel (must be odd) | 11 |
| `--iterations` | Number of synthesis iterations | 50 (10 for comparison) |
| `--visualize` | Show synthesis process (slow) | False |
| `--output_dir` | Custom output directory (comparison only) | Auto-generated |

## Understanding the Results

### Metrics Explained

1. **MSE (Mean Squared Error)**
   - Measures pixel-level differences
   - Lower = better quality
   - Expected: Ablated method should have higher MSE

2. **DSSIM (Structural Dissimilarity)**
   - Measures structural differences
   - Lower = better structural similarity
   - Expected: Ablated method should have higher DSSIM

3. **LBP Distance (Local Binary Pattern)**
   - Measures texture pattern similarity
   - Lower = better texture matching
   - Expected: Ablated method should have higher LBP distance

### Interpreting Results

**If hypothesis is correct:**
- Ablated method should show degraded metrics (higher MSE, DSSIM, LBP)
- Visual inspection should reveal blurred boundaries
- Geological coherence should be reduced

**If hypothesis is incorrect:**
- Similar or better metrics suggest zone separation may not be critical
- May indicate that basic texture synthesis is sufficient

## Workflow

1. **Run Comparison**
   ```bash
   python ablation_compare_no_zones.py --sample_path [path] --sample_semantic_mask_path [path] --generat_mask_path [path] --iterations 10
   ```

2. **Review Quantitative Results**
   - Open `comparative_statistics.csv`
   - Check percentage differences between methods
   - Look for significant degradation in ablated method

3. **Visual Inspection**
   - Compare images in `complete_method/` vs `ablated_method/`
   - Focus on boundary regions between salt and rock
   - Look for blurring, artifacts, or loss of coherence

4. **Document Findings**
   - Fill in `todo/ablation_no_zones_results.md`
   - Add quantitative data from CSV files
   - Include visual comparisons (screenshots or image embeds)
   - Write conclusions about zone separation importance

5. **Generate Visualizations (Optional)**
   - Use `Boxplots.ipynb` to create boxplot comparisons
   - Create side-by-side image comparisons
   - Generate statistical plots

## Tips

- **Start Small**: Run with `--iterations 3` first to verify everything works
- **Use Cache**: The complete method uses patches DB cache for speed
- **Visualize Sparingly**: `--visualize` is very slow, use only for debugging
- **Multiple Samples**: Run on different samples to ensure results are consistent
- **Document Everything**: Fill in the results template as you go

## Troubleshooting

**Error: "Unable to read image from sample_path"**
- Check that file paths are correct
- Use absolute paths if relative paths fail

**Error: "Patches DB loading failed"**
- Run with `--rebuild_patches_db` flag (complete method only)
- Check that dataset paths in `suport/locals.py` are configured

**Slow Performance**
- Reduce `--iterations` for testing
- Increase `--kernel_size` slightly (makes synthesis faster but less detailed)
- Reduce `--window_height` and `--window_width`

**Out of Memory**
- Reduce window size
- Process fewer iterations at once
- Close other applications

## Next Steps

After completing this ablation study, consider:

1. **Ablation Study 2**: Remove VAE mask generation (use geometric transforms)
2. **Ablation Study 3**: Remove angle-oriented boundary synthesis
3. **Statistical Analysis**: Run multiple samples and aggregate results
4. **Expert Evaluation**: Get qualitative feedback from domain experts
5. **Publication**: Include results in paper's ablation study section

## References

- Original paper: Efros & Leung (1999) - Texture Synthesis by Non-parametric Sampling
- Implementation: `synthesis.py` (complete method)
- Ablated version: `synthesis_ablation_no_zones.py`
- Comparison tool: `ablation_compare_no_zones.py`
