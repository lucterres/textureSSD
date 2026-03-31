# Ablation Study: No Zone Separation - Results

**Experiment Date**: _[To be filled]_  
**Sample Image**: _[To be filled]_  
**Iterations**: _[To be filled]_

## Experiment Parameters

| Parameter | Value |
|-----------|-------|
| Window Size | _[height x width]_ |
| Kernel Size | _[size]_ |
| Iterations per Method | _[n]_ |
| Sample Path | _[path]_ |
| Semantic Mask Path | _[path]_ |
| Generation Mask Path | _[path]_ |

## Objective

Evaluate the importance of dividing the image into distinct zones (salt, conventional rock, and boundary) by comparing:
- **Complete Method**: Synthesis with zone separation (edge, zone0, zone1)
- **Ablated Method**: Synthesis without zone separation (single-pass using complete image)

## Hypothesis

Removing zone separation should result in:
- Degraded image quality, especially at boundaries
- Blurred or unrealistic transitions between salt and rock
- Higher error metrics (MSE, DSSIM, LBP Distance)
- Loss of geological coherence

## Quantitative Results

### Processing Time

| Method | Mean (s) | Std Dev (s) | Min (s) | Max (s) |
|--------|----------|-------------|---------|---------|
| Complete | _[value]_ | _[value]_ | _[value]_ | _[value]_ |
| Ablated | _[value]_ | _[value]_ | _[value]_ | _[value]_ |
| **Difference** | _[%]_ | - | - | - |

### Mean Squared Error (MSE)

| Method | Mean | Std Dev | Median | Q25 | Q75 |
|--------|------|---------|--------|-----|-----|
| Complete | _[value]_ | _[value]_ | _[value]_ | _[value]_ | _[value]_ |
| Ablated | _[value]_ | _[value]_ | _[value]_ | _[value]_ | _[value]_ |
| **Difference** | _[%]_ | - | - | - | - |

> [!NOTE]
> Lower MSE indicates better quality (less difference from original texture characteristics)

### Structural Dissimilarity (DSSIM)

| Method | Mean | Std Dev | Median | Q25 | Q75 |
|--------|------|---------|--------|-----|-----|
| Complete | _[value]_ | _[value]_ | _[value]_ | _[value]_ | _[value]_ |
| Ablated | _[value]_ | _[value]_ | _[value]_ | _[value]_ | _[value]_ |
| **Difference** | _[%]_ | - | - | - | - |

> [!NOTE]
> Lower DSSIM indicates better structural similarity to original

### LBP Distance

| Method | Mean | Std Dev | Median | Q25 | Q75 |
|--------|------|---------|--------|-----|-----|
| Complete | _[value]_ | _[value]_ | _[value]_ | _[value]_ | _[value]_ |
| Ablated | _[value]_ | _[value]_ | _[value]_ | _[value]_ | _[value]_ |
| **Difference** | _[%]_ | - | - | - | - |

> [!NOTE]
> Lower LBP Distance indicates better texture pattern similarity

## Qualitative Analysis

### Visual Observations

#### Boundary Quality
_[Describe the quality of transitions between salt and rock regions in both methods]_

#### Texture Coherence
_[Describe how well the texture maintains geological plausibility in both methods]_

#### Artifacts
_[Note any visible artifacts, blurring, or unrealistic patterns]_

### Sample Comparisons

_[Insert side-by-side image comparisons here]_

````carousel
![Complete Method - Sample 1](path/to/complete_001.jpg)
<!-- slide -->
![Ablated Method - Sample 1](path/to/ablated_001.jpg)
<!-- slide -->
![Complete Method - Sample 2](path/to/complete_002.jpg)
<!-- slide -->
![Ablated Method - Sample 2](path/to/ablated_002.jpg)
````

## Conclusions

### Quantitative Findings

_[Summarize the statistical differences between methods]_

> [!IMPORTANT]
> _[Highlight the most significant metric changes]_

### Qualitative Findings

_[Summarize visual quality differences]_

### Hypothesis Validation

_[Confirm or reject the initial hypothesis based on results]_

> [!WARNING]
> _[Note any unexpected findings or limitations]_

### Implications

_[Discuss what these results mean for the importance of zone separation in the methodology]_

## Recommendations

_[Based on results, provide recommendations for:]_
- Whether zone separation is essential for quality synthesis
- Potential improvements to either method
- Future ablation studies to conduct

## References

- Raw data: `result/ablation_comparison_[sample]_[timestamp]/raw_results.csv`
- Statistics: `result/ablation_comparison_[sample]_[timestamp]/comparative_statistics.csv`
- Complete method images: `result/ablation_comparison_[sample]_[timestamp]/complete_method/`
- Ablated method images: `result/ablation_comparison_[sample]_[timestamp]/ablated_method/`
