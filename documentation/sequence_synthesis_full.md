## Sequence Diagram - Full Synthesis Run (Loop + Metrics)

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant M as main()
    participant Arg as argparse
    participant CV as OpenCV(cv2)
    participant Syn as synthesize()
    participant PM as patchesMethods (pm)
    participant FS as fillSample()
    participant Met as analizeMetrics()
    participant OS as OS/FS

    U->>M: python synthesis.py --sample_path ...
    M->>Arg: parse_args()
    Arg-->>M: args
    M->>CV: imread(sample_path)
    CV-->>M: sample (BGR)
    alt semantic mask provided
        M->>CV: imread(semantic_mask)
        CV-->>M: semantic_mask (BGR)
        M->>CV: cvtColor -> grayscale
    else none
        M->>M: semantic_mask = sample
    end
    alt generat_mask provided
        M->>CV: imread(generat_mask)
        CV-->>M: generat_mask (BGR)
        M->>CV: cvtColor -> grayscale
    else none
        M->>M: generat_mask = ones(window_size)
    end
    M->>M: validate_args()
    M->>OS: create run_dir result/run_<sample_name>
    loop n iterations (benchmark)
        M->>Syn: synthesize(sample, semantic_mask, generat_mask,...)
        Syn->>PM: probHough(generat_mask)
        PM-->>Syn: segments
        Syn->>PM: loadDataBase()
        PM-->>Syn: samplesPatchesDB
        Syn->>PM: create_Masks(generat_mask)
        PM-->>Syn: dilated_edge, zone0, zone1, fullmask
        loop edge segments
            Syn->>PM: searchNearestPatchByAngleAndHistogram(angle, currentSample)
            PM-->>Syn: matchedSample
            Syn->>PM: makePatchMask(...)
            PM-->>Syn: patchMask
            alt first segment
                Syn->>Syn: initialize(matchedSample)
            else subsequent segment
                Syn->>Syn: update(matchedSample)
            end
            Syn->>FS: fillSample() frontier growth
            FS-->>Syn: updated resultRGBW
        end
        Syn->>FS: fillSample() complete edge zone
        Syn->>PM: sampleBreak(original_sample, semantic_mask)
        PM-->>Syn: sampleEdge, sampleZ0, sampleZ1
        Syn->>Syn: extractBiggestSquare(sampleZ1)
        Syn->>FS: fillSample() zone1
        Syn->>Syn: extractBiggestSquare(sampleZ0)
        Syn->>FS: fillSample() zone0
        Syn->>FS: fillSample() fullmask remainder
        Syn-->>M: synthesized_texture
        M->>CV: imwrite(run_dir/random.jpg)
        M->>CV: cvtColor(original->gray)
        M->>CV: cvtColor(result->gray)
        M->>Met: analizeMetrics(gray_original, gray_result)
        Met-->>M: mse, lbp_distance, dssim
        M->>M: append metrics row
    end
    alt exception during loop
        M->>M: catch + record partial durations
        M->>M: traceback.print_exc()
    end
    M->>OS: write run_metrics_<uuid>.csv (header + per-iteration rows)
    M-->>U: path to metrics CSV + summary (mean, std)
```

### Notes
- `fillSample()` embodies the pixel growth using weighted normalized SSD and Gaussian kernel.
- Zones: edge (dilated), zone1, zone0, then completion (fullmask).
- Metrics are computed per iteration in grayscale for comparability.
- CSV header lines starting with `#` store run metadata.

### Potential Extensions
- Add early stopping if coverage ratio > threshold.
- Parameterize iteration count via CLI (e.g., `--repeat`).
- Optional disable per-iteration metrics for speed.
