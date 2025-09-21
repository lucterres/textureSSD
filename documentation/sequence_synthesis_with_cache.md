## Sequence Diagram - Synthesis Run With Patch DB Caching (Single DB Load)

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant M as main()
    participant Arg as argparse
    participant CV as OpenCV
    participant PM as patchesMethods
    participant Syn as synthesize()
    participant FS as fillSample()
    participant Met as analizeMetrics()
    participant OS as OS/FS

    U->>M: python synthesis.py --sample_path ... --patches_cache_path ... --iterations N
    M->>Arg: parse_args()
    Arg-->>M: args
    M->>CV: imread(sample_path)
    CV-->>M: sample(BGR)
    alt semantic mask provided
        M->>CV: imread(semantic_mask)
        CV-->>M: semantic_mask(BGR)
        M->>CV: cvtColor -> gray
    else none
        M->>M: semantic_mask = sample
    end
    alt generat_mask provided
        M->>CV: imread(generat_mask)
        CV-->>M: generat_mask(BGR)
        M->>CV: cvtColor -> gray
    else none
        M->>M: generat_mask = ones(window_size)
    end
    M->>M: validate_args()
    M->>OS: mkdir result/run_<sample>
    note over M: Prepare synthesis loop metadata

    rect rgb(235,245,255)
    M->>PM: loadDataBase(cache_path, rebuild?)
    alt cache exists & not rebuild
        PM-->>M: load patches from .npz
    else need rebuild
        PM->>PM: build patches (Hough, extract, serialize)
        PM-->>M: patches_db (list[Patch])
        PM->>OS: save .npz (compressed)
    end
    note over PM,M: Patches DB reused for all iterations
    end

    loop iterations (i = 1..N)
        M->>Syn: synthesize(sample, semantic_mask, generat_mask, patches_db)
        Syn->>PM: probHough(generat_mask)
        PM-->>Syn: segments
        Syn->>PM: create_Masks(generat_mask)
        PM-->>Syn: dilated_edge, zone0, zone1, fullmask
        loop edge segments
            Syn->>PM: searchNearestPatchByAngleAndHistogram(angle, lastSample)
            PM-->>Syn: matchedSample
            Syn->>PM: makePatchMask(...)
            PM-->>Syn: patchMask
            alt first segment
                Syn->>Syn: initialize(matchedSample)
            else subsequent
                Syn->>Syn: update(matchedSample)
            end
            Syn->>FS: fillSample() pixel growth
            FS-->>Syn: resultRGBW (partial)
        end
        Syn->>FS: fillSample() complete edge
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
        Met-->>M: mse, lbp_dist, dssim
        M->>M: append metrics row
    end
    alt exception
        M->>M: log traceback; continue to finalize
    end
    M->>OS: write run_metrics_<uuid>.csv
    M-->>U: metrics CSV path & summary
```

### Differences vs Full Diagram
- Patch database loaded **once** before the iteration loop.
- Rebuild only triggered if `--rebuild_patches_db` or cache missing.
- Eliminated redundant per-iteration `loadDataBase()` call.

### Future Enhancements
- Add cache signature (hash of dataset + code) to auto-invalidate.
- Optional flag to skip per-iteration metrics for speed (`--no-metrics`).
- Parallelize angle segment processing where safe.
