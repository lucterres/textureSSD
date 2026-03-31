sequenceDiagram
    participant User
    participant main as main()
    participant synthesize as synthesize()
    participant pm as patchesMethods
    participant cv2 as OpenCV
    participant np as NumPy

    User->>main: Executes `python synthesis.py` with arguments
    main->>main: Parses arguments
    main->>cv2: Reads sample, semantic, and generation masks
    main->>synthesize: Calls with images and parameters

    synthesize->>pm: `probHough()` to find line segments in the generation mask
    pm-->>synthesize: Returns segments (patches)

    note over synthesize: Step 1: Generate Edge Zone by iterating through segments
    loop For each segment
        synthesize->>pm: `searchNearestPatchByAngleAndHistogram()` to find the best sample patch
        pm-->>synthesize: Returns best patch
        synthesize->>pm: `makePatchMask()` to isolate the current patch area
        pm-->>synthesize: Returns patch mask
        alt First segment
            synthesize->>synthesize: `initialize()` to create windows, masks, and place the initial seed
        else Not the first segment
            synthesize->>synthesize: `update()` to get new grayscale sample
        end
        synthesize->>synthesize: `fillSample()` to synthesize the texture for the current patch
    end

    note over synthesize: Complete the rest of the edge zone
    synthesize->>synthesize: `fillSample()`

    synthesize->>pm: `sampleBreak()` to get different zones from the original sample
    pm-->>synthesize: Returns edge, zone0, and zone1 samples

    note over synthesize: Step 2: Generate Zone 1
    synthesize->>synthesize: `extractBiggestSquare()` for zone1 sample
    synthesize->>synthesize: `update()` with the new sample
    synthesize->>synthesize: `fillSample()` to synthesize zone1

    note over synthesize: Step 3: Generate Zone 0
    synthesize->>synthesize: `extractBiggestSquare()` for zone0 sample
    synthesize->>synthesize: `update()` with the new sample
    synthesize->>synthesize: `fillSample()` to synthesize zone0

    note over synthesize: Step 4: Complete the full mask
    synthesize->>synthesize: `fillSample()` to fill any remaining pixels

    synthesize-->>main: Returns the synthesized texture

    main->>cv2: Saves the synthesized texture
    main->>main: `analizeMetrics()` to calculate MSE, DSSIM, LBP
    main->>main: Saves metrics to a CSV file