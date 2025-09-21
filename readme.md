# Texture Synthesis by Non-parametric Sampling

This project implements Efros and Leung's 'Texture Synthesis by Non-parametric Sampling' (1999).

The algorithm synthesizes a new texture from an existing input by sampling the center pixel from a local neighborhood of the sample that best "matches" a local neighborhood in the current state of the synthesized texture.

## Usage

`python synthesis.py --sample_path=<input_path> --out_path=[output_path] --window_height=[win_height] --window_width=[win_width] --kernel_size=[ksize] --iterations=[n] --visualize`

* `input_path` - Path to the input texture sample.
* `output_path` - (Optional) Output path for the synthesized texture.
* `win_height` - (Optional) Height of the synthesized texture. 50 pixels is the default.
* `win_width` - (Optional) Width of the synthesized texture. 50 pixels is the default.
* `ksize` - (Optional) Width of a square synthesis kernel. Each synthesized pixel value is selected by computing a distance metric between local neighborhoods of height and width `ksize` between the current state of the synthesized texture and the input texture sample. 11 pixels is the default, but this value can be raised (lowered) to increase (decrease) the regularity of the synthesized texture.
* `iterations` - (Optional) Number of full synthesis repetitions to perform (each run produces an output image & metrics). Default: 50.
* `visualize` - (Optional) Visualize an in-progress texture synthesis.

## Dependencies

* Python - Tested on version 3.7.0
* OpenCV - Tested on version 3.4.1
* NumPy - Tested on version 1.15.0

## Results

The following are selected results of this procedure. The leftmost column contains input texture samples and the rightmost column contains output synthesized textures. The middle column shows the completion of "layers" of the synthesis process.

| <p align="center"> Input Sample </p> | <p align="center"> Synthesis Process </p> | <p align="center"> Output Texture </p>
| ------------ | ------------- | --------------
| <p align="center">![161 Input](examples/161.jpg)  </p> | <p align="center"> ![161 Synthesis](examples/161.gif) </p> | <p align="center"> ![161 Output](examples/161_out.jpg) </p>
| <p align="center">![D3 Input](examples/D3.jpg) </p> | <p align="center"> ![D3 Synthesis](examples/D3.gif) </p> | <p align="center"> ![D3 Output](examples/D3_out.jpg) </p>
| <p align="center">![Wood Input](examples/wood.jpg) </p> | <p align="center"> ![Wood Synthesis](examples/wood.gif) </p> | <p align="center"> ![Wood Output](examples/wood_out.jpg) </p>

## Patch Database Caching (Performance)

Building the internal patch database (Hough line extraction + per‑patch feature prep) can be expensive. To speed up repeated runs you can persist this database to disk and reuse it.

### Flags

* `--patches_cache_path <path>`  Path to a compressed NumPy cache (default: `result/patches_db_cache.npz`).
	* Use `--patches_cache_path none` to disable caching explicitly.
* `--rebuild_patches_db`  Force a rebuild even if the cache file already exists (the new database overwrites the file).

If no flag is given the code behaves as before (builds in memory only, unless a default was wired in your run configuration).

### Typical Workflows

1. First (slow) run, create cache:
```
python synthesis.py --sample_path tgs_salt/0bdd44d530.png \
	--sample_semantic_mask_path tgs_salt/0bdd44d530Mask.png \
	--generat_mask_path tgs_salt/0bdd44d530Mask.png \
	--window_height 101 --window_width 101 --kernel_size 11 \
	--patches_cache_path result/patches_db_cache.npz
```
2. Subsequent (fast) runs reuse cache automatically with the same command.
3. Rebuild after changing dataset / patch construction logic:
```
python synthesis.py ... --patches_cache_path result/patches_db_cache.npz --rebuild_patches_db
```
4. Disable caching (diagnostics / memory tests):
```
python synthesis.py ... --patches_cache_path none
```

### When Should I Rebuild?
Rebuild if any of these changed since the cache was created:
* Source image / mask dataset contents.
* Patch extraction logic (e.g. edits in `suport/patchesMethods.py`).
* Parameters that influence which patches are collected (e.g. Hough thresholds, line filtering rules) if you integrate them later as arguments.
* Upgrading major NumPy / Python versions (rarely necessary, but safe to rebuild if deserialization fails).

### What Gets Stored
The cache is a `.npz` file containing a serialized list of patch dictionaries with:
* `line` (int32 array of line pixels)
* `angle` (float / orientation)
* `image` (the patch image data)

At load time these are rehydrated into Patch objects identical to those created during an in-memory build.

### Troubleshooting
* Cache file not found: It will be created on first run (ensure parent directory `result/` exists).
* Corrupted / partial cache (e.g. interrupted write): delete the file and rebuild.
* Unexpected results after code change: run with `--rebuild_patches_db` to refresh.
* High RAM usage: try disabling cache (`--patches_cache_path none`) to confirm, then consider slimming patch collection logic.
* Shape mismatch errors while loading: indicates structural change in Patch layout—delete cache and rebuild.

### VS Code Launch Configurations
If you use VS Code, several ready-made run profiles can be added (example names):
* `Synthesis (cache padrão)` – uses / creates the cache
* `Synthesis (forçar rebuild cache)` – forces a rebuild
* `Synthesis (sem cache)` – disables cache

These simply toggle the two flags above; adjust paths as needed for your dataset.

---

### Diagrams

Sequence diagrams of the synthesis pipeline:
* [Full per-iteration version](documentation/sequence_synthesis_full.md)
* [Cached patch DB version](documentation/sequence_synthesis_with_cache.md)