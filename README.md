# Installation

Install with `pip install .` for deployment and with `pip install -e .` for development mode.

Run tests with `bash run_tests.sh`.

# Running Examples

### Step 1: Download the Example Data

First, download the example data from the following link:

[Download Example Data](https://drive.google.com/file/d/1m67Lw0nWF8raE3NDQ8-jiecZkGB0mYoA/view?usp=drive_link) (20MB)

### Step 2: Untar the Example Data

Untar the example data in the "example_scripts" directory

```bash
cd example_scripts
tar -xzvf nvtorchcam_example_data.tar.gz
```

### Step 3: List or run examples

To list available examples:

```bash 
python example_scripts/examples.py
```
Expected output

```
Available example names:
save_pointclouds
warp_to_other_models
backward_warp
mvsnet_fusion
stereo_rectify
backward_warp_sphere
```

To run examples:
```
python example_scripts/examples.py <example_name>
```

Example results will be in directory "examples_output"

# Dependencies
Tested with

- torch                     2.0.1

- plyfile                   1.0 (examples only)

- imageio                   2.31.1 (examples only)

- opencv-python             4.8.0.74 (examples and tests)