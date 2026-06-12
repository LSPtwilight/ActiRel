### 1. Installation for G4splat Framework

<details>
<summary><span style="font-weight: bold;">Click here to see content.</span></summary>

#### 1.1. Install dependencies

Please follow the instructions below to install the dependencies manually:

```shell
conda create --name priorgs -y python=3.9
conda activate actirel

#### Choose the right pytorch version for your system ##############
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install faiss-gpu-cu11
###################################################################

conda install -c fvcore -c iopath -c conda-forge fvcore iopath \
    pytorch3d==0.7.4 -c pytorch3d \
    -c plotly plotly \
    -c conda-forge rich plyfile==0.8.1 jupyterlab nodejs ipywidgets \
    cmake \
    conda-forge::gmp \
    conda-forge::cgal


pip install roma==1.5.0 open3d==0.18.0 opencv-python==4.11.0.86 \
    scipy==1.13.1 einops==0.8.1 trimesh==4.6.4 pyglet==1.5.29 \
    tensorboard scikit-learn==1.6.1 cython==3.0.12 tqdm==4.67.1 \
    matplotlib==3.9.4 huggingface-hub==0.22.2 gradio kiui mediapy \
    diffusers==0.19.3 accelerate transformers==4.28.1 xformers==0.0.20 \
    pytransform3d imageio

# PromDA
git clone https://github.com/DepthAnything/PromptDA.git
cd PromptDA && pip install -e .

# sam
pip install git+https://github.com/facebookresearch/segment-anything.git

# just for visualization, Not Necessary
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Then, install the 2D Gaussian splatting and adaptive tetrahedralization dependencies:

```shell
cd 2d-gaussian-splatting/submodules/diff-surfel-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
cd ../tetra-triangulation
cmake .
# you can specify your own cuda path
export CPATH=/usr/local/cuda-11.8/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
make 
pip install -e .
cd ../../../
```

Finally, install the MASt3R-SfM dependencies:

```shell
cd mast3r/asmk/cython
cythonize *.pyx
cd ..
pip install .
cd ../dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../
```


#### 1.2. Download pretrained models

Start by downloading a pretrained checkpoint for DepthAnythingV2. Several encoder sizes are available; We recommend using the `large` encoder:

```shell
mkdir -p ./Depth-Anything-V2/checkpoints/
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -P ./Depth-Anything-V2/checkpoints/
```

Then, download the MASt3R-SfM checkpoint:

```shell
mkdir -p ./mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P ./mast3r/checkpoints/
```

And finally, download the MASt3R-SfM retrieval checkpoint:

```shell
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P ./mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P ./mast3r/checkpoints/
```

</details>

### 2.data
#### 2.1. Installation
```
# Fast Install
https://huggingface.co/datasets/JunfengNi/G4Splat

# Official Install
https://github.com/facebookresearch/replica-dataset
https://github.com/scannetpp/scannetpp

Then you need to change cam-pos and mesh to Z-up.
```

#### 2.2. Processing
```
└── G4Splat
  └── data
    ├── replica
        ├── scan1 
        ├── scan2 ...
    ├── scannetpp
        ├── scan1 
        ├── scan2 ...

```

### 3. Running
```
sh train.sh
```
If you want to change the experiment parameters, you can modify them in train.sh.

