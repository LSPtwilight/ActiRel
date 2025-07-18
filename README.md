### 1. Detailed installation (if quick install fails)

<details>
<summary><span style="font-weight: bold;">Click here to see content.</span></summary>

#### 1.1. Install dependencies

Please follow the instructions below to install the dependencies manually:

```shell
conda create --name priorgs -y python=3.9
conda activate priorgs
# Choose the right CUDA version for your system
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d==0.7.4 -c pytorch3d
conda install -c plotly plotly
conda install -c conda-forge rich
conda install -c conda-forge plyfile==0.8.1
conda install -c conda-forge jupyterlab
conda install -c conda-forge nodejs
conda install -c conda-forge ipywidgets
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal

pip install ~/src/torch/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl
pip install ~/src/torchvision/torchvision-0.15.2+cu118-cp39-cp39-linux_x86_64.whl
cp ~/src/pytorch3d/pytorch3d.zip ./ && unzip pytorch3d.zip
cd pytorch3d && pip install -e .
cd PromptDA && pip install -e .

pip install roma==1.5.0
pip install open3d==0.18.0
pip install opencv-python==4.11.0.86
pip install scipy==1.13.1
pip install einops==0.8.1
pip install trimesh==4.6.4
pip install pyglet==1.5.29
pip install tensorboard
pip install scikit-learn==1.6.1
pip install cython==3.0.12
# Choose the right CUDA version for your system
pip install faiss-gpu-cu11
pip install tqdm==4.67.1
pip install matplotlib==3.9.4
pip install huggingface-hub==0.22.2
pip install gradio
pip install kiui
pip install mediapy
pip install diffusers==0.19.3
pip install accelerate
pip install transformers==4.28.1
pip install xformers==0.0.20
pip install pytransform3d
pip install imageio

# sam
pip install git+https://github.com/facebookresearch/segment-anything.git

# detectron2 just for visualization
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
#### 2.1. Replica (from DP-Recon)
```
https://drive.google.com/drive/folders/1VUo6egav6j4aBw_ADRFP4NEE8pgB4alq
```

#### 2.2. Mip-NeRF 360 (from ReconFusion)
```
https://drive.google.com/drive/folders/10oT2_OQ9Sjh5wlfJQoGx2y7ZKYwpgNg5
```
