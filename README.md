# ZMPY3D_CP

**Update:**
ZMPY3D PyTorch implementation is available (August 25, 2024).

ZMPY3D: accelerating protein structure volume analysis through vectorized 3D Zernike Moments and Python-based GPU Integration

For CPU support only, please refer to the repository:

`ZMPY3D` supports `NumPy`
(https://github.com/tawssie/ZMPY3D)

For GPU support with TensorFlow, CuPy and PyTorch, please refer to the other three repositories:

`ZMPY3D_TF` supports `Tensorflow`
(https://github.com/tawssie/ZMPY3D_TF)

`ZMPY3D_CP` supports `CuPy`
(https://github.com/tawssie/ZMPY3D_CP)

`ZMPY3D_PT` supports `PyTorch`
(https://github.com/tawssie/ZMPY3D_PT)

Here presents a Python-based software package, ZMPY3D, to accelerate the moments computation by vectorizing the mathematical formulae, enabling their computation in graphical processing units (GPUs). The package offers popular GPU-supported libraries such as CuPy and TensorFlow along with NumPy implementations, aiming to improve computational efficiency, adaptability, and flexibility in future algorithmic development. 

## Installation

**Noted:**
ZMPY3D_CP supports cupy-cuda11x (>=12.2.0) or cupy-cuda12x (>=12.2.0).
ZMPY3D_CP does not auto-install CuPy to avoid conflicts.
Users should verify the installation with `pip list | grep cupy-cuda` and manually [install one version of CuPy](https://docs.cupy.dev/en/stable/install.html).

**Prerequisites:**
* ZMPY3D   : Python >=3.9.16, NumPy >=1.23.5
* ZMPY3D_CP: Python >=3.9.16, NumPy, CuPy >=12.2.0
* ZMPY3D_TF: Python >=3.9.16, NumPy >=1.23.5, Tensorflow >=2.12.0, Tensorflow-Probability >=0.20.1
* ZMPY3D_PT: Python >=3.9.16, NumPy >=1.23.5, PyTorch >= 2.3.1

1. Open the terminal
2. Using pip to install the package through PyPI
3. Run `pip install ZMPY3D_CP` for the installation

## Usage
* 3D Zernike moments with Tensorflow: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tawssie/ZMPY3D/blob/main/ZMPY3D_demo_zm.ipynb)
* Shape similarity with CuPy: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tawssie/ZMPY3D/blob/main/ZMPY3D_demo_shape.ipynb) 
* Structure superposition with NumPy: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tawssie/ZMPY3D/blob/main/ZMPY3D_demo_super.ipynb)
* Runtime evaluation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tawssie/ZMPY3D/blob/main/ZMPY3D_time_evaluation.ipynb) 

## Performances

A voxel cube with dimensions of 100x100x100 was applied to perform 10,000 3D Zernike moment calculations, using 2 different maximum orders 20 and 40.
Execution times for different hardware configurations using TensorFlow, CuPy, and NumPy libraries:

### NumPy

| Order | CPU1       | CPU2       |
|-------|------------|------------|
| 20    | 33m20s     | 14m1s      |
| 40    | 951m40s    | 338m20s    |


### TensorFlow

| Order |            T4 |            RX3070Ti |            V100 |            L4 | 
|-------|---------------|---------------------|-----------------|---------------|
| 20    | 1m1s          | 0m36s               | 0m31s           | 0m39s         | 
| 40    | 24m40s        | 9m3s                | 10m54s          | 11m13s        | 

### CuPy
| Order |      T4 |      RX3070Ti |      V100 |      L4 |
|-------|---------|---------------|-----------|---------|
| 20    | 4m45s   | 2m30s         | 1m42s     | 2m50s   |
| 40    | 35m20s  | 19m19s        | 14m45s    | 18m40s  |

Note: m = minutes, s = seconds.

## Cache data for order 40

Due to GitHub's file size limitations, follow these steps to download the cache data for order 40 (1.3G) in the ZMPY3D_CP package:

### 1. Locate Package Folder

- Open your terminal and execute the following command to find the folder of the ZMPY3D_CP package:
- `python -c "import ZMPY3D_CP; print(ZMPY3D_CP.__file__)"`
- Note the path, which ends with `/User/path/ptyhon/site-packages/ZMPY3D_CP/__init__.py`.

### 2. Navigate to Cache Data Folder
- Go to the `cache_data` folder at the same level as `__init__.py` file, i.e., `/User/path/ptyhon/site-packages/ZMPY3D_CP/cache_data`.

### 3. Download the Cache File:
- Download the 1.3 GB max order 40 `.pkl` file to the `cache_data` folder from the link below. https://drive.google.com/uc?id=1RR1rF_5YJqaxNC5AK0Ie_8MswGb0Tttw


## Further reading: What can 3D Zernike moments do?
- Enhancing fold classification
  * [Real-time structure search and structure classification for AlphaFold protein models](https://doi.org/10.1038/s42003-022-03261-8)
  * [Real time structural search of the Protein Data Bank](https://doi.org/10.1371/journal.pcbi.1007970)
- Facilitating structural superpositions
  * [ZEAL: Protein structure alignment based on shape similarity](https://doi.org/10.1093/bioinformatics/btab205)
- Supporting protein docking
  * [Protein-protein docking using region-based 3D Zernike descriptors](https://doi.org/10.1186/1471-2105-10-407)
- Assisting molecular dynamics
  * [Binding site identification of G protein-coupled receptors through a 3D Zernike polynomials-based method: application to C. elegans olfactory receptors](https://doi.org/10.1007/s10822-021-00434-1)
  * [Quantitative characterization of binding pockets and binding complementarity by means of zernike descriptors](https://doi.org/10.1021/acs.jcim.9b01066)
- Enabling structure-based virtual screening
  * [PL-PatchSurfer3: Improved Structure-Based Virtual Screening for Structure Variation Using 3D Zernike Descriptors](https://doi.org/10.1101/2024.02.22.581511)
- Forecasting interacting interfaces
  * [Antibody interface prediction with 3D Zernike descriptors and SVM](https://doi.org/10.1093/bioinformatics/bty918)
  * [Exploring the potential of 3D Zernike descriptors and SVM for protein-protein interface prediction](https://doi.org/10.1186/s12859-018-2043-3)

## Contributing

Feel free to submit pull requests for improvements or bug fixes.

************************* 


## Citation

Lai, J. S., Burley, S. K., & Duarte, J. M. (2024). ZMPY3D: Accelerating protein structure volume analysis through vectorized 3D Zernike moments and Python-based GPU integration. (Bioinformatics Advances, vbae111, https://doi.org/10.1093/bioadv/vbae111)

## License

This project is licensed under the GNU General Public License v3.0. You can view the full license [here](https://www.gnu.org/licenses/gpl-3.0.en.html).

