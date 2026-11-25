# EMDUL

## Dependency Installation
1. Create a conda environment and activate it:
    ```
    conda create -n emdul python=3.12
    conda activate emdul
    ```
2. Install Lightning:
    ```
    conda install lightning==2.5.5 -c conda-forge
    ```

2. Install dependencies in `requirements.txt`:
    ```
    pip install -r requirements.txt
    ```
3. Install pointnet2 for P4Transformer:
    ```
    cd model/P4Transformer
    python setup.py install
    ```

## Data Preprocessing
To be completed

## Training and Testing
Training EMDUL/P4Transformer on MM-Fi + HmPEAR gave a mean mmBody MPJPE of 15.47 cm across three training seeds (sample SD 0.68 cm). The closest run to the paper's 14.89 cm reached 14.97 cm. The [reproduction report](MMFI_HMPEAR_REPRODUCTION.md) lists all three results, training and evaluation commands, and local checkpoint paths.

### Training
```
python main.py -g 2 -n 1 -w 8 -b 64 -e 64 --exp_name mmfi_p4t -c cfg/base/mmfi.yml --version $(date +'%Y%m%d_%H%M%S')
```
### Testing
```
python main.py -g 2 -n 1 -w 8 -b 64 -e 64 --exp_name mmfi_p4t -c cfg/base/mmfi.yml --version $(date +'%Y%m%d_%H%M%S') --checkpoint_path path_to_ckpt.ckpt --test
```
