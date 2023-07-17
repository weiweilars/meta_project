## installation 

#### Pip

```bash
# clone project
git clone https://github.com/weiweilars/meta_project
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### download coco 

download coco dataset and  organized as follow under the root folder with the same name 

coco/annotation 
coco/train2017
coco/val2017

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```


## TOLIST 
** add fewshot learning dataset*
** change dataloader and dataset according to real labeled data
** evaluation step, should share the category support, but right now each iter calculate once ???!!!
** look into accuracy calculation in evaluation step
** Set val.loss to min in callback--earlystopping
** write the test_step
** Set mlflow
** Write the callback functions to output object detect test image
** download coco data to databricks
** training base model on databricks
** training novel model on databricks
