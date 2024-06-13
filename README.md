## Downloading the datasets
To download the datasets, go to the [URL](https://drive.google.com/file/d/1WRO9wO7LbyvTDwoZMGeGPmRS_L-1k108/view?usp=sharing) and download the data. The dataset should be placed in the `data` folder as follows:

```bash
interact
├── data
    ├── behave
    ├── chairs
    ├── grab
    ├── imhd
    ├── intercap
    ├── neuraldome
    └── omomo
```

Ensure each dataset is placed and unzipped in its respective subdirectory within the `data` folder.

For example, for the behave dataset, you should have the following structure:

```bash
interact
├── data
    ├── behave
        ├── objects
        │   ├── backpack
        │   │   ├── backpack.obj
        │   │   ├── sample_points.npy
        ├── objects_bps
        │   ├── backpack
        │   │   ├── backpack.npy
        ├── sequences
        │   ├── Date01_Sub01_backpack_back_0
        │   │   ├── action.npy
        │   │   ├── action.txt
        │   │   ├── human.npz
        │   │   ├── markers.npy
        │   │   ├── object.npz
        │   │   └── text.txt
        ├── sequences_canonical
        │   ├── Date01_Sub01_backpack_back_0
        │   │   ├── action.npy
        │   │   ├── action.txt
        │   │   ├── human.npz
        │   │   ├── markers.npy
        │   │   ├── object.npz
        │   │   └── text.txt
```



## Preparing the environment

```bash
conda env create -f environment.yml
```
<!-- Follow the requirement file [requirements.txt](requirements.txt). -->

## Downloading the model

To download the *SMPL+H* model, go to [their website](http://mano.is.tue.mpg.de) and register to access the downloads section. 

You should download both the [SMPLH with 10 beta components](https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip), as well as the [extended version with 16 beta components](https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz).

To download the *SMPL+X* model, go to [this project website](https://smpl-x.is.tue.mpg.de/) and register to access the downloads section. 

You should download both original version with 10 beta components, as well as the extended version with 16 beta components.

The path to the model should be a directory with the following structure: 
```bash
interact
├── data
    ├── models
        ├── smplh
        │   ├── female
        │   │   ├── model.npz
        │   ├── male
        │   │   ├── model.npz
        │   ├── neutral
        │   │   ├── model.npz
        │   ├── SMPLH_FEMALE.pkl
        │   ├── SMPLH_MALE.pkl
        │   └── SMPLH_NEUTRAL.pkl
        ├── smplx
        │   ├── SMPLX_FEMALE.npz
        │   ├── SMPLX_MALE.npz
        │   ├── SMPLX_NEUTRAL.npz
        │   ├── SMPLX_FEMALE.pkl
        │   ├── SMPLX_MALE.pkl
        │   └── SMPLX_NEUTRAL.pkl
```


## Visualization

To visualize the datasets, you need to go to the visualization folder

```bash
cd visualization
```

If data folder is not present in the InterAct folder, you need to modify the configuration files 'data/config/*.yml' to point to the correct data folder.

To visualize the data, run

```bash
python visualize.py [dataset_name]
```

The dataset_name can be one of the following: behave, neuraldome, intercap, omomo, grab, imhd, or chairs.

<!-- ```bash
python visualize.py neuraldome
```

To visualize all the InterCap dataset, run

```bash
python visualize.py intercap
```

To visualize all the OMOMO dataset, run

```bash
python visualize.py omomo
``` -->

<!-- If data is not in the InterAct folder, you need to modify the line 167 `data_dir = '../data'` in 'visualize_marker.py' to point to the correct path. -->

To visualize markers, run

```bash
python visualize_markers.py
```

<!-- Before training or testing the models, you need to move 'text.txt' files, 'action.txt' files and 'action.npy' files to the respective 'canonical' folders. To do this, you need to run the following command:

```bash
python move_texts.py 
``` -->


You can use our canonicalized sequences named sequences_canonical. Or you can run the following command to directly canonicalize all the datasets:

```bash
bash canonicalize.sh
```

If data folder is not present in the InterAct folder, you need to modify the python files 'canonicalize_human_grab.py', 'canonicalize_human.py' and 'motion_representation.py' to make 'data_root' point to the correct data folder.

## Download the pretrained models
You can download the pretrained models from the following link: [Pretrained Models](https://drive.google.com/file/d/1lWjjiEsTD4cnFmrEzR0eg_RxGCohlouh/view?usp=sharing).

Then, unzip the zip file and place the models in the corresponding folders according to the specified paths.

For example, for text2interaction and action2interaction, you should copy the 'OpenTMA' folder from the weights folder in the google drive to the respective directory. You should also copy the 'save' folder from the weights folder to the respective directory.



## Text to Interaction

To train and test the text to interaction model, you need to go to the text2interaction folder.

```bash
cd text2interaction
```

<details><summary>No BPS, no interaction-aware text encoder</summary>
<p>

To train or test the model, go to the folder.

```bash
cd InterAct-HOI-Diff-woBPS-woTMA
```

<!-- If data folder is not present in the InterAct folder, you need to modify line 60 `opt.data_root = '../data'` in 'InterAct-HOI-Diff-woBPS-woTMA/data_loaders/utils/get_opt.py' file to point to the correct path. -->

To train the model, run the following command:

```bash
bash scripts/run_train_bps.sh
```

To generate examples from the our model, run the following command:


```bash
bash scripts/run_sample_bps.sh
```

To evaluate the model with quantitative results, run the following command:

```bash
bash scripts/run_test_bps.sh
```

You can modify the '--dataset' in 'scripts/run_train_bps.sh', 'scripts/run_sample_bps.sh', 'scripts/run_test_bps.sh' to select the dataset you want to train or test.

The options are 'interact', 'behave', 'chairs', 'grab', 'imhd', 'intercap', 'neuraldome', 'omomo'. 'interact' is the default dataset including all data.


For sampling and testing, make sure the pretrained models are placed at the correct paths.
<!-- For sampling and testing, pretrained model 'model000200000.pt' needs to be presented in the 'InterAct-HOI-Diff-woBPS-woTMA/save/interact_200000' folder. And the model weights 'epoch=799.ckpt' needs to be presented in the 'InterAct-HOI-Diff-wBPS-wTMA/OpenTMA/checkpoints' folder. -->

</p>
</details> 

<details><summary>BPS, no interaction-aware text encoder</summary>
<p>

To train or test the model with BPS and without interaction-aware text encoder, go to the folder.

```bash
cd InterAct-HOI-Diff-wBPS-woTMA
```

<!-- If data folder is not presented in the InterAct folder, you need to modify line 60 `opt.data_root = '../data'` in 'InterAct-HOI-Diff-woBPS-woTMA/data_loaders/utils/get_opt.py' file to point to the correct data folder. -->

To train the model, run the following command:

```bash
bash scripts/run_train_bps.sh
```

<!-- Pretrained model 'model000200000.pt' needs to present in the 'InterAct-HOI-Diff-woBPS-woTMA/save/interact_200000' folder. We can use it to sample examples from the model or test the model. -->

To sample examples from the model, run the following command:

```bash
bash scripts/run_sample_bps.sh
```

<!-- The model weights 'epoch=799.ckpt' for testing need to be present in the 'InterAct-HOI-Diff-wBPS-wTMA/OpenTMA/checkpoints' folder. -->

To test the model, run the following command:

```bash
bash scripts/run_test_bps.sh
```

You can modify '--dataset' in 'scripts/run_train_bps.sh', 'scripts/run_sample_bps.sh', 'scripts/run_test_bps.sh' files to select the dataset you want to train or test. 

The options are 'interact', 'behave', 'chairs', 'grab', 'imhd', 'intercap', 'neuraldome', 'omomo'. 'interact' is the default dataset meaning the model will be trained on the all data.

For sampling and testing, make sure the pretrained models are placed at the correct paths.
</p>
</details> 

<details><summary>BPS, interaction-aware text encoder</summary>
<p>


To train or test the model with BPS and with Interaction-aware text encoder, you need to go to the folder.

```bash
cd InterAct-HOI-Diff-wBPS-wTMA
```

<!-- If data folder is not presented in the InterAct folder, you need to modify line 60 `opt.data_root = '../data'` in 'InterAct-HOI-Diff-woBPS-woTMA/data_loaders/utils/get_opt.py' file to point to the correct data folder. -->

<!-- The model weights 'epoch=999-v2.ckpt' for text embedding need to be present in the 'InterAct-HOI-Diff-wBPS-wTMA/OpenTMA/checkpoints' folder. -->

To train the model, run the following command:

```bash
bash scripts/run_train_bps.sh
```

<!-- Pretrained model 'model000200000.pt' needs to present in the 'InterAct-HOI-Diff-woBPS-woTMA/save/interact_200000' folder. We can use it to sample examples from the model or test the model. -->

To sample examples from the pretrained model, run the following command:

```bash
bash scripts/run_sample_bps.sh
```

<!-- The model weights 'epoch=799.ckpt' for testing need to be present in the 'InterAct-HOI-Diff-wBPS-wTMA/OpenTMA/checkpoints' folder. -->

To test the pretrained model, you need to run the following command:

```bash
bash scripts/run_test_bps.sh
```
You can modify '--dataset' in 'scripts/run_train_bps.sh', 'scripts/run_sample_bps.sh', 'scripts/run_test_bps.sh' files to select the dataset you want to train or test. 

The options are 'interact', 'behave', 'chairs', 'grab', 'imhd', 'intercap', 'neuraldome', 'omomo'. 'interact' is the default dataset meaning the model will be trained on the all data.

For sampling and testing, make sure the pretrained models are placed at the correct paths.
</p>
</details> 

## Action to Interaction

To train and test the action to interaction model, you need to go to the action2interaction folder.

```bash
cd action2interaction
```

<details><summary>one-hot action</summary>
<p>

To train or test the model using one-hot action representation, you need to go to the folder.

```bash
cd InterAct-HOI-Diff-Action-OneHot
```

<!-- If data folder is not present in the InterAct folder, you need to modify line 60 `opt.data_root = '../data'` in 'InterAct-HOI-Diff-Action-OneHot/data_loaders/utils/get_opt.py' file to point to the correct data folder. -->


To train the model, you need to run the following command:

```bash
bash scripts/run_train_bps.sh
```

<!-- Pretrained model 'model000200000.pt' needs to present in the 'InterAct-HOI-Diff-Action-OneHot/save/interact_200000' folder. We can use it to sample examples from the model or test the model. -->

To sample examples from the model, you need to run the following command:

```bash
bash scripts/run_sample_bps.sh
```

<!-- The model weights 'epoch=799.ckpt' for testing need to be present in the 'InterAct-HOI-Diff-wBPS-wTMA/OpenTMA/checkpoints' folder. -->

To test the model, you need to run the following command:

```bash
bash scripts/run_test_bps.sh
```

You can modify '--dataset' in 'scripts/run_train_bps.sh', 'scripts/run_sample_bps.sh', 'scripts/run_test_bps.sh' files to select the dataset you want to train or test. 

The options are 'interact', 'behave', 'chairs', 'grab', 'imhd', 'intercap', 'neuraldome', 'omomo'. 'interact' is the default dataset meaning the model will be trained on the all data.

For sampling and testing, make sure the pretrained models are placed at the correct paths.
</p>
</details> 

<details><summary>interaction-aware text embedding</summary>
<p>

To train or test the model with interaction-aware text embedding, you need to go to the folder.

```bash
cd InterAct-HOI-Diff-Action-TMA
```

<!-- If data folder is not present in the InterAct folder, you need to modify line 60 `opt.data_root = '../data'` in 'InterAct-HOI-Diff-woBPS-woTMA/data_loaders/utils/get_opt.py' file to point to the correct data folder.

The model weights 'epoch=999-v2.ckpt' for text embedding need to be present in the 'InterAct-HOI-Diff-wBPS-wTMA/OpenTMA/checkpoints' folder. -->

To train the model, you need to run the following command:

```bash
bash scripts/run_train_bps.sh
```

<!-- Pretrained model 'model000200000.pt' needs to present in the 'InterAct-HOI-Diff-woBPS-woTMA/save/interact_200000' folder. We can use it to sample examples from the model or test the model. -->


To sample examples from the model, you need to run the following command:

```bash
bash scripts/run_sample_bps.sh
```

<!-- The model weights 'epoch=799.ckpt' for testing need to be present in the 'InterAct-HOI-Diff-wBPS-wTMA/OpenTMA/checkpoints' folder. -->

To test the model, you need to run the following command:

```bash
bash scripts/run_test_bps.sh
```

You can modify '--dataset' in 'scripts/run_train_bps.sh', 'scripts/run_sample_bps.sh', 'scripts/run_test_bps.sh' files to select the dataset you want to train or test. 

The options are 'interact', 'behave', 'chairs', 'grab', 'imhd', 'intercap', 'neuraldome', 'omomo'. 'interact' is the default dataset meaning the model will be trained on the all data.

For sampling and testing, make sure the pretrained models are placed at the correct paths.
</p>
</details> 

## Object motion to Human motion

To train and test the human motion generation given object motion, you need to go to the object2human folder.

```bash
cd object2human
```

<details><summary>detailed instructions</summary>
<p>
<!-- If data folder is not present in the InterAct folder, you need to modify the 'scripts/run_train_all_layer4.sh' file to set the hyperparameter '--data_root_folder' to point to the correct data folder. For other bash files, you also need to modify the hyperparameter '--data_root_folder' to point to the correct data folder. -->

Before training the model, you need to modify the 'scripts/run_train_all_layer4.sh' file to set the hyperparameter '--entity' to your wandb account. You can create a free account at <https://wandb.ai/>. For other bash files, you also need to modify the hyperparameter '--entity' to your wandb account.

To get bps features, you need to modify line 201 in 'object2human/manip/data/hand_foot_dataset_all.py' file to Mannually enable `get_bps_from_window_data_dict()`. And it only needs to be run once.


To train the different models on different datasets, you can run the following commands:

```bash
bash scripts/run_train_all_layer4.sh
```

```bash
bash scripts/run_train_all_layer6.sh
```

```bash
bash scripts/run_train_omomo_layer4.sh
```

```bash
bash scripts/run_train_omomo_layer6.sh
```

<!-- Pretrained model needs to present in the 'object2human/omomo_runs/*/weights' folder. We can test the model. -->

To test the different models on OMOMO dataset, you can run the following commands:

```bash
bash scripts/run_test_all_layer4.sh
```

```bash
bash scripts/run_test_all_layer6.sh
```

```bash
bash scripts/run_test_omomo_layer4.sh
```

```bash
bash scripts/run_test_omomo_layer6.sh
```
</p>
</details> 

## Human motion to Object motion

To train and test the object motion generation model, you need to go to the human2object folder.

```bash
cd human2object
```


<details><summary>detailed instructions</summary>
<p>

<!-- If data folder is not present in the InterAct folder, you need to modify the 'scripts/run_train_all_layer4.sh' file to set the hyperparameter '--data_root_folder' to point to the correct data folder. For other bash files, you also need to modify the hyperparameter '--data_root_folder' to point to the correct data folder. -->

Before training the model, you need to modify the 'scripts/run_train_all_layer4.sh' file to set the hyperparameter '--entity' to your wandb account. You can create a free account at <https://wandb.ai/>. For other bash files, you also need to modify the hyperparameter '--entity' to your wandb account.

To get bps features, you need to modify line 201 in 'object2human/manip/data/hand_foot_dataset_all.py' file to Mannually enable `get_bps_from_window_data_dict()`. And it only needs to be run once.

To train the different models on different datasets, you can run the following commands:

```bash
bash scripts/run_train_all_layer4.sh
```

```bash
bash scripts/run_train_omomo_layer4.sh
```

To test the different models on OMOMO dataset, you can run the following commands:

<!-- Pretrained model needs to present in the 'human2object/omomo_runs/*/weights' folder. We can test the model. -->

```bash
bash scripts/run_test_all_layer4.sh
```

```bash
bash scripts/run_test_omomo_layer4.sh
```


</p>
</details> 


## Past Interaction to Future

To train and evaluate the models, you need to go to the interdiff folder.

```bash
cd past2future
```

<details><summary>detailed instructions</summary>
<p>
There are six pretrained models, you can download them and put them in the results folder as follows:

```bash
past2future
├── results
    ├── models
    │   ├── 1.0.ckpt
    │   ├── 1.0_512.ckpt
    │   ├── 1.0_1536.ckpt
    │   ├── 1.0_behave.ckpt
    │   ├── 1.0_behave_512.ckpt
    │   └── 1.0_behave_1536.ckpt
```

You can train the models by the provided script.

To train the medium model with all data, run 

```bash
bash scripts/train_1.0.sh
```

To train the small model with all data, run 

```bash
bash scripts/train_1.0_512.sh
```

To train the large model with all data, run 

```bash
bash scripts/train_1.0_1536.sh
```

To train the medium model with BEHAVE data, run 

```bash
bash scripts/train_1.0_behave.sh
```

To train the small model with behave data, run 

```bash
bash scripts/train_1.0_behave_512.sh
```

To train the large model with behave data, run 

```bash
bash scripts/train_1.0_behave_1536.sh
```

We also provide code to evaluate the performance of the pretrained models.

```bash
bash scripts/eval_1.0.sh
```

```bash
bash scripts/eval_1.0_512.sh
```

```bash
bash scripts/eval_1.0_1536.sh
```

```bash
bash scripts/eval_1.0_behave.sh
```


```bash
bash scripts/eval_1.0_behave_512.sh
```

```bash
bash scripts/eval_1.0_behave_1536.sh
```

</p>
</details> 
