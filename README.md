### Installation requirements
- Detectron2 library, [see the official installation guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
  *(recommended to make a new directory, e.g. extern, and install the required library here)*
- Albumentations library, [see the official github page](https://github.com/albumentations-team/albumentations)
- OpenCV, Pillow

You can also see the [sample installation script](./scripts/01_metacentrum_installation.sh)
for the [MetaCentrum](https://metavo.metacentrum.cz/en/index.html)

### Data preparation
Download and extract development/test set from the AIcrowd page of the [ImageCLEF](https://www.imageclef.org/2021) DrawnUI 2021 challenge,
i.e. [Screenshot task (SS)](https://www.aicrowd.com/clef_tasks/39/task_dataset_files?challenge_id=663)
or [Wireframe task (WF)](https://www.aicrowd.com/clef_tasks/39/task_dataset_files?challenge_id=662) (both sets should be the same),
in the following format:

```
./data/
├── screenshot_development_set/
│   ├── train/
│   │   ├── images/
│   │   │   └── <...>.jpg
│   │   └── train_set.json
│   └── validation/
│       ├── images/
│       │   └── <...>.jpg
│       └── val_set.json
├── screenshot_test_set/
│   └── test/images/
│       └── <...>.jpg
├── wireframe_development_set/
│   ├── images/
│   │   └── <...>.jpg
│   └── development_set.json
└── wireframe_test_set/
    └── test/
        └── <...>.jpg
```

### Configuration
The basic configuration for both Screenshot and Wireframe tasks can be adjusted in the "CONFIG" dictionary of the [run.py](./run.py).

The default configuration is consistent with the final settings that have been used for comparison of different architectures in our approach of the challenge (see [our working notes](https://scholar.google.com/scholar?hl=cs&as_sdt=0%2C5&q=Improving+web+user+interface+element+detection+using+Faster+R-CNN&btnG=), chapter 4 Backbones comparison).
Additional used configuration is summarized in the following tables *(SS - Screenshot task, WF - Wireframe task)*:

| Character | Explanation                                                                                                    |
| :-------- | :------------------------------------------------------------------------------------------------------------- |
| "empty"   | not used / not relevant                                                                                        |
| ✓         | used / relevant                                                                                                |
| ~         | modified, see the <a href="#addit_configs">Additional configurations</a> for specific parameters of the CONFIG |
| default   | specific for Anchor generator, see the <a href="#exp5">Anchor box proposals</a>                                |

<table>
<thead>
  <tr>
    <th rowspan="2">Subsection name</th>
    <th colspan="2">Task relevance</th>
    <th colspan="2">Discarding images</th>
    <th colspan="2">Relative Resize</th>
    <th colspan="2">Cutout</th>
    <th rowspan="2">Anchor generator</th>
  </tr>
  <tr>
    <td>SS</td>
    <td>WF</td>
    <td>SS</td>
    <td>WF</td>
    <td>ratio</td>
    <td>delta</td>
    <td>holes</td>
    <td>size</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="#exp1">Filtering noisy data</a></td>
    <td>✓</td>
    <td></td>
    <td><a href="#exp1">~</a></td>
    <td></td>
    <td>0.8</td>
    <td>0.1</td>
    <td></td>
    <td></td>
    <td>default</td>
  </tr>
  <tr>
    <td><a href="#exp2">Image resize</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td></td>
    <td><a href="#exp2">~</a></td>
    <td><a href="#exp2">~</a></td>
    <td></td>
    <td></td>
    <td>default</td>
  </tr>
  <tr>
    <td><a href="#exp3">Cutout augmentation</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td></td>
    <td>0.8</td>
    <td>0.1</td>
    <td><a href="#exp3">~</a></td>
    <td><a href="#exp3">~</a></td>
    <td>default</td>
  </tr>
  <tr>
    <td><a href="#exp4">Color space</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td></td>
    <td>0.8</td>
    <td>0.1</td>
    <td></td>
    <td ></td>
    <td>default</td>
  </tr>
  <tr>
    <td><a href="#exp5">Anchor box proposals</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td></td>
    <td>0.8</td>
    <td>0.2</td>
    <td>4</td>
    <td>0.05</td>
    <td><a href="#exp5">~</a></td>
  </tr>
</tbody>
</table>

#### <span id="addit_configs">Additional configurations</span>
<details>
<summary id="exp1">Filtering noisy data</summary>

| Subexperiment                     | discard_data              | threshold_box |
| :-------------------------------- | :------------------------ | :------------ |
| #1 Original data                  | [ ]                       |               |
| #2 No homogenous images           | ["hmg_imgs"]              |               |
| #3 No homogenous images and boxes | ["hmg_imgs", "hmg_boxes"] | 0.2           |
| #4 No homogenous images and boxes | ["hmg_imgs", "hmg_boxes"] | 0.6           |
| #5 No homogenous images and boxes | ["hmg_imgs", "hmg_boxes"] | 1.0           |
| #6 No homogenous images and boxes | ["hmg_imgs", "hmg_boxes"] | 1.4           |
| #7 No homogenous images and boxes | ["hmg_imgs", "hmg_boxes"] | 1.8           |
</details>

<details>
<summary id="exp2">Image resize</summary>

| Subexperiment             | relative_resize_with_crop | resize_ratio | resize_ratio_delta |
| :------------------------ | :------------------------ | :----------- | :----------------- |
| #1 Resize Shortest Edge   | False                     |              |                    |
| #2 Random Relative Resize | True                      | 0.8          | 0.1                |
| #3 Random Relative Resize | True                      | 0.8          | 0.2                |
| #4 Random Relative Resize | True                      | 0.7          | 0.3                |
</details>

<details>
<summary id="exp3">Cutout augmentation</summary>

| Subexperiment     | cutout | cutout_max_holes | cutout_max_size |
| :---------------- | :----- | :--------------- | :-------------- |
| #1 Without Cutout | False  |                  |                 |
| #2 With Cutout    | True   | 4                | 0.05            |
| #3 With Cutout    | True   | 8                | 0.05            |
| #4 With Cutout    | True   | 16               | 0.05            |
| #5 With Cutout    | True   | 16               | 0.025           |
</details>

<details>
<summary id="exp4">Color space</summary>

No additional specification of the "CONFIG" dictionary is needed,
just use the argument [-G, --greyscale] for running the script when you want to convert images into greyscale (see section <a href="#training">Training</a>).
</details>

<details>
<summary id="exp5">Anchor box proposals</summary>

| Subexperiment                  | aspect_ratios        | anchor_sizes                      |
| :----------------------------- | :------------------- | :-------------------------------- |
| #1 default                     | [0.5, 1.0, 2.0]      | [[32], [64], [128], [256], [512]] |
| #2 default + horizontal        | [0.2, 0.5, 1.0, 2.0] | [[32], [64], [128], [256], [512]] |
| #3 statistical                 | [0.1, 0.5, 1.0, 1.5] | [[32], [64], [128], [256], [512]] |
| #4 statistical + smaller sizes | [0.1, 0.5, 1.0, 1.5] | [[16], [32], [64], [128], [256]]  |
</details>

### <span id="training">Training</span>
To train the models use the following format
(add argument [-G, --greyscale] for training on the greyscale images, these parameters are sufficient for the reproduction of the experiments in the article):
```
python run.py --train -b <batch_size> -a <accum_grad> -lr <learning_rate> -e <epochs> -DT <dataset> -m <model_name>
```

<table>
<thead>
  <tr>
    <th>Argument</th>
    <th>Recommended value</th>
    <th>Short description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>[-T, --train]</td>
    <td></td>
    <td>Train the model</td>
  </tr>
  <tr>
    <td>[-b, --batch_size]</td>
    <td>1</td>
    <td>Batch size</td>
  </tr>
  <tr>
    <td>[-a, --accum_grad]</td>
    <td>4</td>
    <td>Accumulated gradient (preservation of the batch size if there is not enough memory)</td>
  </tr>
  <tr>
    <td>[-lr, --base_lr]</td>
    <td>0.0025</td>
    <td>Base learning rate</td>
  </tr>
  <tr>
    <td rowspan="2">[-e, --epochs]</td>
    <td>20</td>
    <td rowspan="2">Number of epochs (20 for the Screenshot task, 40 for the Wireframe task)</td>
  </tr>
  <tr>
    <td>40</td>
  </tr>
  <tr>
    <td rowspan="2">[-DT, --dataset]</td>
    <td>"screenshot"</td>
    <td rowspan="2">Dataset for training</td>
  </tr>
  <tr>
    <td>"wireframe"</td>
  </tr>
  <tr>
    <td rowspan="3">[-m, --model]</td>
    <td>"faster_rcnn_R_50_FPN_3x"</td>
    <td rowspan="3">Model to be trained (ResNet-50, ResNet-101, or ResNeXt-101)</td>
  </tr>
  <tr>
    <td>"faster_rcnn_R_101_FPN_3x"</td>
  </tr>
  <tr>
    <td>"faster_rcnn_X_101_32x8d_FPN_3x"</td>
  </tr>
</tbody>
</table>

#### Additional arguments
<table>
<thead>
  <tr>
    <th>Argument</th>
    <th>Recommended value</th>
    <th>Short description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>[-G, --greyscale]</td>
    <td></td>
    <td>Use greyscale images</td>
  </tr>
  <tr>
    <td>[-P, --predict]</td>
    <td></td>
    <td>Show predictions of the model, use this with argument [-T, --train] and [-LP, --lim_predict]</td>
  </tr>
  <tr>
    <td>[-LP, --lim_predict]</td>
    <td>100</td>
    <td>Limit the number of predictions to be visualized, take into account that higher value rapidly slows down the evaluation runtime of the script</td>
  </tr>
  <tr>
    <td>[-R, --resume]</td>
    <td></td>
    <td>Resume training from the last checkpoint, argument [-C, --checkpoint] is needed</td>
  </tr>
  <tr>
    <td>[-C, --checkpoint]</td>
    <td>&lt;path_to_your_model&gt;</td>
    <td>Path to the checkpoint of your model (must ends with model_final.pth)</td>
  </tr>
  <tr>
    <td>[-pre, --prefix]</td>
    <td>&lt;prefix&gt;</td>
    <td>Prefix specifying the output directory, where all output files (checkpoint, predictions, submission file, ...) should be stored</td>
  </tr>
</tbody>
</table>

You can also see the example scripts for [training the model](./scripts/02_metacentrum_train_screenshot_sample.sh)
and for [continue training from the checkpoint](./scripts/03_metacentrum_continue_training_screenshot_sample.sh)
using [MetaCentrum](https://metavo.metacentrum.cz/en/index.html) services.

#### Output directory
All files are stored in the ./&#95;&#95;OUTPUT&#95;&#95;/ directory. Your models are stored under following format:
```
./__OUTPUT__/model_output/<challenge_task>/<color_space>/<prefix><model_name_with_arguments>
```

Where:

| Definition                    | Description                                                                                                     |
| :---------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| \<challenge_task\>            | The task of the challenge, i.e. Screenshot or Wireframe                                                         |
| \<color_space\>               | Used color space, i.e. RGB (default) or greyscale (use [-G, --greyscale] argument)                              |
| \<prefix\>                    | A prefix specifying the output directory, i.e. use [-pre, --prefix] argument                                    |
| \<model_name_with_arguments\> | Used model ([-m, --model] argument) and additional arguments, such as [-lr, --learning rate], or [-e, --epochs] |

#### Tensorboard
You also can use tensorboard to show your results online:
```
tensorboard --logdir=<model_outputs>
```

Where the *<model_outputs>* is path to the desired output directory, e.g. the described one in the previous section.

### Challenge results (DrawnUI 2021)

#### Screenshot task
| Participants       | mAP@0.5 IoU | Recall@0.5 IoU |
| :----------------- | :---------: | :------------: |
| **vyskocj (ours)** | **0.628**   | **0.830**      |
| *baseline*         | *0.329*     | *0.408*        |

#### Wireframe task
| Participants       | mAP@0.5 IoU | Recall@0.5 IoU |
| :----------------- | :---------: | :------------: |
| **vyskocj (ours)** | **0.900**   | **0.934**      |
| pwc                | 0.836       | 0.865          |
| *baseline*         | *0.747*     | *0.763*        |
| AIMultimediaLab    | 0.216       | 0.319          |

### Citation
If you use these scripts in your research or wish to refer to our approach, please use the following BibTeX entry.

```BibTeX
@inproceedings{vyskocil2021improving,
  title     = {Improving web user interface element detection using Faster R-CNN},
  author    = {Vysko{\v{c}}il, Ji{\v{r}}{\'\i} and Picek, Luk{\'a}{\v{s}}},
  year      = {2021},
  booktitle = {CLEF2021 Working Notes},
  series    = {{CEUR} Workshop Proceedings},
  pages     = {1375-1386},
  month     = {September 21-24},
  address   = {Bucharest, Romania},
}
```
