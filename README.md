# GEMEL: Generative Multimodal Entity Linking

<div align="center">

 [Overview](https://github.com/HITsz-TMG/GEMEL#sparkles-overview) | [News](https://github.com/HITsz-TMG/GEMEL#fire-news) | [Architecture](https://github.com/HITsz-TMG/GEMEL#rocket-architecture) | [Usage](https://github.com/HITsz-TMG/GEMEL#rotating_light-usage) | [Citation](https://github.com/HITsz-TMG/GEMEL#citation)

</div>


## :sparkles: Overview

This repository contains the official implementation of our **LREC-COLING 2024** paper, [**Generative Multimodal Entity Linking**](https://arxiv.org/abs/2306.12725).

GEMEL is a simple yet effective Generative Multimodal Entity Linking framework based on Large Language Models (LLMs), which directly generates target entity names. We keep the vision and language model frozen and only train a feature mapper to enable cross-modality interactions. Extensive experiments show that, with only ~0.3% of the model parameters fine-tuned, GEMEL achieves state-of-the-art results on two well-established MEL datasets, namely [WikiDiverse](https://arxiv.org/abs/2204.06347) and [WikiMEL](https://dl.acm.org/doi/abs/10.1145/3477495.3531867). The performance gain stems from mitigating the popularity bias of LLM predictions and disambiguating less common entities effectively. Our framework is compatible with any off-the-shelf language model, paving the way towards an efficient and general solution for utilizing LLMs in the MEL task.

**Checkpoints** and preprocessed **data** can be accessed [here](https://drive.google.com/drive/folders/1M2wF2RkWpzeCKYj032bOryVPMM_DSubE?usp=sharing).

If you have any question, please feel free to contact me via email at shisenbaohit@gmail.com or submit your issue in the repository.

## :fire: News

[23.07.14] We release the codes and the checkpoints of GEMEL.

[24.03.19] We have updated our paper.

## :rocket: Architecture

[Here](https://arxiv.org/abs/2306.12725), you can see the detailed architecture and some experimental analyses of GEMEL.

<p align="center" width="60%"><img src="GEMEL.png" alt="GEMEL" style="width: 100%;  display: block; margin: auto;"></p>

## 🛠️ Usage

### Environment

```
conda create -n GEMEL python=3.7
conda activate GEMEL
pip install -r requirements.txt
```
For different CUDA versions you need to install the corresponding PyTorch package. Find the appropriate installation package on the [PyTorch](https://pytorch.org/get-started/previous-versions/) website. To install PyTorch, we use the following command:

```
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```


### Data
We have preprocessed the text, image, and knowledge base data. Download data from [here](https://drive.google.com/drive/folders/1M2wF2RkWpzeCKYj032bOryVPMM_DSubE?usp=sharing) and move to the `./data` folder. [Here](https://github.com/Senbao-Shi/How-to-Build-and-Use-a-Prefix-Tree/tree/main) we offer guidelines on how to build and use a prefix tree for constrained decoding.

```
train.json, dev.json, test.json         ->      textual data files
clip_vit_large_patch14_1024.hdf5        ->      visual data file
prefix_tree_opt.pkl                     ->      prefix tree of entity name
SimCSE_train_mention_embeddings.pkl     ->      training set mention embeddings
```

### Train
Running `main.py` directly will use the WikiDiverse dataset, opt-6.7b model:

```
python main.py
```

The model structure is in model.py, the default parameters are in params.py, and most of the data processing is in utils.py.

You can customize some parameter settings, see `params.py` for details. Here are some examples of how to train GEMEL:

For training with the WikiDiverse dataset:
```
python main.py --dataset wikidiverse --model_name opt-6.7b --ICL_examples_num 16
```

For training with the WikiMEL dataset:
```
python main.py --dataset wikimel --model_name opt-6.7b --ICL_examples_num 16
```

### Test
Download the checkpoint from [here](https://drive.google.com/drive/folders/1M2wF2RkWpzeCKYj032bOryVPMM_DSubE?usp=sharing) and move to the `./checkpoint` folder.

For testing on WikiDiverse test set:
```
python infe.py --dataset wikidiverse --model_name opt-6.7b --best_ckpt opt-6.7b_wikidiverse_linear_4token_16examples_82_77.pkl
```

For testing on WikiMEL test set:
```
python infe.py --dataset wikimel --model_name opt-6.7b --best_ckpt opt-6.7b_wikimel_linear_4token_16examples_75_53.pkl
```


## 📚 Citation
```
@inproceedings{shi-etal-2024-generative,
    title = "Generative Multimodal Entity Linking",
    author = "Shi, Senbao and Xu, Zhenran and Hu, Baotian and Zhang, Min",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    year = "2024",
    url = "https://aclanthology.org/2024.lrec-main.676/",
    pages = "7654--7665"
}
```




