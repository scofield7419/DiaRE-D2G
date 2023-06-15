**Python implementation of IJCAI 2022 Paper [Global Inference with Explicit Syntactic and Discourse Structures for Dialogue-Level Relation Extraction](https://www.ijcai.org/proceedings/2022/0570.pdf)**

----------

# Requirement

Install the packages as in `requirement.txt`.
```bash
pip install -r requirement.txt
```

# Data Preprocessing

* Downloading the [the dataset and pretrained embeddings](https://github.com/nlpdata/dialogre), which are officially provided by [Dialogue-Based Relation Extraction](https://arxiv.org/abs/2004.08056). 
* Constructing **Dialogue-level Mixed Dependency Graph** for instances via `data/D2G_construction`.
* Preparing data inputs via `data/gen_data.py`.



# System Running:
* Training: 
```
python train.py \
        --save_name dialogre  \
        --use_spemb True  \
        --use_wratt True  \
        --use_arc True
```

* Testing:

```
python test.py \
         --save_name dialogre \
         --use_spemb True \
         --use_wratt True  \
         --use_arc True
```


## Credit

Codes are based on the repos of 1) the ACL-20 Paper 
"[Reasoning with Latent Structure Refinement for Document-Level Relation Extraction](https://arxiv.org/abs/2005.06312)",
and TASLP Paper 
"[Relation Extraction in Dialogues: A Deep Learning Model Based on the Generality and Specialty of Dialogue Text](https://ieeexplore.ieee.org/document/9439807?source=authoralert)".




## Citation

```
@inproceedings{FeiDiaREIJCAI22,
  title     = {Global Inference with Explicit Syntactic and Discourse Structures for Dialogue-Level Relation Extraction},
  author    = {Fei, Hao and Li, Jingye and Wu, Shengqiong and Li, Chenliang and Ji, Donghong and Li, Fei},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI}},
  pages     = {4082--4088},
  year      = {2022},
}
```


