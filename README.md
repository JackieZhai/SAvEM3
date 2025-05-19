# SAEM<sup>2</sup>-SAvEM<sup>3</sup>

[![Commit](https://img.shields.io/github/last-commit/JackieZhai/SAvEM3)](https://github.com/JackieZhai/SAvEM3/commits)
![RepoSize](https://img.shields.io/github/repo-size/JackieZhai/SAvEM3)
![CodeSize](https://img.shields.io/github/languages/code-size/JackieZhai/SAvEM3)

**Hao Zhai\*, Jinyue Guo\*, Yanchao Zhang\*, Jing Liu, Hua Han**

*Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology, Institute of Automation, Chinese Academy of Science* <br>
*School of Future Technology, School of Artificial Intelligence, University of Chinese Academy of Sciences*

## To-do List

Codes and data collections are under development. The following tasks are planned to be completed in the next few months.

- [ ] 2025.04.xx: Release the first version of SAEM<sup>2</sup>-SAvEM<sup>3</sup>.
- [x] 2024.09.24: Finish the development of the pipeline.
- [x] 2024.02.05: Create the repository.


## Pipeline

## SAEM<sup>2</sup>

### Data bank: diverse imaging methods and animal species

* [EMNeuron](https://huggingface.co/datasets/yanchaoz/EMNeuron)

### Data engine: unified labeling styles of the boundaries among four different imaging methods

### Auxiliary learning: hybrid representation of neural morphology for SAM

```mermaid
flowchart TD
classDef SAM-HQ fill:#666
classDef SAM-HQ-Membrane fill:#c66
points-->sparse_prompt_embed
boxes-->sparse_prompt_embed
masks-->dense_prompt_embed
image_embed--ConvTrans-->Add1[+]:::SAM-HQ
interm_embed:::SAM-HQ--ConvTrans-->Add1
Add1-->hq_feature:::SAM-HQ
dense_prompt_embed-->Add2[+]
image_embed-->Add2
iou_token-->tokens
mask_token-->tokens
hq_token:::SAM-HQ-->tokens
mem_mask_token:::SAM-HQ-Membrane-->tokens
sparse_prompt_embed-->tokens
tokens--point_embed-->Transformer
image_pe--position_encode-->Transformer
Add2--image_embed-->Transformer
Transformer--queries-->iou_token_out
Transformer--queries-->mask_token_out
Transformer--queries-->hq_token_out:::SAM-HQ
Transformer--keys-->mask_feature
mask_token_out--MLP-->Matmul1[matmul]
mask_feature--ConvTrans-->Matmul1
Matmul1-->SAM_masks
mask_feature--ConvTrans-Conv-->Add3[+]:::SAM-HQ
hq_feature-->Add3
hq_token_out--MLP-->Matmul2
Add3-->Matmul2[matmul]:::SAM-HQ
Matmul2-->SAM-HQ_masks:::SAM-HQ
Transformer--queries-->mem_mask_token_out:::SAM-HQ-Membrane
Add3-->Matmul3
mem_mask_token_out--MLP-->Matmul3[matmul]:::SAM-HQ-Membrane
Matmul3-->SAM-HQ_mem_masks:::SAM-HQ-Membrane
```

## SAvEM<sup>3</sup>


## Acknowledgements

- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [sam-hq](https://github.com/SysCV/sam-hq)
- [micro-sam](https://github.com/computational-cell-analytics/micro-sam)
- [SuperHuman](https://github.com/weih527/SuperHuman)
- [SegNeuron](https://github.com/yanchaoz/SegNeuron)


## Citation

```
@inproceedings{Zhai-SAvEM3,
    author={Zhai, Hao and Guo, Jinyue and Zhang, Yanchao and Liu, Jing and Han, Hua},
    booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
    title={SAvEM3: Pretrained and Distilled Models for General-purpose 3D Neuron Reconstruction}, 
    year={2024},
    pages={3972-3977},
    doi={10.1109/BIBM62325.2024.10822494}
}
```