
## Instructions

A quick instruction is given for readers to reproduce the whole process.



Requirements 

- pytorch  1.9.1+cu102
- torch_scatter 2.0.9



For transductive reasoning

    cd transductive
    python -W ignore train.py --data_path=data/WN18RR



For inductive reasoning

    cd inductive
    python -W ignore train.py --data_path=data/WN18RR_v1



### Data splition in transductive setting

We follow the rule mining methods, i.e., [Neural-LP](https://github.com/fanyangxyz/Neural-LP) and [DRUM](https://github.com/alisadeghian/DRUM), to randomly split triplets in the original `train.txt` file into two files `facts.txt` and `train.txt` with ratio 3:1. This step is to make sure that the query triplets will not be leaked in the fact triplets used in HyperKGR. Empirically, increasing the ratio of facts, e.g. from 3:1 to 4:1, will lead to better performance.




