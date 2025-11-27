
## Introduction


## Dependencies

- torch == 1.12.1
- torch_scatter == 2.0.9
- numpy == 1.21.6
- scipy == 1.10.1



## Reproduction

### Transductive settings (in `\transductive`)

#### Reproduction with training scripts

##### Family dataset

```
python3 train.py --data_path ./data/family/ --train --topk 100 --layers 8 --fact_ratio 0.90 --gpu 0
```

##### UMLS dataset
```
python3 train.py --data_path ./data/umls/ --train --topk 100 --layers 5 --fact_ratio 0.90 --gpu 0
```

##### WN18RR dataset
```
python3 train.py --data_path ./data/WN18RR/ --train --topk 1000 --layers 8 --fact_ratio 0.96 --gpu 0
```

##### FB15k-237 dataset
```
python3 train.py --data_path ./data/fb15k-237/ --train --topk 2000 --layers 7 --fact_ratio 0.99 --remove_1hop_edges --gpu 0
```

##### NELL995 dataset
```
python3 train.py --data_path ./data/nell/ --train --topk 2000 --layers 6 --fact_ratio 0.95 --gpu 0
```

##### YAGO3-10 dataset
```
python3 train.py --data_path ./data/YAGO/ --train --topk 1000 --layers 8 --fact_ratio 0.995 --gpu 0
```


#### Reproduction with saved model checkpoints

##### Family dataset

```
python3 train.py --data_path ./data/family/ --eval --topk 100 --layers 8 --gpu 0 --weight ./data/family/8-layers-best.pt
```

##### UMLS dataset

```
python3 train.py --data_path ./data/umls/ --eval --topk 100 --layers 5 --gpu 0 --weight ./data/umls/5-layers-best.pt
```

##### WN18RR dataset

```
python3 train.py --data_path ./data/WN18RR/ --eval --topk 1000 --layers 8 --gpu 0 --weight ./data/WN18RR/8-layers-best.pt
```

##### FB15k-237 dataset

```
python3 train.py --data_path ./data/fb15k-237/ --eval --topk 2000 --layers 7 --gpu 0 --weight ./data/fb15k-237/7-layers-best.pt
```

##### NELL995 dataset

```
python3 train.py --data_path ./data/nell/ --eval --topk 2000 --layers 6 --gpu 0 --weight ./data/nell/6-layers-best.pt
```

##### YAGO3-10 dataset

```
python3 train.py --data_path ./data/YAGO/ --eval --topk 1000 --layers 8 --gpu 0 --weight ./data/YAGO/8-layers-best.pt
```


### Inductive settings (in `\inductive`)

#### Reproduction with training scripts



For example, training on `WN18RR v1` dataset:

```
python3 train.py --data_path ./data/WN18RR_v1
```



