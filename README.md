
# Context Is(n't) King: Named Entity Recognition Based Solely on Surrounding Words
## Second Year Project (Introduction to Natural Language Processing and Deep Learning)



## Authors

- Christian Hetling, [chrhe@itu.dk](mailto:chrhe@itu.dk)
- Krzysztof Parocki, [krpa@itu.dk](mailto:krpa@itu.dk)
- Malthe Have Musaeus, [mhmu@itu.dk](mailto:mhmu@itu.dk)


## Getting started

1. Start by cloning the repo: `git clone https://github.com/Hetling/NLP-second-year-project.git`
1. Download the contextualized word embedding `pickle` files here: [https://drive.google.com/drive/folders/1SinJt4EaPbn2el-Yjhj_KN7KkaCW7LY2](https://drive.google.com/drive/folders/1SinJt4EaPbn2el-Yjhj_KN7KkaCW7LY2)
2. Create a new `models` folder in the root directory
3. Place the downloaded `data` folder inside of the newly created `models` folder. 

## Usage

Now you are ready to train, validate, and test the models. The `main.py` file acts as a simple CLI to interact with the models. The usage of which is described below:
#### To train all models and save them to disk

```bash
  python main.py --train
```
#### To train only approach 1 and 2 without saving them

```bash
  python main.py --train --approach-1 --approach-2 --save False
```

#### To validate all models from disk. Remember to train them first
```bash
  python main.py --validate
```

#### To validate only approach 1 and 2
```bash
  python main.py --validate --approach-1 --approach-2
```

#### To test all models from disk. Again remember to train them first
```bash
  python main.py --test
```
#### To test only approach 1 and 2
```bash
  python main.py --test --approach-1 --approach-2
```

To train the baseline model, run `baseline.py` without any arguments. To reproduce visualizations, see `visualize_net.ipynb`