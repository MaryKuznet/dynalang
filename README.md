# Running Dynalang in a TextCrafter environment

## Getting Started

1. Clone this project:
```
git clone https://github.com/MaryKuznet/dynalang.git
```

2. In the data lang/embedded/ends/data folder, put all pl files with embedded datasets from https://drive.google.com/drive/folders/1pyjotyVI6Op5VRXxkKaujL3uGxKhkLc9?usp=sharing

3. Build container:
```
docker build -f dynalang/Dockerfile -t dynalang_crafter .
```
4. To run the container, use your WANDB_API_KEY and the path to dynalang. The weights of the model will be saved to the logdir folder.
```
docker run -it --rm --gpus all --env WANDB_API_KEY=$WANDB_API_KEY -v ~/dynalang/dynalang:/dynalang -v ~/logdir:/root/logdir dynalang_crafter
```
5. Activate the dynalang environment in the container
```
conda init bash
bash
conda activate dynalang
```

## Starting the train
Launching experiments in the format:

```
sh scripts/run_text_crafter.sh data+_train_new $NAME_OF_EXPERIMENT$ $GPU$ $SEED$
```
Example:
```
sh scripts/run_text_crafter.sh data_train_new debug_data 1 2
```
data+_train_new is the mode. The first is the choice of the dataset $data$ – simple tasks, $data+$ - simple tasks mixed with instructions, $dataset$ – only instructions.

There are $train$ and $test$ modes, but test is run using a different script. $new$ and $old$ data encoding modes. $old$ is the old way with t5, $new$ is your new one.
Thus, there are 6 modes:
```
data_train_new
data+_train_new
dataset_train_new
data_train_old
data+_train_old
dataset_train_old
```

## Starting the test
The test is run in the format:
```
sh scripts/test_text_crafter.sh data+_test_new $NAME_OF_EXPERIMENT$ $GPU$ $SEED$ $NAME_OF_FOLODER_WITH_WEIGHTS$
```
Example:
```
sh scripts/test_text_crafter.sh data+_test_new debug_test_data+ 1 2 train_dataset_2
```
