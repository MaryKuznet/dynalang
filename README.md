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
sh scripts/run_text_crafter.sh $TYPE_of_DATA$ $TYPE_OF_LEARNING$ $TYPE_OF_ENCODER$ $NAME_OF_EXPERIMENT$ $GPU$ $SEED$
```
Example:
```
sh scripts/run_text_crafter.sh data train old experiment_1 1 2
```
$TYPE_of_DATA$ --- simple instructions (data), complex instructions (dataset) or a mixture of them (data+).
$TYPE_OF_LEARNING$ --- train or test mode.
$TYPE_OF_ENCODER$ --- $new$ and $old$ data encoding modes. $old$ is the old way with t5, $new$ is your new one.

## Starting the test
The test is run in the format:
```
sh scripts/test_text_crafter.sh $TYPE_of_DATA$ $TYPE_OF_LEARNING$ $TYPE_OF_ENCODER$ $SEED_OF_THE_ENVIRONMENT$ $NAME_OF_EXPERIMENT$ $GPU$ $SEED$ $NAME_OF_THE_CHECKPOINT_FOLODER$
```
Example:
```
sh scripts/test_text_crafter.sh data test old 1 test_experiment 1 2 experiment_1
```
