import MyUtils.Util.Misc as util
import keras
import numpy as np
import wandb
from omegaconf import OmegaConf
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback
import argparse
import torch
import tensorflow as tf
import os
import sys
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process input files and set verbosity.")

    # Add arguments
    parser.add_argument('--is_sweep', type=bool, help='run sweep or not', default=False)
    parser.add_argument('--sweep_id', type=str, help='sweep id', default=None)
    # Parse the arguments
    args = parser.parse_args()
    return args
assert torch.cuda.is_available(), "No GPU available"

def train():
    # Load configuration
    cfg = util.load_and_override_config(".", "config")
    wandb.init(project=cfg.project_name)
    
    cfg = util.load_and_override_config(".", "config")
    print(OmegaConf.to_yaml(cfg))
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    # Load the data
    x_train = np.load(cfg.x_train_path)
    y_train = np.load(cfg.y_train_path)
    x_val = np.load(cfg.x_val_path)
    y_val = np.load(cfg.y_val_path)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)
    
    # Create the dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    augmenter = util.get_data_augmentation_layers(cfg.data_augmentation)

    def augment(x, y):
        x = augmenter(x)
        return x, y
    
    train_dataset = train_dataset.map(augment)

    num_classes = cfg.num_classes
    input_shape = tuple(cfg.input_shape)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(int(64*cfg.param_scale), kernel_size=(3, 3), activation=cfg.activation),
            keras.layers.Conv2D(int(64*cfg.param_scale), kernel_size=(3, 3), activation=cfg.activation),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(int(128*cfg.param_scale), kernel_size=(3, 3), activation=cfg.activation),
            keras.layers.Conv2D(int(128*cfg.param_scale), kernel_size=(3, 3), activation=cfg.activation),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(cfg.dropout_rate),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()

    model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
    )

    callbacks = [ 
    keras.callbacks.EarlyStopping(monitor="val_acc",patience=cfg.patience, verbose=1, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(cfg.best_model_path, save_best_only=True),
    keras.callbacks.TensorBoard(log_dir="./logs"), 
    WandbMetricsLogger(),
    ]

    model.fit(train_dataset,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=callbacks,
        validation_data=val_dataset

    )

        # Log the best model as an artifact
    artifact = wandb.Artifact('best-model', type='model')
    artifact.add_file(cfg.best_model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    cfg = util.load_and_override_config(".", "config")
    args = parse_args()
    print("args: ", args)
    is_sweep = args.is_sweep
    if is_sweep:
        # use sweep id if in args
        if args.sweep_id:
            sweep_id = args.sweep_id
            wandb.agent(sweep_id, function=train,project=cfg.project_name)
        else:
            print("Please provide sweep_id")
    else:
        train()

