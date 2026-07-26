"""
Training script for the Plant Disease Detection model.

Trains an EfficientNetB3-based classifier on the PlantVillage dataset
(38 classes: 14 crops x healthy/disease states).

Usage:
    python train_model.py --data_dir /path/to/PlantVillage --epochs 15

Expected data layout (standard ImageFolder / Keras flow_from_directory format):
    data_dir/
        Apple___Apple_scab/
            img1.jpg
            img2.jpg
        Apple___Black_rot/
            ...
        ...

Dataset: https://huggingface.co/datasets/1aurent/PlantVillage
"""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB3

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.model_utils import CLASS_NAMES, IMG_SIZE, NUM_CLASSES  # noqa: E402


def build_model(num_classes: int = NUM_CLASSES, fine_tune_at: int = 0) -> tf.keras.Model:
    """Build an EfficientNetB3 transfer-learning model.

    A frozen (or partially fine-tuned) EfficientNetB3 backbone pretrained on
    ImageNet feeds a small classification head. `fine_tune_at` controls how
    many of the backbone's top layers get unfrozen for fine-tuning; 0 means
    the backbone stays fully frozen (feature-extraction mode).
    """
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = fine_tune_at > 0
    if fine_tune_at > 0:
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


def get_data_generators(data_dir: str, batch_size: int, validation_split: float = 0.2):
    """Create train/validation datasets using Keras' image_dataset_from_directory.

    Class order is forced to match CLASS_NAMES (sorted) so the trained
    model's output index matches utils/model_utils.py::CLASS_NAMES exactly.
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        class_names=CLASS_NAMES,
        label_mode="categorical",
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        class_names=CLASS_NAMES,
        label_mode="categorical",
    )

    normalization = layers.Rescaling(1.0 / 255)
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (augmentation(normalization(x), training=True), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Train the plant disease classifier.")
    parser.add_argument("--data_dir", required=True, help="Path to PlantVillage-style dataset directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=5, help="Additional fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="../model/plant_disease_model.keras")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    train_ds, val_ds = get_data_generators(args.data_dir, args.batch_size)

    print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        callbacks.ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True),
    ]

    print("\n=== Phase 1: Training classification head (frozen backbone) ===")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb)

    if args.fine_tune_epochs > 0:
        print("\n=== Phase 2: Fine-tuning top backbone layers ===")
        base_model = model.layers[1]
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr / 10),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=cb,
        )

    model.save(args.output)
    print(f"\nModel saved to {args.output}")

    val_loss, val_acc, val_top3 = model.evaluate(val_ds)
    metrics = {"val_loss": val_loss, "val_accuracy": val_acc, "val_top3_accuracy": val_top3}
    with open(os.path.join(os.path.dirname(args.output), "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Final validation metrics: {metrics}")


if __name__ == "__main__":
    main()
