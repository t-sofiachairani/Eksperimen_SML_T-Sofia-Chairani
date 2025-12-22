import argparse
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf

from automate_preprocess import main as preprocess_main  # reuse script


def load_npz(npz_path: Path):
    data = np.load(npz_path)
    return data["X"], data["y"]


def build_model(num_classes: int):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tracking_uri", type=str, required=True)
    ap.add_argument("--experiment_name", type=str, default="kue-ci")
    ap.add_argument("--dagshub_user", type=str, required=True)
    ap.add_argument("--dagshub_repo", type=str, required=True)
    ap.add_argument("--dagshub_token", type=str, required=True)
    args = ap.parse_args()

    # --- Preprocess (convert raw -> npz)
    import sys

    sys.argv = [
        "automate_preprocess.py",
        "--raw_dir",
        args.raw_dir,
        "--out_dir",
        args.out_dir,
        "--img_size",
        "224",
    ]
    preprocess_main()

    data_dir = Path(args.out_dir)
    with open(data_dir / "label_map.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    X_train, y_train = load_npz(data_dir / "train.npz")
    X_val, y_val = load_npz(data_dir / "validation.npz")
    X_test, y_test = load_npz(data_dir / "test.npz")

    num_classes = len(classes)

    # --- MLflow online (DagsHub)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Set auth via env
    os.environ["MLFLOW_TRACKING_USERNAME"] = args.dagshub_user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = args.dagshub_token

    # AUTLOG STANDAR
    mlflow.autolog()

    with mlflow.start_run(run_name="ci_train"):
        model = build_model(num_classes)

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
        )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss={loss:.4f}, test acc={acc:.4f}")

    print("CI training done.")


if __name__ == "__main__":
    main()
