import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

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


def save_confmat(cm, classes, out_path: Path):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


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
    # panggil preprocess script dengan arg-style
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

    # --- MLflow online
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow.environments._mlflow_env.set_env_vars(
        {
            "MLFLOW_TRACKING_USERNAME": args.dagshub_user,
            "MLFLOW_TRACKING_PASSWORD": args.dagshub_token,
        }
    )

    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)

    with mlflow.start_run(run_name="ci_train"):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_classes": num_classes,
            }
        )

        model = build_model(num_classes)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
        )

        # log per-epoch metrics
        for i in range(args.epochs):
            mlflow.log_metric("loss", float(history.history["loss"][i]), step=i)
            mlflow.log_metric("accuracy", float(history.history["accuracy"][i]), step=i)
            mlflow.log_metric("val_loss", float(history.history["val_loss"][i]), step=i)
            mlflow.log_metric(
                "val_accuracy", float(history.history["val_accuracy"][i]), step=i
            )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric("test_loss", float(loss))
        mlflow.log_metric("test_accuracy", float(acc))

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=classes)

        cm_path = outputs / "confusion_matrix.png"
        rep_path = outputs / "classification_report.txt"
        save_confmat(cm, classes, cm_path)
        rep_path.write_text(report, encoding="utf-8")

        # artifacts
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(rep_path))
        mlflow.log_artifact(str(data_dir / "label_map.json"))

        # model artifact
        mlflow.tensorflow.log_model(model, artifact_path="model")

    print("CI training done.")


if __name__ == "__main__":
    main()
