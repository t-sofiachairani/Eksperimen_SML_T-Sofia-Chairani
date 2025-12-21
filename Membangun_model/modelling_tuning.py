import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def load_npz(path: Path):
    d = np.load(path)
    return d["X"], d["y"]


def build_model(num_classes: int, lr: float, fine_tune: bool):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
    )
    base.trainable = fine_tune

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
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


def save_history_plot(history, out_path: Path):
    fig = plt.figure()
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["accuracy"], label="acc")
    plt.plot(history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.title("Training Curves")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--epochs", default=3, type=int)
    ap.add_argument("--batch_size", default=16, type=int)

    # DagsHub / MLflow online
    ap.add_argument("--tracking_uri", required=True, type=str)
    ap.add_argument("--experiment_name", default="kue-indonesia", type=str)
    ap.add_argument("--username", required=True, type=str)  # username dagshub
    ap.add_argument("--token", required=True, type=str)  # token dagshub
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    # load classes
    with open(data_dir / "label_map.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    num_classes = len(classes)

    X_train, y_train = load_npz(data_dir / "train.npz")
    X_val, y_val = load_npz(data_dir / "validation.npz")
    X_test, y_test = load_npz(data_dir / "test.npz")

    # auth basic (DagsHub pakai username + token)
    import os

    os.environ["MLFLOW_TRACKING_USERNAME"] = args.username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = args.token

    # setup MLflow online
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # simple hyperparameter tuning (WAJIB skilled/advanced)
    grid = [
        {"lr": 3e-4, "fine_tune": False},
        {"lr": 1e-4, "fine_tune": True},
    ]

    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)

    with mlflow.start_run(run_name="tuning_parent"):
        # params global
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_classes": num_classes,
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_val)),
                "test_samples": int(len(y_test)),
            }
        )

        for cfg in grid:
            run_name = f"lr={cfg['lr']}_fine_tune={cfg['fine_tune']}"
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.log_params(cfg)

                model = build_model(num_classes, cfg["lr"], cfg["fine_tune"])

                hist = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    verbose=1,
                ).history

                # manual metrics per epoch (ini inti ADVANCED)
                for i in range(args.epochs):
                    mlflow.log_metric("loss", float(hist["loss"][i]), step=i)
                    mlflow.log_metric("accuracy", float(hist["accuracy"][i]), step=i)
                    mlflow.log_metric("val_loss", float(hist["val_loss"][i]), step=i)
                    mlflow.log_metric(
                        "val_accuracy", float(hist["val_accuracy"][i]), step=i
                    )

                # test metrics (extra beyond basic)
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                mlflow.log_metric("test_loss", float(test_loss))
                mlflow.log_metric("test_accuracy", float(test_acc))

                # predictions
                y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

                # extra metric (di luar autolog umum): macro_f1 dari report
                report_dict = classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0
                )
                mlflow.log_metric(
                    "macro_f1", float(report_dict["macro avg"]["f1-score"])
                )

                # ====== EXTRA ARTIFACTS (minimal 2) ======
                cm = confusion_matrix(y_test, y_pred)
                cm_path = outputs / "confusion_matrix.png"
                save_confmat(cm, classes, cm_path)

                report_path = outputs / "classification_report.txt"
                report_path.write_text(
                    classification_report(
                        y_test, y_pred, target_names=classes, zero_division=0
                    ),
                    encoding="utf-8",
                )

                curve_path = outputs / "training_curves.png"
                save_history_plot(hist, curve_path)

                # log artifacts
                mlflow.log_artifact(str(cm_path))
                mlflow.log_artifact(str(report_path))
                mlflow.log_artifact(str(curve_path))
                mlflow.log_artifact(str(data_dir / "label_map.json"))

                # log model
                mlflow.tensorflow.log_model(model, artifact_path="model")

    print("DONE. Check DagsHub MLflow UI.")


if __name__ == "__main__":
    main()
