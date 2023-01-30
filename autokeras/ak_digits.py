from sklearn.datasets import load_digits
import autokeras as ak
import tensorflow as tf
from kerastuner import Objective
import tensorflow_addons as tfa


def load_data():
    data, target = load_digits(return_X_y=True, as_frame=False)
    data = data.reshape(-1, 8, 8, 1)
    target = target.reshape(-1, 1)
    return data, target


def train_classifier(data, target):
    clf = ak.ImageClassifier(
        num_classes=10,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(name="sparse_loss"),
        metrics=[
            tfa.metrics.MatthewsCorrelationCoefficient(num_classes=10, name="mcc")
        ],
        overwrite=True,
        max_trials=10,
        objective=Objective(name="val_loss", direction="min"),
    )

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=3, restore_best_weights=True, verbose=1
    )
    checkpoint_filepath = "./keras_models_checkpoint"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor="val_mcc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    clf.fit(
        data,
        target,
        epochs=10,
        batch_size=32,
        workers=-1,
        use_multiprocessing=True,
        callbacks=[early_stop_callback, model_checkpoint_callback],
    )
    model = clf.export_model()
    try:
        model.save("model_autokeras_digits", save_format="tf")
    except Exception:
        model.save("model_autokeras_digits.h5")


def main():
    data, target = load_data()
    train_classifier(data, target)


if __name__ == "__main__":
    main()
