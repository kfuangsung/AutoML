import pandas as pd
import openml
import autokeras as ak
from keras_tuner import Objective
import tensorflow as tf
import tensorflow_addons as tfa


def load_data(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, _, _ = dataset.get_data(target="RS", dataset_format="dataframe")
    return X, y


def load_synthtic_data(file_path):
    synth_data = pd.read_csv(file_path)
    synth_label = synth_data.pop("RS")
    return synth_data, synth_label


def get_model(max_trials=100):
    model = ak.StructuredDataRegressor(
        loss="mean_squared_error",
        metrics=[tfa.metrics.RSquare(name="r_square")],
        max_trials=max_trials,
        objective=Objective("val_loss", "min"),
        overwrite=True,
    )
    return model


def train_model(
    model,
    X,
    y,
    val_data=None,
    val_split=0.2,
    epochs=1000,
    batch_size=64,
    add_callback=True,
):
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=10,
        verbose=1,
        restore_best_weights=True,
    )
    # checkpoint_filepath = "./keras_models/moneyball_checkpoint"
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=False,
    #     monitor="val_loss",
    #     mode="min",
    #     save_best_only=True,
    #     verbose=1,
    # )
    if add_callback:
        callbacks = [early_stop_callback]
    else:
        callbacks = None

    model.fit(
        X,
        y,
        epochs=epochs,
        validation_split=val_split,
        validation_data=val_data,
        callbacks=callbacks,
        batch_size=batch_size,
        verbose=1,
        workers=-1,
        use_multiprocessing=True,
    )

    return model


def save_model(model, save_name):
    model_to_save = model.export_model()
    try:
        model_to_save.save(save_name, save_format="tf")
    except Exception:
        model_to_save.save(f"{save_name}.h5")
    print(f"{save_name} is saved")


def main():
    id = 41021  # moneyball
    max_trials = 10
    val_split = 0
    epochs = int(1e2)
    batch_size = 64
    add_callback = True
    save_name = "./keras_models/ak_model_moneyball"
    syth_path = "synthetic_moneyball.csv"

    print("Loading real data")
    X, y = load_data(id)
    print(X.shape, y.shape)
    print("Reading synthetic data")
    X_synth, y_synth = load_synthtic_data(syth_path)
    print(X_synth.shape, y_synth.shape)
    val_data = (X, y)
    model = get_model(max_trials)
    print("Begin training model")
    model = train_model(
        model=model,
        X=X_synth,
        y=y_synth,
        val_data=val_data,
        val_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        add_callback=add_callback,
    )
    print("Saving training model")
    save_model(model, save_name)


if __name__ == "__main__":
    main()
