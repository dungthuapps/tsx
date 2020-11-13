import os
import numpy as np
import pandas as pd

from tsx.perturbation import TimeSeriesPerturbation
from tsx.xai.lime import LIMETimeSeries

DATA_DIR = "tests/data" if os.path.isdir("tests") else "data"
mts = np.array([np.arange(1, 9), np.arange(2, 10)])
uts = np.arange(1, 9).reshape(1, -1)


def test_perturbation_time_series():
    with TimeSeriesPerturbation(window_size=3, replacement_method="local_mean") as t:
        samples = list(t.perturb(mts, n_samples=5))
        z_prime, z, pi_z = samples[0]
        assert z.shape == mts.shape
        assert len(z_prime) == len(t.labels)
        assert pi_z >= 0

        # Custom replacement
        replacements = 99
        samples = list(t.perturb(mts, n_samples=5, replacements=replacements))
        z_prime, z, pi_z = samples[0]
        if 0 in z_prime:
            assert replacements in z


def test_lime_linear():
    ts_lime = LIMETimeSeries(window_size=2, sample_size=100)
    ts_lime.explain(mts, predict_fn=lambda x: np.random.randint(100))
    xai_model = ts_lime.xai_estimator
    assert len(xai_model.coef_) == len(ts_lime._perturbator.labels)


def load_data_set_bejin():
    data_link = f"{DATA_DIR}/pollution.csv"
    df = pd.read_csv(data_link)

    # Set date-time as index
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df


def test_lime_with_two_models():
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    # Prepare data set
    df = load_data_set_bejin()

    # Encoding wind_direction to integer
    encoder = LabelEncoder()
    df["wind_direction"] = encoder.fit_transform(df["wind_direction"])

    # Normalization
    independents = ["dew", "temp", "press", "wind_direction", "wind_speed", "snow", "rain"]
    dependent = "pollution"
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler.fit(df[independents].values)
    y_scaler.fit(df[dependent].values.reshape(-1, 1))

    # Prepare predict function
    wavenet = tf.keras.models.load_model(f"{DATA_DIR}/wavenet_mts_128_1.h5")
    lstm = tf.keras.models.load_model(f"{DATA_DIR}/lstm_mts_128_1.h5")

    def predict_fn(z, model):
        _, steps, features = model.input.shape
        z = z.reshape((1, steps, features))
        z_hat = model.predict(z)
        z_hat = y_scaler.inverse_transform(z_hat.reshape(-1, 1))  # to avoid zero coef_ for z_hat in[0, 1]
        z_hat = z_hat.ravel()
        return z_hat[0]

    def lstm_predict_fn(z):
        return predict_fn(z, model=lstm)

    def wavenet_predict_fn(z):
        return predict_fn(z, model=wavenet)

    # 1- Load an instance
    idx = 100
    x = df[idx:idx + 128].copy()    # 128 steps
    x[independents] = x_scaler.transform(x[independents].values)
    ts_x = x[independents].values.reshape(7, 128)   # 7 features, 128 steps

    # 2- Choose XAI model
    #   Here - with LIME for Time Series (Perturbation)
    #       - XAI estimator in default is Lasso(alpha=0.5)
    ts_lime = LIMETimeSeries(window_size=4, sample_size=100, perturb_method='zeros')

    ts_lime.explain(ts_x, predict_fn=lstm_predict_fn)
    lstm_coef = ts_lime.xai_estimator.coef_

    ts_lime.explain(ts_x, predict_fn=wavenet_predict_fn)
    wavenet_coef = ts_lime.xai_estimator.coef_

    assert lstm_coef.shape == wavenet_coef.shape
    assert (wavenet_coef == lstm_coef).sum() != 0

    # Difference of two models
    from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
    x1 = wavenet_coef.reshape(1, -1)
    x2 = lstm_coef.reshape(1, -1)
    d = euclidean_distances(x1, x2)
    s = cosine_similarity(x1, x2)

    assert d > 0
    assert s < 1.0
    z_prime, perturbed_z, prediction, sample_weight = ts_lime.get_a_local_sample()
