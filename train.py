import os
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy

# Конфигурационные параметры
INPUT_DATA = "***.pkl"  # путь к исходным данным
FILL_VALUES = "***.pkl"  # файл для сохранения значений заполнения
FEATURE_VOCABS = "***.csv"  # словарь фичей
PROVIDER_ENCODER = "***.pkl"  # encoder для меток
BEST_MODEL = "***.keras"  # файл для лучшей модели
PROVIDER_EMBEDDINGS = "***.csv"  # файл с эмбеддингами

# Гиперпараметры модели
FEATURE_EMBEDDING_DIM = 11
REQUEST_EMBEDDING_DIM = 11
HIDDEN_DENSE_UNITS_1 = 22
DROPOUT_RATE = 0.3

# Параметры обучения
EPOCHS = 10
BATCH_SIZE = 512
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 0.001

# Кастомные слои модели
class L2NormalizeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

class ProviderEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, num_providers, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_providers = num_providers
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.provider_embedding_matrix = self.add_weight(
            shape=(self.num_providers, self.embedding_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
            name='provider_embedding_matrix'
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.provider_embedding_matrix, transpose_b=True)

def main():
    # Создаем необходимые директории
    os.makedirs(os.path.dirname(INPUT_DATA), exist_ok=True)
    os.makedirs(os.path.dirname(BEST_MODEL), exist_ok=True)
    
    # -------------------------------
    # Шаг 1. Загрузка данных
    # -------------------------------
    columns = ["***", "***", "***", "***", "***", "***", 
               "***", "***", "***", "***", "***", "***", "***"]

    try:
        with open(INPUT_DATA, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{INPUT_DATA}' не найден.")
        return
    except Exception as e:
        print(f"Ошибка при загрузке '{INPUT_DATA}': {e}")
        return

    df = pd.DataFrame(data, columns=columns)
    print("Данные загружены. Размер:", df.shape)

    # -------------------------------
    # Шаг 2. Предобработка данных
    # -------------------------------
    # Удаление лишних столбцов
    columns_to_drop = []
    if "***" in df.columns:
        columns_to_drop.append("***")

    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)
    
    # Определение целевой переменной
    label_column = "***"
    if label_column not in df.columns:
        print(f"Ошибка: Целевой столбец '{label_column}' отсутствует.")
        return
    
    input_feature_columns = [col for col in df.columns if col != label_column]
    
    # Обработка пропусков
    fill_values = {}
    for col in input_feature_columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(["Unknown", "unknown", ""], np.nan)
        
        if df[col].isnull().any():
            mode_val = df[col].mode()
            fill_value = mode_val[0] if not mode_val.empty else "missing"
            df[col].fillna(fill_value, inplace=True)
            fill_values[col] = fill_value
    
    # Сохранение значений для заполнения
    with open(FILL_VALUES, "wb") as f:
        pickle.dump(fill_values, f)
    
    # -------------------------------
    # Шаг 3. Кодирование признаков
    # -------------------------------
    # Кодирование входных признаков
    n_features = len(input_feature_columns)
    encoded_features = np.zeros((df.shape[0], n_features), dtype=np.int32)
    feature_vocabs = {}

    for i, col in enumerate(input_feature_columns):
        le = LabelEncoder()
        encoded_features[:, i] = le.fit_transform(df[col].astype(str))
        feature_vocabs[col] = {'encoder': le, 'vocab_size': len(le.classes_)}
    
    X_encoded = encoded_features
    
    # Кодирование целевой переменной
    provider_encoder = LabelEncoder()
    y = provider_encoder.fit_transform(df[label_column])
    num_providers = len(provider_encoder.classes_)
    
    with open(PROVIDER_ENCODER, "wb") as f:
        pickle.dump(provider_encoder, f)
    
    # -------------------------------
    # Шаг 4. Построение модели
    # -------------------------------
    input_layers = []
    embedding_layers = []
    
    for i, col in enumerate(input_feature_columns):
        vocab_size = feature_vocabs[col]['vocab_size']
        feature_input = keras.Input(shape=(1,), name=f'input_{col}', dtype='int32')
        input_layers.append(feature_input)
        
        feature_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=FEATURE_EMBEDDING_DIM,
            name=f'embedding_{col}'
        )(feature_input)
        feature_embedding = layers.Flatten(name=f'flatten_{col}')(feature_embedding)
        embedding_layers.append(feature_embedding)
    
    if len(embedding_layers) > 1:
        concatenated_embeddings = layers.Concatenate(name='concatenate_features')(embedding_layers)
    else:
        concatenated_embeddings = embedding_layers[0]
    
    hidden_layer = layers.Dense(HIDDEN_DENSE_UNITS_1, activation='relu', name='hidden_dense_1')(concatenated_embeddings)
    hidden_layer = layers.Dropout(DROPOUT_RATE, name='dropout_1')(hidden_layer)
    request_embedding_layer = layers.Dense(REQUEST_EMBEDDING_DIM, activation=None, name='request_embedding')(hidden_layer)
    normalized_request_embedding = L2NormalizeLayer(name='norm_request_embedding')(request_embedding_layer)
    
    provider_layer = ProviderEmbeddingLayer(num_providers, REQUEST_EMBEDDING_DIM, name='provider_layer')
    logits = provider_layer(normalized_request_embedding)
    
    model = keras.Model(inputs=input_layers, outputs=logits)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    # -------------------------------
    # Шаг 5. Обучение модели
    # -------------------------------
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL,
        monitor='val_top_5_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    X_train_list = [X_encoded[:, i] for i in range(n_features)]
    
    history = model.fit(
        X_train_list,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpoint_callback]
    )
    
    # -------------------------------
    # Шаг 6. Сохранение эмбеддингов
    # -------------------------------
    provider_layer = model.get_layer(name='provider_layer')
    learned_provider_embeddings = provider_layer.provider_embedding_matrix.numpy()
    
    provider_classes = provider_encoder.classes_
    columns_out = ["***"] + [f"***_{i+1}" for i in range(REQUEST_EMBEDDING_DIM)]
    
    df_out = pd.DataFrame(
        [[provider_classes[i]] + vec.tolist() for i, vec in enumerate(learned_provider_embeddings)],
        columns=columns_out
    )
    
    df_out.to_csv(PROVIDER_EMBEDDINGS, index=False)
    print(f"Эмбеддинги сохранены в '{PROVIDER_EMBEDDINGS}'")

if __name__ == "__main__":
    # Парсер аргументов для настройки параметров
    parser = argparse.ArgumentParser(description='Обучение модели рекомендации DSP-провайдеров')
    parser.add_argument('--input', type=str, default=INPUT_DATA, help='Путь к входным данным')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Количество эпох обучения')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Размер батча')
    args = parser.parse_args()
    
    # Обновление параметров
    INPUT_DATA = args.input
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    
    print(f"Запуск обучения с параметрами:")
    print(f" - Входные данные: {INPUT_DATA}")
    print(f" - Эпохи: {EPOCHS}")
    print(f" - Размер батча: {BATCH_SIZE}")
    
    main()
    print("Обучение завершено успешно!")