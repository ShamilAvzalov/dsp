INPUT_DATA = "***.pkl" # данные для обучения
FILL_VALUES = "***.pkl" # файл для заполнения пропусков
FEATURE_VOCABS = "***.csv" # словарь фичей
PROVIDER_ENCODER = "***.pkl" # label encoder обученный
BEST_MODEL = "***.keras" # название для выгрузки лучшей модели
PROVIDER_EMBEDDINGS = "***.csv" # файл с эмбеддингами провайдеров
# -------------------------------
# Шаг 1. Загрузка данных и создание DataFrame с именами столбцов
# -------------------------------
columns = ["***", "***", "***", "***", "***", "***",
           "***", "***", "***", "***", "***", "***", "***"]

try:
    with open(INPUT_DATA, 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    print(f"Ошибка: Файл '{INPUT_DATA}' не найден. Убедитесь, что он находится в правильной директории.")
    exit()
except Exception as e:
    print(f"Ошибка при загрузке '{INPUT_DATA}': {e}")
    exit()


# Преобразуем в DataFrame с заданными именами столбцов
df = pd.DataFrame(data, columns=columns)
print("Исходные данные (первые 5 строк):")
print(df.head())
print(f"\nРазмер DataFrame: {df.shape}")

# Опционально: Проверка количества уникальных значений и самих значений (может быть долго для больших данных)
print("\nКоличество уникальных значений в каждом столбце:")
print(df.nunique())
print("\nУникальные значения для некоторых столбцов (пример):")
for col in ["***", "***", "***", "***"]:
    if col in df.columns:
        unique_values = df[col].unique()
        print(f"Столбец {col} (до {10} уникальных): {unique_values[:10]}")
# -------------------------------
# Шаг 2. Удаляем лишние столбцы
# -------------------------------
columns_to_drop = []
if "***" in df.columns:
    columns_to_drop.append("***")

if columns_to_drop:
    df.drop(columns_to_drop, axis=1, inplace=True)
    print(f"\nДанные после удаления столбцов: {columns_to_drop}")
    print(df.head())

# Определяем входные признаки и целевую переменную
label_column = "***"
if label_column not in df.columns:
    print(f"Ошибка: Целевой столбец '{label_column}' отсутствует в данных после удаления.")
    exit()

input_feature_columns = [col for col in df.columns if col != label_column]
print("\nВходные признаки:", input_feature_columns)
print("Целевая переменная:", label_column)
# -------------------------------
# Шаг 3. Замена "Unknown" и обработка пропусков (NaN)
# -------------------------------
fill_values = {}  # сюда будем записывать, чем заполнять пропуски в каждом столбце

print("\nОбработка 'Unknown' и NaN:")
for col in input_feature_columns:
    # Заменяем строковый "Unknown" на NaN
    if df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
        df[col] = df[col].replace(["Unknown", "unknown", ""], np.nan)

    # Если в столбце есть NaN, заполним их найденным значением
    if df[col].isnull().any():
        mode_val = df[col].mode()
        if not mode_val.empty:
            fill_value = mode_val[0]
            print(f"- Столбец '{col}': NaN заменяем на моду '{fill_value}'")
        else:
            fill_value = "missing"
            print(f"- Столбец '{col}': NaN заменяем на '{fill_value}' (мода не найдена)")

        # Запоминаем, чем заполнять этот столбец
        fill_values[col] = fill_value

        # Заполняем пропуски
        df[col].fillna(fill_value, inplace=True)
    else:
        # Если в этом столбце не было пропусков и не нужно ничего заменять
        # можно запоминать None или сам столбец (как угодно)
        fill_values[col] = None

# Убедимся, что в целевом столбце нет пропусков
if df[label_column].isnull().any():
    print(f"\nВнимание: Обнаружены пропуски в целевом столбце '{label_column}'. Удаляем строки с пропусками.")
    df.dropna(subset=[label_column], inplace=True)
    print(f"Новый размер DataFrame после удаления строк с NaN в '{label_column}': {df.shape}")

print("\nДанные после обработки 'Unknown' и NaN (первые 5 строк):")
print(df.head())

# Теперь сохраняем fill_values в pickle, чтобы в будущем не пересчитывать
pickle_filename = FILL_VALUES
with open(pickle_filename, "wb") as f:
    pickle.dump(fill_values, f)

print(f"\nСловарь значений для fillna() сохранён в '{pickle_filename}':")
print(fill_values)
# -------------------------------
# Шаг 4. Label Encoding для входных признаков и сохранение словарей
# -------------------------------
n_features = len(input_feature_columns)
encoded_features = np.zeros((df.shape[0], n_features), dtype=np.int32)
feature_vocabs = {} # Словарь для хранения энкодеров и размеров словарей

print("\nКодирование входных признаков:")
for i, col in enumerate(input_feature_columns):
    print(f"- Кодирование колонки: {col}")
    le = LabelEncoder()
    # Применяем fit_transform к данным из DataFrame
    encoded_features[:, i] = le.fit_transform(df[col].astype(str))
    # Сохраняем сам энкодер и размер словаря (количество уникальных классов)
    feature_vocabs[col] = {'encoder': le, 'vocab_size': len(le.classes_)}
    print(f"  > Найдено уникальных значений: {len(le.classes_)}")

# Сохраняем словари признаков
with open(FEATURE_VOCABS, "wb") as f:
    pickle.dump(feature_vocabs, f)
print(f"\nСловари признаков сохранены в '{FEATURE_VOCABS}'.")

# ВАЖНО: X теперь - это НЕнормализованные закодированные признаки!
# Нормализация здесь не нужна и вредна перед Embedding слоями.
X_encoded = encoded_features
print("Форма массива закодированных входных признаков X_encoded:", X_encoded.shape)
# -------------------------------
# Шаг 5. Label Encoding для меток DSP провайдеров и сохранение энкодера
# -------------------------------
provider_encoder = LabelEncoder()
y = provider_encoder.fit_transform(df[label_column])
num_providers = len(provider_encoder.classes_)

with open(PROVIDER_ENCODER, "wb") as f:
    pickle.dump(provider_encoder, f)

print(f"\nНайдено {num_providers} уникальных провайдеров.")
print(f"LabelEncoder для провайдеров сохранён в '{PROVIDER_ENCODER}'.")
print("Форма массива меток y:", y.shape)
# -------------------------------
# Шаг 6. Опеределение пользовательских слоёв
# -------------------------------
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
        # inputs = (batch_size, embedding_dim)
        # возвращаем (batch_size, num_providers)
        return tf.matmul(inputs, self.provider_embedding_matrix, transpose_b=True)
# -------------------------------
# Шаг 7. Определение архитектуры модели
# -------------------------------
# Гиперпараметры
feature_embedding_dim = 11   # Размерность эмбеддинга для каждого категориального признака
request_embedding_dim = 11   # Финальная размерность эмбеддинга запроса (и провайдеров)
hidden_dense_units_1 = 22    # Нейроны в первом скрытом слое
dropout_rate = 0.3           # Доля нейронов для Dropout (регуляризация)

# --- Построение модели с использованием Functional API и Embedding слоев ---
input_layers = []
embedding_layers = []

# Создаем отдельный Input и Embedding слой для каждого признака
print("\nСоздание слоев модели:")
for i, col in enumerate(input_feature_columns):
    vocab_size = feature_vocabs[col]['vocab_size']
    print(f"- Создание Input и Embedding для '{col}' (vocab_size={vocab_size})")
    # Вход для i-го признака (принимает один integer ID)
    feature_input = keras.Input(shape=(1,), name=f'input_{col}', dtype='int32')
    input_layers.append(feature_input)

    # Embedding слой для i-го признака
    feature_embedding = layers.Embedding(
        input_dim=vocab_size,             # Размер словаря для этого признака
        output_dim=feature_embedding_dim, # Гиперпараметр
        name=f'embedding_{col}',
        mask_zero=False                   # False, если 0 - валидная категория
    )(feature_input)
    # Убираем измерение '1' после Embedding слоя -> (batch_size, feature_embedding_dim)
    feature_embedding = layers.Flatten(name=f'flatten_{col}')(feature_embedding)
    embedding_layers.append(feature_embedding)

# Объединяем все эмбеддинги признаков в один вектор
if len(embedding_layers) > 1:
    concatenated_embeddings = layers.Concatenate(name='concatenate_features')(embedding_layers)
    print(f"- Конкатенация {len(embedding_layers)} эмбеддингов признаков.")
else:
    concatenated_embeddings = embedding_layers[0]  # Если только один признак

# Скрытый Dense-слой для обработки объединенных признаков
print(f"- Добавление Dense слоя ({hidden_dense_units_1}) и Dropout ({dropout_rate}).")
hidden_layer = layers.Dense(hidden_dense_units_1, activation='relu', name='hidden_dense_1')(concatenated_embeddings)
hidden_layer = layers.Dropout(dropout_rate, name='dropout_1')(hidden_layer)

# Финальный слой для создания эмбеддинга запроса
print(f"- Создание слоя эмбеддинга запроса (размерность {request_embedding_dim}).")
request_embedding_layer = layers.Dense(request_embedding_dim, activation=None, name='request_embedding')(hidden_layer)

# Нормализация эмбеддинга запроса
normalized_request_embedding = L2NormalizeLayer(name='norm_request_embedding')(request_embedding_layer)
print("- Добавление L2-нормализации эмбеддинга запроса.")

# Создаём слой эмбеддингов провайдеров и вычисляем logits
provider_layer = ProviderEmbeddingLayer(num_providers, request_embedding_dim, name='provider_layer')
logits = provider_layer(normalized_request_embedding)
print("- Добавление слоя вычисления логитов (скалярное произведение).")

# Собираем финальную модель
model = keras.Model(inputs=input_layers, outputs=logits)

# Компиляция модели с метрикой Top-5 Accuracy
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  
    loss='sparse_categorical_crossentropy',                
    metrics=[
        'accuracy',
        SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ]
)

print("\nИтоговая архитектура модели:")
model.summary()

# Callback для сохранения лучшей модели
checkpoint_filepath = BEST_MODEL
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_top_5_accuracy',   # Следим за top-5 точностью на валидац. выборке
    save_best_only=True,           # Сохраняем только лучшую модель
    mode='max',                    # 'max', так как точность нужно максимизировать
    save_weights_only=False,       # Сохраняем всю модель (архитектуру + веса + оптимизатор)
    verbose=1                     
)
print(f"\nНастроено сохранение лучшей модели по метрике 'val_top_5_accuracy' в файл: {checkpoint_filepath}")
# -------------------------------
# Шаг 7. Обучение модели
# -------------------------------
epochs = 10
batch_size = 512

# --- Подготовка данных для обучения ---
X_train_list = [X_encoded[:, i] for i in range(n_features)]

print(f"\nНачало обучения модели ({epochs} эпох, batch_size={batch_size})...")
history = model.fit(
    X_train_list, # Используем весь датасет для обучения (или X_train_list_split)
    y,            # (или y_train)
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.15, # Простой способ выделить валидационную выборку
    shuffle=True,
    callbacks=[model_checkpoint_callback]
)
print("Обучение завершено.")
# -------------------------------
# Шаг 8. Извлечение и сохранение обученных эмбеддингов провайдеров в CSV
# -------------------------------

# 1) Получаем слой по имени
provider_layer = model.get_layer(name='provider_layer')

# 2) Извлекаем обученные эмбеддинги провайдеров из trainable_weights (или напрямую через атрибут)
learned_provider_embeddings = provider_layer.provider_embedding_matrix.numpy()

# 3) Сопоставляем провайдеров (provider_encoder.classes_) с извлечёнными векторами
provider_classes = provider_encoder.classes_
columns_out = ["***"] + [f"***_{i+1}" for i in range(request_embedding_dim)]

data_out = []
for i, vec in enumerate(learned_provider_embeddings):
    row = [provider_classes[i]] + vec.tolist()  # Используем исходные имена классов
    data_out.append(row)

df_out = pd.DataFrame(data_out, columns=columns_out)
csv_filename = PROVIDER_EMBEDDINGS

try:
    df_out.to_csv(csv_filename, index=False)
    print(f"\nОбученные эмбеддинги провайдеров сохранены в файл '{csv_filename}'.")
    print(df_out.head())
except Exception as e:
    print(f"\nОшибка при сохранении эмбеддингов в CSV: {e}")
