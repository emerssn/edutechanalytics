import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col

warnings.filterwarnings('ignore')

print("="*50)
print("Parte 1: Análisis Exploratorio y Visualización (3 puntos)")
print("="*50)

# ==========================================
# 1.1  Exploración Inicial con Pandas y NumPy (1 punto)
# ==========================================
print("\n--- 1.1 Exploración Inicial con Pandas y NumPy ---")
# Cargamos el primer csv que encontremos en la carpeta
csv_file = [f for f in os.listdir('.') if f.endswith('.csv')][0]
df = pd.read_csv(csv_file)
# Mostrar las primeras 10 filas y estadísticas descriptivas de variables numéricas
print("\nPrimeras 10 filas:")
print(df.head(10))

print("\nEstadísticas descriptivas (variables numéricas):")
print(df.describe())
# Mostrar valores nulos originales
print("\nValores nulos originales:")
print(df.isnull().sum())

# Llenamos nulos: usamos la mediana para los números y la moda para las categorías
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

print("\nValores nulos tratados:")
print(df.isnull().sum())

# Correlación usando NumPy
print("\nMatriz de correlación usando NumPy:")
cor_matrix = np.corrcoef(df[num_cols].values, rowvar=False)
df_corr_numpy = pd.DataFrame(cor_matrix, columns=num_cols, index=num_cols)
print(df_corr_numpy)

# Buscamos outliers usando (IQR)
def contar_outliers(serie):
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    return len(serie[(serie < lim_inf) | (serie > lim_sup)])

if 'horas_estudio_semanal' in df.columns and 'monto_inversion' in df.columns:
    print(f"\nOutliers en horas_estudio_semanal: {contar_outliers(df['horas_estudio_semanal'])}")
    print(f"Outliers en monto_inversion: {contar_outliers(df['monto_inversion'])}")

# ==========================================
# 1.2  Visualización Avanzada con Seaborn (1 punto)
# ==========================================
print("\n--- 1.2 Visualización Avanzada con Seaborn ---")
# Pairplot
cols_pair = ['horas_estudio_semanal', 'participacion_foros', 'promedio_evaluaciones', 'puntaje_satisfaccion']
cols_exist = [c for c in cols_pair if c in df.columns]
if 'curso_completado' in df.columns and len(cols_exist) > 0:
    sns.pairplot(df, vars=cols_exist, hue='curso_completado', palette='Set2')
    plt.savefig('pairplot_analisis.png')
    plt.close()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.savefig('heatmap_correlacion.png')
plt.close()

# Violinplot
if 'nivel_educacion' in df.columns and 'puntaje_satisfaccion' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='nivel_educacion', y='puntaje_satisfaccion', palette='pastel')
    plt.title("Distribución de Satisfacción por Nivel de Educación")
    plt.tight_layout()
    plt.savefig('violinplot_satisfaccion.png')
    plt.close()

print("Gráficos generados y guardados.")

# ==========================================
# 1.3 Personalización con Matplotlib (1 punto)
# ==========================================
print("\n--- 1.3 Personalización con Matplotlib ---")
if 'horario_estudio' in df.columns and 'puntaje_satisfaccion' in df.columns:
    # Promedio por horario
    df_horario = df.groupby('horario_estudio')['puntaje_satisfaccion'].mean().reset_index()

    plt.figure(figsize=(8, 6))
    colores = sns.color_palette("husl", len(df_horario))
    barras = plt.bar(df_horario['horario_estudio'], df_horario['puntaje_satisfaccion'], 
                     color=colores, edgecolor='black', linewidth=1.5)

    plt.title("Promedio de Puntaje de Satisfacción por Horario de Estudio", fontsize=14)
    plt.xlabel("Horario de Estudio", fontsize=12)
    plt.ylabel("Puntaje Promedio", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Anotaciones exactas
    for b in barras:
        yval = b.get_height()
        plt.text(b.get_x() + b.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    plt.ylim(0, df_horario['puntaje_satisfaccion'].max() * 1.15)
    plt.tight_layout()
    plt.savefig('satisfaccion_horario.png')
    plt.close()
    
print("Gráfico 'satisfaccion_horario.png' guardado.")

print("\n" + "="*50)
print("Parte 2: Preparación de Datos y Feature Engineering (2 puntos)")
print("="*50)
# Transformación de Variables (1 punto)
print("\n--- 2.1 Transformación de Variables ---")
# Crear nuevas variables
df['tasa_completitud'] = (df['tareas_completadas'] / df['tareas_totales']) * 100
df['estudiante_activo'] = np.where(df['sesiones_totales'] > 50, 1, 0)
# Codifica variables categóricas usando LabelEncoder para variables ordinales y OneHotEncoder para nominales
# Botamos columnas de ID y nombres que no sirven para predecir
df_model = df.drop(columns=['id_estudiante', 'nombre'])

# Definir las categorías
nominales = ['genero', 'pais', 'dispositivo_principal', 'horario_estudio', 'plataforma_origen']
ordinales = ['nivel_educacion', 'experiencia_previa']
numericas = df_model.select_dtypes(include=[np.number]).columns.drop(['curso_completado', 'puntaje_satisfaccion', 'estudiante_activo'])

# LabelEncoder para variables ordinales
le = LabelEncoder()
for col in ordinales:
    df_model[col] = le.fit_transform(df_model[col])

df_model = pd.get_dummies(df_model, columns=nominales, drop_first=True)

# Normaliza las variables numéricas usando StandardScaler
scaler = StandardScaler()
df_model[numericas] = scaler.fit_transform(df_model[numericas])

print("Nuevas variables creadas.")
print("Variables categóricas codificadas (LabelEncoder y OneHotEncoding).")
print("Variables numéricas normalizadas (StandardScaler).")
# División y Preparación (1 punto)
print("\n--- 2.2 División y Preparación ---")
# Separamos nuestras variables predictoras (X) de los objetivos (y)
X = df_model.drop(columns=['curso_completado', 'puntaje_satisfaccion'])
y_clasificacion = df_model['curso_completado']
y_regresion = df_model['puntaje_satisfaccion']

# Dividir 70/30
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_clasificacion, y_regresion, test_size=0.3, random_state=42
)
# Aplicar SMOTE por desbalanceo en clasificación    
print(f"Dimensiones de entrenamiento (X): {X_train.shape}")
print(f"Dimensiones de prueba (X): {X_test.shape}")
print("\nAplicando SMOTE para balancear clases (curso_completado):")
print(f"Distribución Original en Entrenamiento:\n{y_class_train.value_counts().to_string()}")

smote = SMOTE(random_state=42)
X_train_class_res, y_class_train_res = smote.fit_resample(X_train, y_class_train)

print(f"Distribución Post-SMOTE:\n{y_class_train_res.value_counts().to_string()}")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Modelado Predictivo con Machine Learning (3 puntos)
print("\n" + "="*50)
print("Parte 3: Modelado Predictivo con Machine Learning (3 puntos)")
print("="*50)

# ==========================================
# 3.1 Modelo de Clasificación (1.5 puntos)
# ==========================================
print("\n--- 3.1 Modelo de Clasificación ---")
# Entrena dos modelos: RandomForestClassifier y LogisticRegression
# Utiliza GridSearchCV para optimizar hiperparámetros
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_clf = RandomForestClassifier(random_state=42)
grid_rf_clf = GridSearchCV(rf_clf, param_grid_rf, cv=5, scoring='accuracy')
grid_rf_clf.fit(X_train_class_res, y_class_train_res)

param_grid_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}
lr_clf = LogisticRegression(solver='liblinear', random_state=42)
# Grid Search para obtener los mejores hiperparámetros
grid_lr_clf = GridSearchCV(lr_clf, param_grid_lr, cv=5, scoring='accuracy')
grid_lr_clf.fit(X_train_class_res, y_class_train_res)
#Imprime los mejores hiperparámetros encontrados para cada modelo.
print("Mejores hiperparámetros encontrados:")
print(f"RandomForest: {grid_rf_clf.best_params_}")
print(f"LogisticRegression: {grid_lr_clf.best_params_}")

# Evaluaciones
y_pred_rf = grid_rf_clf.predict(X_test)
y_prob_rf = grid_rf_clf.predict_proba(X_test)[:, 1]

y_pred_lr = grid_lr_clf.predict(X_test)
y_prob_lr = grid_lr_clf.predict_proba(X_test)[:, 1]

def evaluar_clasificacion(y_true, y_pred, y_prob, nombre):
    print(f"\nResultados {nombre}:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.3f}")
    print(f"Matriz de Confusión:\n{confusion_matrix(y_true, y_pred)}")

evaluar_clasificacion(y_class_test, y_pred_rf, y_prob_rf, "RandomForestClassifier")
evaluar_clasificacion(y_class_test, y_pred_lr, y_prob_lr, "LogisticRegression")

print("Conclusión clasificación: El uso de SMOTE fue el correcto. Al balancear los datos, se evita que los modelos ignoren a la clase minoritaria, logrando un gran equilibrio entre Precision y Recall.")

# ==========================================
# 3.2 Modelo de Regresión (1.5 puntos)
# ==========================================
print("\n--- 3.2 Modelo de Regresión ---")
param_grid_rf_reg = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10]
}
rf_reg = RandomForestRegressor(random_state=42)
grid_rf_reg = GridSearchCV(rf_reg, param_grid_rf_reg, cv=5, scoring='r2')
grid_rf_reg.fit(X_train, y_reg_train)

param_grid_ridge = {
    'alpha': [0.1, 1.0, 10.0]
}
ridge = Ridge(random_state=42)
grid_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='r2')
grid_ridge.fit(X_train, y_reg_train)

# Evaluaciones
y_pred_rf_reg = grid_rf_reg.predict(X_test)
y_pred_ridge = grid_ridge.predict(X_test)

def evaluar_regresion(y_true, y_pred, nombre):
    print(f"\nResultados {nombre}:")
    print(f"R²:   {r2_score(y_true, y_pred):.3f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")

evaluar_regresion(y_reg_test, y_pred_rf_reg, "RandomForestRegressor")
evaluar_regresion(y_reg_test, y_pred_ridge, "Ridge")

# Graficamos Valores reales vs lo que predijo el modelo
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_reg_test, y=y_pred_rf_reg, alpha=0.7, color='darkorange', edgecolor='k')
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'b--', lw=2)
plt.title("Valores Reales vs Predicciones (Random Forest Regressor)")
plt.xlabel("Puntaje Satisfacción (Real)")
plt.ylabel("Puntaje Satisfacción (Predicho)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('dispersion_satisfaccion.png')
plt.close()
print("\nGráfico de dispersión 'dispersion_satisfaccion.png' guardado.")

# Sacamos cuáles fueron las variables que más pesaron en la predicción
importancias = grid_rf_reg.best_estimator_.feature_importances_
df_import = pd.DataFrame({'Variable': X.columns, 'Importancia': importancias})
df_import = df_import.sort_values(by='Importancia', ascending=False)
print("\nTop 5 Variables Más Importantes para Predecir la Satisfacción:")
print(df_import.head(5).to_string(index=False))

print("\n" + "="*50)
print("Parte 4: Procesamiento con Apache Spark MLlib (1.5 puntos)")
print("="*50)

print("\n--- 4.1 Preparación en Spark ---")
# Iniciar sesión de Spark
spark = SparkSession.builder.appName("EduTechAnalytics").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Cargar el dataset en un DataFrame de Spark
csv_file = [f for f in os.listdir('.') if f.endswith('.csv')][0]
spark_df = spark.read.csv(csv_file, header=True, inferSchema=True)

# Crea la columna label binaria para clasificación (curso_completado)
spark_df = spark_df.withColumn("label", col("curso_completado").cast("double")).na.drop()

# Definir variables
categorical_cols = ['genero', 'pais', 'dispositivo_principal', 'horario_estudio', 'nivel_educacion', 'experiencia_previa', 'plataforma_origen']
numeric_cols = ['edad', 'horas_estudio_semanal', 'sesiones_totales', 'tareas_completadas', 'tareas_totales', 'participacion_foros', 'promedio_evaluaciones', 'monto_inversion']

# Aplica StringIndexer a variables categóricas
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_indexed").fit(spark_df) for c in categorical_cols]
for indexer in indexers:
    spark_df = indexer.transform(spark_df)

indexed_cols = [f"{c}_indexed" for c in categorical_cols]

# Utiliza VectorAssembler para crear columna features
assembler = VectorAssembler(inputCols=numeric_cols + indexed_cols, outputCol="features")
spark_df = assembler.transform(spark_df)

final_df = spark_df.select("features", "label")

# Divide en train/test (70/30)
train_data, test_data = final_df.randomSplit([0.7, 0.3], seed=42)
print("Datos cargados, transformados (StringIndexer y VectorAssembler) y divididos en 70/30.")

print("\n--- 4.2 Modelo en Spark ---")
# Entrena un modelo de clasificación: RandomForestClassifier
rf_spark = SparkRF(featuresCol="features", labelCol="label", seed=42)
rf_model_spark = rf_spark.fit(train_data)

# Genera predicciones en el conjunto de prueba
predictions = rf_model_spark.transform(test_data)

# Evalúa el modelo
evaluator_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
roc_auc_spark = evaluator_roc.evaluate(predictions)

evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy_spark = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
f1_spark = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})

print("\nEvaluación en Spark MLlib (RandomForestClassifier):")
print(f"ROC-AUC (areaUnderROC): {roc_auc_spark:.3f}")
print(f"Accuracy: {accuracy_spark:.3f}")
print(f"F1-Score: {f1_spark:.3f}")

# Muestra una tabla con las columnas: label, prediction, probability
print("\nMuestra de las predicciones (label, prediction, probability):")
predictions.select("label", "prediction", "probability").show(10, truncate=False)

spark.stop()
print("Sesión de Spark finalizada exitosamente.")

print("\n" + "="*50)
print("Parte 5: Inferencia Estadística (0.5 puntos)")
print("="*50)

# 1. Intervalo de confianza al 95%
satisfaccion = df['puntaje_satisfaccion']
media_sat = satisfaccion.mean()
error_est = stats.sem(satisfaccion)
intervalo_confianza = stats.t.interval(0.95, df=len(satisfaccion)-1, loc=media_sat, scale=error_est)

print("\n--- Intervalo de Confianza (95%) ---")
print(f"Media de Satisfacción: {media_sat:.3f}")
print(f"El intervalo de confianza al 95% está entre {intervalo_confianza[0]:.3f} y {intervalo_confianza[1]:.3f}")

# 2. Prueba de hipótesis (t-test)
sat_completados = df[df['curso_completado'] == 1]['puntaje_satisfaccion']
sat_no_completados = df[df['curso_completado'] == 0]['puntaje_satisfaccion']

print("\n--- Prueba de Hipótesis ---")
print("H0 (Hipótesis Nula): El puntaje promedio es IGUAL entre quienes completan el curso y quienes no.")
print("H1 (Hipótesis Alternativa): El puntaje promedio es DISTINTO entre quienes completan el curso y quienes no.")

# Realizamos un t-test
t_stat, p_value = stats.ttest_ind(sat_completados, sat_no_completados, equal_var=False)

print(f"Estadístico t: {t_stat:.3f}")
print(f"Valor-p:       {p_value:.3e}")

alpha = 0.05
print("\nConclusión con nivel de significancia alpha = 0.05:")
if p_value < alpha:
    print("Se rechaza h0, hay evidencia estadística para afirmar que el nivel de satisfacción difiere significativamente entre ambos grupos.")
else:
    print("No se puede rechazar h0, no hay evidencia estadística para afirmar que la satisfacción sea distinta entre ambos grupos.")
