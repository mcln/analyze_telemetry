import pandas as pd
import os
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Usar un backend no interactivo para gráficos

# Función para limpiar el archivo de telemetría
def clean_telemetry_file(input_file, output_file):
    try:
        # Leer solo la hoja "Data Telemetría"
        data = pd.read_excel(input_file, sheet_name="Data Telemetría")
        print("Archivo cargado exitosamente.")

        # Normalizar los nombres de columnas: sin espacios, en minúsculas y con guiones bajos
        data.columns = (
            data.columns.str.strip()               # Eliminar espacios en los extremos
                           .str.lower()            # Convertir a minúsculas
                           .str.replace(" ", "_")  # Reemplazar espacios con guiones bajos
                           .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)  # Eliminar caracteres especiales
        )

        # Eliminar filas vacías o incompletas
        data = data.dropna()

        # Limpiar las columnas numéricas que tienen texto adicional
        data['consumo'] = data['consumo'].str.replace(' lts', '').astype(float)
        data['odmetro'] = data['odmetro'].str.replace(' km', '').astype(float)
        data['torque'] = data['torque'].str.replace(' %', '').astype(float)
        data['acel'] = data['acel'].str.replace(' %', '').astype(float)
        data['nivel'] = data['nivel'].str.replace(' %', '').astype(float)
        data['rpm'] = data['rpm'].str.replace(' rpm', '').astype(float)
        data['velocidad'] = data['velocidad'].str.replace(' km/hr', '').astype(float)

        # Crear el directorio 'output' si no existe
        os.makedirs('C:/xampp/htdocs/python/gpschile/output', exist_ok=True)

        # Guardar el DataFrame limpio en un nuevo archivo CSV
        data.to_csv(output_file, index=False)
        print(f"Archivo limpio guardado como {output_file}")
        return data

    except Exception as e:
        print(f"Error durante el proceso de limpieza: {e}")
        return None

# Función para detectar posibles robos de combustible
def detect_fuel_theft(data):
    if data is None:
        print("No se pudo realizar la detección de robos. Los datos no fueron cargados correctamente.")
        return None, None

    try:
        data = data[data['nivel'] > 0.1].copy()  # Crear una copia explícita para evitar advertencias
        data.loc[:, 'rendimiento'] = data['odmetro'] / data['nivel']
        
        # Detección de anomalías
        model = IsolationForest(contamination=0.1, random_state=42)
        data.loc[:, 'anomalía'] = model.fit_predict(data[['rendimiento']])
        
        anomalies = data[data['anomalía'] == -1]
        print(f"Anomalías detectadas: {len(anomalies)} posibles robos de combustible.")
        return data, anomalies
    except Exception as e:
        print(f"Error en la detección de robos de combustible: {e}")
        return data, None

# Función para generar y guardar gráficos de dispersión
def generate_and_save_plots(data, output_dir='C:/xampp/htdocs/python/gpschile/output/'):
    os.makedirs(output_dir, exist_ok=True)

    # Explicaciones para cada gráfico
    explanations = {
        'acel_vs_torque': "La relación entre la aceleración y el torque muestra cómo la fuerza de giro afecta la velocidad de un vehículo.",
        'acel_vs_RPM': "La aceleración suele correlacionarse positivamente con el RPM, ya que una mayor aceleración generalmente implica un mayor régimen de revoluciones.",
        'acel_vs_velocidad': "La aceleración y la velocidad están directamente relacionadas, ya que una mayor aceleración aumenta la velocidad.",
        'acel_vs_consumo': "Aceleraciones rápidas pueden aumentar el consumo de combustible, ya que se requiere más energía para cambiar la velocidad rápidamente.",
        'torque_vs_RPM': "El torque y el RPM están relacionados en los motores, un torque mayor generalmente ocurre a bajas revoluciones.",
        'torque_vs_velocidad': "El torque influye en la velocidad del vehículo; un mayor torque puede ayudar a mantener una velocidad más alta.",
        'torque_vs_consumo': "El consumo de combustible puede incrementarse con un torque mayor, ya que requiere más combustible para generar más fuerza.",
        'RPM_vs_velocidad': "El RPM (revoluciones por minuto) y la velocidad están relacionados, especialmente en vehículos de transmisión manual, ya que a mayor RPM, mayor velocidad.",
        'RPM_vs_consumo': "A mayor RPM, generalmente se requiere más combustible, ya que el motor trabaja más intensamente.",
        'velocidad_vs_consumo': "La velocidad y el consumo de combustible están vinculados, normalmente a velocidades más altas, el consumo aumenta debido a la resistencia aerodinámica.",
    }

    # Generación de gráficos
    plots = [
        ('acel', 'torque'),
        ('acel', 'rpm'),
        ('acel', 'velocidad'),
        ('acel', 'consumo'),
        ('torque', 'rpm'),
        ('torque', 'velocidad'),
        ('torque', 'consumo'),
        ('rpm', 'velocidad'),
        ('rpm', 'consumo'),
        ('velocidad', 'consumo')
    ]

    for x_col, y_col in plots:
        plot_path = os.path.join(output_dir, f'{x_col}_vs_{y_col}.png')
        plt.figure(figsize=(8, 6))
        plt.scatter(data[x_col], data[y_col], alpha=0.5)
        plt.title(f'{x_col.capitalize()} vs {y_col.capitalize()}')
        plt.xlabel(x_col.capitalize())
        plt.ylabel(y_col.capitalize())
        
        # Valor promedio
        avg_value = data[[x_col, y_col]].mean()
        avg_text = f"Promedio: {x_col.capitalize()} = {avg_value[x_col]:.2f}, {y_col.capitalize()} = {avg_value[y_col]:.2f}"
        plt.figtext(0.5, -0.1, avg_text, wrap=True, horizontalalignment='center', fontsize=12)

        # Explicación del gráfico
        explanation = explanations.get(f'{x_col}_vs_{y_col}', 'Sin explicación disponible.')
        plt.figtext(0.5, -0.2, explanation, wrap=True, horizontalalignment='center', fontsize=10)

        # Guardar el gráfico
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfico guardado: {plot_path}")

# Función para guardar análisis en un archivo Excel
def save_analysis_to_excel(data, anomalies, output_dir='C:/xampp/htdocs/python/gpschile/output/'):
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Guardar anomalías en un archivo Excel
    anomalies_path = os.path.join(output_dir, 'posibles_robo_combustible.xlsx')
    anomalies.to_excel(anomalies_path, index=False)
    print(f"Archivo Excel guardado: {anomalies_path}")

# Ruta de entrada
input_file = r'C:\xampp\htdocs\python\gpschile\input\1140_-_SHHX21_TAC_-_Reporte_de_Monitoreo_04.11.2024_13-25-07.xlsx'

# Limpiar archivo de telemetría
cleaned_data = clean_telemetry_file(input_file, 'C:/xampp/htdocs/python/gpschile/output/telemetria_limpia.csv')

if cleaned_data is not None:
    # Detectar posibles robos de combustible
    cleaned_data, anomalies = detect_fuel_theft(cleaned_data)

    # Generar y guardar gráficos
    if cleaned_data is not None:
        generate_and_save_plots(cleaned_data, 'C:/xampp/htdocs/python/gpschile/output/')

    # Guardar análisis y anomalías en archivos Excel separados
    save_analysis_to_excel(cleaned_data, anomalies)
else:
    print("El archivo no pudo ser limpiado correctamente, revisa el log de errores.")


