***************************************************************************************************************************
*                                       Method for Object-base Diagnostic Evaluation (MODE)                               *
*                                                                                                                         * 
*                                                                                                    Aleen Arnaiz Cedeño  *
***************************************************************************************************************************

Librerias usadas para el procesamiento de imagenes:

from scipy.ndimage import gaussian_filter, label, binary_closing
from skimage.measure import regionprops
from matplotlib.patches import Ellipse
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

gaussian_filter : Suaviza el campo de precipitación para reducir ruido y mejorar la 
coherencia espacial antes de la detección.

label : Etiqueta regiones conectadas en una máscara binaria

binary_closing : Operación morfológica que cierra huecos y une regiones cercanas, 
eliminando fragmentación artificial.

regionprops : Extrae propiedades geométricas e intensivas de cada objeto etiquetado: área, 
centroide, orientación, intensidad media/máxima, coordenadas, etc.

ConvexHull : Calcula la envoltura convexa de las coordenadas geográficas de un objeto. 
Se usa para dibujar contornos suaves en los mapas.




##-----------------------------------------------------------------------------
## 0. Preprocesamiento de los datos 
##-----------------------------------------------------------------------------


El paso inicial es el preprocesamiento de los datos que se lleva a cabo desde dos 
scripts independientes:

    data_loader.py : se encarga de cargar y acumular de manera flexible los datos de GPM y WRF.
    preprocessor.py : se dedica al cálculo de precipitación (variable a verificar), alineación 
    espaciotemporal y preparación para MODE.
     
Estos scripts son esenciales para adaptar datos de diferente frecuencia (GPM cada 30 min, WRF cada 1 h) 
a una ventana de acumulación común (1H, 3H o 6H), manteniendo coherencia física y temporal antes de la 
verificación espacial con MODE.



##-----------------------------------------------------------------------------
## 1. Metodo read_h5py_data
##-----------------------------------------------------------------------------


* Metodo read_h5py_data(filepath) : Leer un archivo GPM en formato HDF5 y convertirlo a xarray.Dataset. 

    Abre el archivo HDF5 con h5py.File.
    Extrae variables:
        precipitation: array 3D (1, 3600, 1800) → (time, lon, lat).
        lat: vector de 1800 valores (-90° a 90°).
        lon: vector de 3600 valores (-180° a 180°).
         
    Extrae la fecha/hora del nombre del archivo:
        Ej: 3B-HHR.MS.MRG.3IMERG.20190501-S010000-E012959... → 2019-05-01 01:00:00.
         
    Reemplaza valores faltantes (-9999.9) por np.nan.
    Construye un xarray.Dataset con:
        Coordenadas: time, lat, lon.
        Variable: precipitation.
         
     
##-----------------------------------------------------------------------------
## 2. Metodo accumulate_gpm_flexible
##-----------------------------------------------------------------------------

* Metodo read_h5py_data : Acumula datos GPM según la ventana (1H, 3H, 6H). 

    Si accumulation_window == '1H' usa accumulate_gpm_hourly_original.

        En el caso de 1H, GPM tiene datos cada 30 min, pero la acumulación horaria requiere suma de dos pasos (00:00 + 30:00 → 01:00). 
        El método original agrupa manualmente por hora.        
    Si no usa accumulate_gpm_with_resample.
   
        Cada 3H/6H Se puede usar resample() directamente porque suma múltiples pasos sin ambigüedad.
     
##-----------------------------------------------------------------------------
## 4. Metodo accumulate_gpm_hourly_original (ANTES)
##-----------------------------------------------------------------------------

* Metodo accumulate_gpm_hourly_original :
Propósito: Acumulación horaria manual para GPM (30-min → 1-h). 


Tenemos los siguientes tiempos en nuestros ds_gpm.time:
    2019-05-20 00:00:00 
    2019-05-20 00:30:00 
    2019-05-20 01:00:00 
    2019-05-20 01:30:00 
    2019-05-20 02:00:00 
    2019-05-20 02:30:00 
    ...

Cada uno tiene un campo de precipitación en mm (por media hora).

1. Convertir tiempos a objetos pandas.Timestamp 
        gpm_times = pd.to_datetime(ds_gpm.time.values)
Lo cual asegura que todos los tiempos sean del mismo tipo para manipulación consistente.
     

2. Agrupar por "hora completa" 
        hour_key = time_val.replace(minute=0, second=0, microsecond=0)
Esto trunca cualquier tiempo a la hora entera anterior: 

Tiempo Original              Hour_Key (Clave de Agrupacion)
2019-05-20 00:00:00          2019-05-20 00:00:00
2019-05-20 00:30:00          2019-05-20 00:00:00
2019-05-20 01:00:00          2019-05-20 01:00:00
2019-05-20 01:30:00          2019-05-20 01:00:00
...                          ...

Los dos pasos de 30 min dentro de la misma hora se agrupan bajo la misma clave.

El diccionario hourly_groups queda así: 
{
  Timestamp('2019-05-20 00:00:00'): [índice_0, índice_1],   # 00:00 y 00:30
  Timestamp('2019-05-20 01:00:00'): [índice_2, índice_3],   # 01:00 y 01:30
  ...
}


##-----------------------------------------------------------------------------
## 4. Metodo accumulate_gpm_hourly_original (NUEVO!!)
##-----------------------------------------------------------------------------

* Metodo accumulate_gpm_hourly_original : alinea correctamente los datos de GPM IMERG Half-hourly (HHR) 
con la convención meteorológica estándar y con las salidas de WRF.
 
El valor etiquetado como 01:00 represente la precipitación total caída entre las 00:00 y las 01:00.
Este acumulado se construya sumando los dos pasos consecutivos de GPM que cubren ese intervalo: 00:30 y 01:00.
El resultado sea directamente comparable con la precipitación horaria de WRF, que se calcula como 
la diferencia entre acumulados consecutivos.
     

2. Obtención y conversión de tiempos 

        gpm_times = pd.to_datetime(ds_gpm.time.values)
Convierte los tiempos del DataArray a una lista de objetos pandas.Timestamp.
Ejemplo de entrada: [00:30, 01:00, 01:30, 02:00, ...].
     

3. Agrupación por hora calendario (inicio del intervalo) 
        hourly_groups = {}
        for i, time_val in enumerate(gpm_times):
            interval_start = time_val - pd.Timedelta(minutes=30)
            hour_key = interval_start.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_groups:
                hourly_groups[hour_key] = []
            hourly_groups[hour_key].append(i)
         
Corrige el timestamp:
    Cada valor de GPM con timestamp HH:MM representa la precipitación de HH:MM - 30 min a HH:MM.
    Por lo tanto, el inicio del intervalo real es time_val - 30 minutos. 

    Asigna a la hora calendario:
    El inicio del intervalo se trunca a la hora entera anterior.
        00:30 → inicio = 00:00 → hour_key = 00:00  
        01:00 → inicio = 00:30 → hour_key = 00:00  
        01:30 → inicio = 01:00 → hour_key = 01:00  
        02:00 → inicio = 01:30 → hour_key = 01:00
         

    Resultado:
        Grupo 00:00: índices de 00:30 y 01:00 → 00:00–01:00  
        Grupo 01:00: índices de 01:30 y 02:00 → 01:00–02:00
         
     
4. Acumulación de precipitación por grupo 

        accumulated_data = []
        accumulated_times_start = []
        for hour_key, indices in sorted(hourly_groups.items()):
            if len(indices) >= 1:
                hourly_precip = ds_gpm.isel(time=indices).sum(dim='time')
                accumulated_data.append(hourly_precip)
                accumulated_times_start.append(hour_key)
         
Suma los valores de todos los índices en cada grupo.
Guarda los tiempos de inicio del intervalo (00:00, 01:00, ...).
     
5. Reetiquetado final: de inicio a fin del intervalo 

        ds_gpm_hourly_start = xr.concat(accumulated_data, dim='time')
        ds_gpm_hourly_start = ds_gpm_hourly_start.assign_coords(time=accumulated_times_start)

# REETIQUETADO: convertir tiempos de INICIO a FIN del intervalo
        accumulated_times_end = [t + pd.Timedelta(hours=1) for t in accumulated_times_start]
        ds_gpm_hourly = ds_gpm_hourly_start.assign_coords(time=accumulated_times_end)
         
Crea un DataArray temporal con tiempos de inicio. 

    Suma 1 hora a cada tiempo de inicio para obtener el fin del intervalo. 
        00:00 (inicio) → 01:00 (fin)
        01:00 (inicio) → 02:00 (fin)
         
    Resultado final:
        El valor en 01:00 = precipitación de 00:00–01:00  
        El valor en 02:00 = precipitación de 01:00–02:00
         
     
##-----------------------------------------------------------------------------
## 5. Metodo accumulate_gpm_with_resample
##-----------------------------------------------------------------------------

* Metodo accumulate_gpm_with_resample : Acumular datos de precipitación de GPM IMERG Half-hourly (HHR) en 
ventanas temporales mayores o iguales a 3 horas (por ejemplo, 3H, 6H) de forma automática, robusta y eficiente, 
aprovechando la funcionalidad nativa de xarray.

        ds_accumulated = ds_gpm.resample(time=window, origin='start_day').sum(skipna=True)

resample(time=window): es un método de xarray que reorganiza los datos en intervalos regulares según la frecuencia 
especificada (window).
        Si window = '3H', agrupa los tiempos en bloques de 3 horas consecutivas: [00:00–03:00), [03:00–06:00), etc.
        Si window = '6H', crea bloques de 6 horas: [00:00–06:00), [06:00–12:00), etc.
         
origin='start_day': define el punto de alineación de los bloques.
        'start_day' fuerza a que todos los bloques comiencen exactamente a las 00:00, 03:00, 06:00... del día calendario.
        Esto evita que los bloques se deslicen o se alineen de forma arbitraria (por ejemplo, a las 01:00), lo que garantiza 
        consistencia entre días y experimentos.
         
.sum(skipna=True): aplica la operación de suma a todos los valores dentro de cada bloque.
        skipna=True indica que los valores faltantes (NaN) deben ignorarse en la suma, en lugar de propagar NaN al resultado.
        Dado que GPM HHR ya entrega tasas de precipitación en mm por intervalo de 30 minutos, sumar todos los pasos dentro de 
        un bloque de 3H da directamente la precipitación total en mm para ese periodo de 3 horas.
 

##-----------------------------------------------------------------------------
## 6. Metodo accumulate_wrf_flexible
##-----------------------------------------------------------------------------

* Metodo accumulate_wrf_flexible : Acumular los campos de precipitación de WRF según la ventana temporal deseada (1H, 3H, 6H).

Caso 1: accumulation_window == '1H' 

    No se realiza ninguna acumulación adicional.
    Las salidas horarias de WRF ya están en la frecuencia deseada. La precipitación neta por hora se calculará más adelante 
    en el preprocesador como la diferencia entre pasos consecutivos (ver calculate_hourly_precipitation).
     
Caso 2: accumulation_window == '3H' o '6H' 

    Identificación de variables de precipitación:
        RAINNC: precipitación no convectiva (estratiforme).
        RAINC: precipitación convectiva.
        La precipitación total es la suma de ambas: RAIN = RAINNC + RAINC.
         
Se aplica resample().sum() por separado a cada variable.
WRF almacena la precipitación acumulada desde el inicio de la simulación. 
Al sumar los valores en una ventana, se obtiene la precipitación total caída durante esa ventana.
     
Construcción del nuevo Dataset:

    Se crea un nuevo xr.Dataset (ds_accumulated) y se le asignan las variables acumuladas (RAINNC, RAINC).
     
Preservación de coordenadas y metadatos:

    Coordenadas temporales: se asignan usando resample().first(), lo que toma el primer timestamp de 
    cada bloque como representante (ej. 00:00 para el bloque 00:00–03:00).
    Coordenadas espaciales (XLAT, XLONG): se mantienen intactas, ya que no varían en el tiempo (o se 
    toma el primer paso si tienen dimensión temporal).



##-----------------------------------------------------------------------------
## 7. Metodo extract_time_from_wrf_filename
##-----------------------------------------------------------------------------

* Metodo extract_time_from_wrf_filename : Extraer de forma confiable y automática la marca de tiempo real de 
un archivo de salida de WRF (wrfout_*) a partir de su nombre de archivo

        match = re.search(r'wrfout_d01_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', filename)

re.search(): es una función del módulo re (expresiones regulares) que busca un patrón específico dentro de 
una cadena de texto (filename).
Patrón de la expresión regular:
        wrfout_d01_: coincide con el prefijo fijo del nombre del archivo para el dominio 1.
        (\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}): es un grupo de captura que extrae la parte variable:
            \d{4}: 4 dígitos para el año.
            \d{2}: 2 dígitos para mes, día, hora, minuto y segundo.
            El guion (-) y los dos puntos (:) son literales.
             
match.group(1): devuelve el contenido del primer grupo de captura, es decir, la cadena de fecha/hora pura 
(ej. '2019-05-20_12:00:00'). 
pd.to_datetime(..., format=...): convierte la cadena de texto en un objeto pandas.Timestamp, que es el estándar p
ara manejar fechas en xarray y pandas.
     
Por qué es crucial: garantiza que todos los archivos se procesen con su tiempo real correcto, evitando errores de 
alineación temporal que arruinarían toda la verificación. 
     


##-----------------------------------------------------------------------------
## 8. Metodo preprocess_wrf
##-----------------------------------------------------------------------------

* Metodo preprocess_wrf : Estandarizar y limpiar un Dataset de WRF individual para que tenga una estructura uniforme 
y compatible con el resto del flujo de trabajo.
     
Eliminación de la dimensión Time unitaria:

        if 'Time' in ds.dims and ds.dims['Time'] == 1:
            ds = ds.isel(Time=0)
         
Los archivos wrfout suelen tener una dimensión Time de tamaño 1.
isel(Time=0) selecciona el primer (y único) elemento, eliminando la dimensión innecesaria y simplificando la estructura.
     

Renombrado de dimensiones espaciales:

        ds = ds.rename({'south_north': 'lat', 'west_east': 'lon'})
 
WRF usa nombres de dimensión propios (south_north, west_east).
Se renombran a los nombres estándar (lat, lon) para que sean compatibles 
con GPM y con la clase MODE3DVerifier.
     

Almacenamiento del tiempo real:

        ds.attrs['real_time'] = file_time
     
Guarda el tiempo extraído del nombre del archivo como un atributo del dataset, lo que facilita 
su recuperación más adelante.
         


##-----------------------------------------------------------------------------
## 9. Metodo load_wrf_with_correct_times
##-----------------------------------------------------------------------------

* Metodo load_wrf_with_correct_times : Orquestar la carga completa de múltiples archivos de WRF, asegurando que se 
extraigan los tiempos correctos, se preprocesen individualmente y se combinen en un solo Dataset coherente, al que luego 
se le aplica la acumulación temporal si es necesario.


Iteración sobre la lista de archivos:

    Usa tqdm para mostrar una barra de progreso visual durante la carga.
     
Extracción y procesamiento por archivo:

    Para cada archivo, llama a extract_time_from_wrf_filename y preprocess_wrf.
    Almacena el Dataset resultante y su tiempo real en listas (datasets, wrf_times).
     
Concatenación final:
        ds_wrf = xr.concat(datasets, dim='time')
        ds_wrf = ds_wrf.assign_coords(time=wrf_times)
         
    xr.concat une todos los Dataset individuales a lo largo de una nueva dimensión time.
    assign_coords asigna la lista de tiempos reales como coordenadas de la dimensión temporal.
     
Aplicación de acumulación:

    Si la ventana no es 1H, llama a accumulate_wrf_flexible para generar los acumulados de 3H o 6H.
     


##-----------------------------------------------------------------------------
## 10. Metodo load_gpm_data
##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------
## 11. Metodo load_wrf_data
##-----------------------------------------------------------------------------

* Metodo load_gpm_data y  load_wrf_data: Actuan como funciones de interfaz pública y de conveniencia que encapsulan 
toda la lógica de carga y acumulación, permitiendo al usuario final (o al script principal) cargar los datos con una 
sola llamada, sin necesidad de conocer los detalles internos.

    load_gpm_data(accumulation_window=config.accum_window_to_mode): 
        Usa glob.glob(config.path_gpm + '...') para encontrar todos los archivos HDF5 de GPM que coincidan con el patrón 
        (ej. una fecha específica).
        Llama a accumulate_gpm_flexible(gpm_hourly_files, accumulation_window), que a su vez decide si usar el método horario 
        corregido o resample según la ventana.
        Devuelve un DataArray listo para el preprocesamiento.
         

    load_wrf_data(accumulation_window=config.accum_window_to_mode): 
        Usa glob.glob(config.path_wrf + 'wrfout_d01_...') para encontrar los archivos de WRF del dominio 1 (o d02 para 1 km).
        Llama a load_wrf_with_correct_times(wrf_files, accumulation_window).
        Devuelve un Dataset (o DataArray tras el preprocesamiento) listo para MODE.
         
     












##-----------------------------------------------------------------------------
## 1. Definicion de la clase: Class MODE3DVerifier 
##-----------------------------------------------------------------------------


* Class MODE3DVerifier : es la clase de Python que encapsula toda la funcionalidad 
del metodo MODE.

Una clase es una plantilla o molde que define la estructura y comportamiento de los 
objetos que se crearan a partir de ella. 
Una clase contiene atributos (variables) y metodos (funciones).

Un objeto , es una instancia concreta de una clase que tiene atributos y metodos y se 
crea llamando al nombre de la clase como si fuera una funcion.

Se suelen especificar los tipos de datos en los parametros de la clase 
(xr.DataArray o float = 1.0). Son especificaciones opcionales que indican que tipo de 
datos se espera de un parametro o retorno de la funcion. Se usan:

: para parametros
-> para retorno

##-----------------------------------------------------------------------------
## 2. Metodo __init__
##-----------------------------------------------------------------------------

* Metodo __init__ (Constructor) : el constructor se ejecuta cuando se crea una nueva 
instancia de la clase y recibe:

    forecast: datos del modelo
    observed: datos observados
    threshold: umbral para identificar objetos (5 mm para la lluvia)
    conv_radio: radio para suavizado gaussiano
    time_window: ventana temporal para agrupar objetos en pasos de tiempo
    min_object_size: minimo tamaño (pixeles) para considerar un objeto


# Validación y estandarización de dimensiones
self.forecast = self._standardize_dims(forecast)
self.observed = self._standardize_dims(observed)
self._validate_inputs(self.forecast, self.observed)

    Estandariza nombres de dimensiones (time, lat, lon).
    Valida que ambos datasets tengan los mismos tiempos.
    Guarda coordenadas geográficas originales.
    Inicializa listas vacías para objetos, matches y métricas.
     

self : se refiere a la instancia de la clase y se relaciona como funcionan los atributos de 
instancia y los metodos en las clases de Python. Cuando se escribe self.forecast, se esta 
creando o modificando un atributo que pertenece a la instancia especifica de la clase, del 
lado derecho esta llamando a un metodo de la clase. Sin self, Python buscaria, por ejemplo, 
_standardize_dims como una funcion global(no como un metodo de la clase). 

##-----------------------------------------------------------------------------
## 3. Metodo _standardize_dims
##-----------------------------------------------------------------------------

* Metodo _standardize_dims : mapea nombres de dimensiones a formato standar 
('time', 'lat', 'lon'), sin importar como vengan en los datos originales para garantizar que 
siempre usen los nombres estandarizados: time, lat y lon.

Ejemplo: si el WRF tiene dims=('t', 'south_north', 'west_east'), se convierte a ('time', 'lat', 'lon').

##-----------------------------------------------------------------------------
## 4. Metodo _validate_inputs
##-----------------------------------------------------------------------------

* Metodo _validate_inputs : verifica que los datos sean de DataArrays de xarray, tengan las 
dimensiones requeridas y tengan las mismas dimensiones espaciales y temporales.

##-----------------------------------------------------------------------------
## 5. Metodo run_verification
##-----------------------------------------------------------------------------

* Metodo run_verification : hace todo el proceso de verificacion a partir de cuatro metodos 
principales de la clase.

##-----------------------------------------------------------------------------
## 6. Metodo _preprocess_data
##-----------------------------------------------------------------------------

* Metodo _preprocess_data : Aplica un filtro gaussiano para suavizar los datos espaciales usando 
scipy.ndimage.gaussian_filter, es decir, prepara los dato para la deteccion de objetos usando el 
suavizado gaussiano y la umbralizacion.
    gaussian_filter (de scipy.ndimage) : aplica el filtro que difumina los datos usando una funcion 
                                         gaussiana
    sigma = self.conv_radius : controla el grado de suavizado dado por el conv_radius, que para un 
    valor mayor, genera un suavizado mas intenso.
    mode = 'nearest' : trata los bordes (evoita artefactos en los limites)

El filtro gaussiano se aplica con la funcion xr.apply_ufunc() al cual le pasamos:
    gaussian_filter : la funcion a aplicar
    self.forecast : datos de entrada
    input_core_dims : indica las dimensiones a las que se le aplica el filtro, en este caso las 
                      espaciales ('lat', 'lon'), dejando otras como 'time' intactas.
    vectorize : hace que la funcion trabaje correctamente incluso si la funcion tiene dimensiones 
                adicionales.

self.smoothed_fcst : contendra el DataArray con los datos suavizados


Lo siguiente completa el preprocesamiento de los datos aplicando umbralizacion y limpieza morfologica:

# Umbralización y limpieza
        self.binary_fcst = self.smoothed_fcst > self.threshold
        self.binary_obs = self.smoothed_obs > self.threshold
        
        # Operación morfológica para eliminar artefactos pequeños
        for t in range(len(self.forecast.time)):
            self.binary_fcst[t] = binary_closing(self.binary_fcst[t].values)
            self.binary_obs[t] = binary_closing(self.binary_obs[t].values) 

Convierte los datos suavizados en mascaras binarias (True/False o 1/0) donde
    True : son los valores que superan el threshold
    False : son los valores por debajo del umbral

Lo siguiente es la limpieza morfologica que usa:
    binary_closing (de scipy.ndimage) : que cierra pequenos huecos dentro de los objetos y elimina 
                                        artefactos diminutos como pixeles aislados. Luce algo asi:

La imagen inicial:

□□□□□□□□□□
□□□□□□□□□□
□□■■■■■■□□
□□■□□□■□□
□□■□■■■□□
□□■□■■■□□
□□■□□□■□□
□□■■■■■■□□
□□□□□□□□□□
□□□□□□□□□□

Al aplicar binary_closing:

□□□□□□□□□□
□□□□□□□□□□
□□■■■■■■□□
□□■■■■■■□□
□□■■■■■■□□
□□■■■■■■□□
□□■■■■■■□□
□□■■■■■■□□
□□□□□□□□□□
□□□□□□□□□□

Esto se hace con el objetivo de unir regiones cercanas por ejemplo, si hay dos nucleos de lluvia casi 
tocandose, binary_closing los fusiona. Ademas, elimina pixiles aislados; es decir, ruido residual que 
supero el umbral pero no es parte de un objeto real.

##-----------------------------------------------------------------------------
## 7. Metodo _identify_objects
##-----------------------------------------------------------------------------

* Metodo _identify_objects : coordina la identificacion de objetos en pronosticos y observaciones y 
llama a _process_time_slices para ambos conjuntos de datos (pronostico y la observacion). Usa 
time_coords paera mantener consistencia temporal.
                                                                            
##-----------------------------------------------------------------------------
## 8. Metodo _process_time_slices
##-----------------------------------------------------------------------------

* Metodo _process_time_slices : procesa cada instante de tiempo (timestep) para identificar objetos individuales 
en los datos binarios (binary_data) y extraer sus propiedades de los datos de intensidad (intensity_data).
Devuelve una lista de "objetos temporales" que representan sistemas que persisten en el tiempo.


Parámetros de entrada: 
binary_data:  xr.DataArray del Campo binario (True/False) resultado del suavizado + umbralización. Dimensiones: (time, lat, lon)

intensity_data: xr.DataArray del Campo original suavizado (no binario), con valores reales de precipitación. Se usa para calcular intensidad media/máxima de cada objeto.

obj_type: str "forecast" o "observed" Define qué umbrales y coordenadas usar.

time_coords: np.ndarray Lista de marcas de tiempo (pandas.Timestamp) para iterar.


Se seleciona el umbral de tamaño mínimo según resolución:
    if obj_type == 'forecast':
        min_size = self.min_object_size_forecast     # Ej: 10 píxeles
        resolution_info = "WRF (3km)"
    else:
        min_size = self.min_object_size_observed     # Ej: 2 píxeles
        resolution_info = "GPM (10km)"

Se itera sobre cada instante de tiempo del campo para extraer:
    binary_slice = binary_data.isel(time=t_idx).values        # Ej: array 2D de True/False
    intensity_slice = intensity_data.isel(time=t_idx).values  # Ej: array 2D de mm de lluvia

# Paso 1: Etiquetado de regiones conectadas
labels, n_objects = label(binary_slice)

    labels (de scipy.ndimage.label) : identifica grupos de pixeles conectados y asigna un 
    número entero único a cada región de píxeles conectados con valor True. Por ejemplo:
                                        Input: [[True, True, False], [False, True, True]]

                                        Output: [[1, 1, 0], [0, 2, 2]] (2 objetos etiquetados.

    n_objects : número total de objetos en ese timestep.


# Paso 2: Extracción de propiedades
props = regionprops(labels, intensity_image=intensity_slice)

    regionprops : calcula propiedades como area, centroide, intensidad media, orientacion, etc.
                  Es decir para cada objeto, calcula:
                    area : numero de pixeles.
                    centroid : coordenadas (y,x) del centroide.
                    bbox: bounding box (min_row, min_col, max_row, max_col).
                    coords: lista de coordenadas (fila, col) de todos los píxeles del objeto.
                    mean_intensity, max_intensity: promedio y máximo de intensity_slice en los píxeles del objeto.
                    orientacion : angulo del eje mayor (0° = vertical).
                    eccentricity: alargamiento del objeto.

# Paso 3: Filtrado y creación de diccionarios (ANTES)
if prop.area >= self.min_object_size:
    obj_dict = self._create_object_dict(prop, obj_type, time_val, t_idx)

Descarta objetos pequenos, es decir menores que min_object_size (min_object_size = 5 pixeles).
Para la creacion del diccionario se usa otro nuevo metodo _create_object_dict y finalmente
_group_temporal_objects.


# Paso 3: Filtrado y creación de diccionarios (AHORA)

for prop in props:
    areas_detected.append(prop.area)
    if prop.area >= min_size:
        obj_dict = self._create_object_dict(prop, obj_type, time_val, t_idx)
        objects_2d.append(obj_dict)

Guarda todas las áreas detectadas (solo para debugging) y filtra los objetos con area >= min_size.
Convierte cada objeto en un diccionario estructurado usando _create_object_dict() → este diccionario incluye:
        Coordenadas geográficas reales (no índices de píxeles).
        Información de tipo, tiempo, intensidad, forma, etc.
         
     
# Paso 4: Fusión espacial  (AHORA)

objects_2d = self._simple_spatial_merge(objects_2d, spatial_threshold=config.spatial_threshold)

Evitar que un mismo sistema físico se fragmente en varios objetos cercanos (ej. por estructura multicelular).
Cómo funciona (ver _simple_spatial_merge):

    Compara la distancia entre centroides de todos los pares de objetos en el mismo timestep.
    Si la distancia ≤ spatial_threshold (ej. 80 km), los fusiona en un solo objeto.
    La fusión es recursiva: si A está cerca de B, y B está cerca de C, se fusionan A+B+C.
    El objeto fusionado tiene:
        Área = suma de áreas.
        Centroide = promedio ponderado.
        Coordenadas = unión de todos los píxeles.
        Intensidad máxima = máx de todos.
        Bounding box que contiene a todos.
         
     
Luego Añade los objetos 2D (ya filtrados y fusionados) a la lista global objects_3d.


# Paso 5: Agrupamiento temporal

return self._group_temporal_objects(objects_3d, time_coords)

Convierte la lista de objetos 2D en objetos 3D (temporales) que representan sistemas que persisten en el tiempo.


##-------------------------------------------------------------------------
## 3.8.1 Metodo _simple_spatial_merge  NUEVO!!!!
##-------------------------------------------------------------------------




##-------------------------------------------------------------------------
## 3.8.2 Metodo _create_merged_object  NUEVO!!!!
##-------------------------------------------------------------------------






##-----------------------------------------------------------------------------
## 9. Metodo _create_object_dict
##-----------------------------------------------------------------------------

* Metodo _create_object_dict: devuelve en base regionprops con metadata especial para cada
objeto.

def _create_object_dict(self, prop, obj_type, time_val, t_idx):
    return {
        'type': obj_type,  # 'forecast' o 'observed'
        'time': pd.Timestamp(time_val),
        'time_idx': t_idx,
        'centroid': (prop.centroid[0], prop.centroid[1]),  # Posición (lat, lon)
        'area': prop.area,  # Tamaño en píxeles
        'orientation': prop.orientation,  # Ángulo de orientación
        'intensity_mean': prop.mean_intensity,  # Intensidad promedio (ej.: lluvia media)
        'intensity_max': prop.max_intensity,  # Intensidad máxima
        'bbox': prop.bbox,  # Bounding box (y_min, x_min, y_max, x_max)
        'coords': prop.coords,  # Coordenadas de todos los píxeles del objeto
        'label': prop.label,  # ID único del objeto
        'eccentricity': prop.eccentricity  # Circularidad (0=círculo, 1=linea)
    }

##-----------------------------------------------------------------------------
## 10. Metodo _group_temporal_objects
##-----------------------------------------------------------------------------

*Metodo _group_temporal_objects : es clave para rastrear objetos meteorologicos como tormentas
o sistemas de precipitacion a lo largo del tiempo, agrupando objetos individuales (objects_2d)
en objetos temporales consolidados (temporal_objects). Responde a Son estos objetos detectados
en diferentes momentos parte de la misma entidad fisica que evoluciona en el tiempo?. 


Recibe como entrada:
    objets 2D (detectados en cada timestep) : lista de diccionarios, donde cada diccionario representa 
    un objeto detectado en  un instante de tiempo (esto es la salida de _create_object_dict). Por ejemplo:

             [
    {'type': 'forecast', 'time': Timestamp('2023-01-01 12:00'), 'centroid': (10.2, -70.5), 'coords': [(10, -70), ...], ...},
    {'type': 'forecast', 'time': Timestamp('2023-01-01 13:00'), 'centroid': (10.3, -70.6), 'coords': [(10, -71), ...], ...},
    ...
             ]
    time_coords : array de timestamps para normalizar las diferencias temporales.

y agruparlos en "objetos 3D" o "objetos temporales" que representan sistemas físicos que persisten y evolucionan en el tiempo, bajo dos condiciones clave: 

    Proximidad temporal: los objetos deben estar separados por un número razonable de pasos de tiempo.
    Proximidad espacial: sus centroides no deben moverse más allá de un umbral físico (ej. 20 km).
     

# Convertir time_coords a pandas.Timestamp para comparaciones consistentes
        time_coords = [pd.Timestamp(t) for t in time_coords]
Convierte time_coords a pandas.Timestamp para consistencia y luego se calcula el promedio de diferencias 
temporales entre pasos consecutivos (avg_time_diff), normalizado en horas.

# Calcular diferencias temporales promedio
        time_diffs = np.diff(time_coords)
        avg_time_diff = np.mean(time_diffs).total_seconds() / 3600 if len(time_diffs) > 0 else 1  # en horas
        
Lo cual es bueno para hacer el algoritmo robusto a intervalos de tiempo irregulares o variables. En los casos en que 
WRF puede tener datos cada 1 hora, pero GPM puede tener pasos de 30 min, 1h, 1h30min, etc. (Pero no es nuestro caso)

Se calcula la diferencia promedio entre pasos de tiempo consecutivos (avg_time_diff) en horas.

Luego, cualquier diferencia temporal entre dos objetos se normaliza dividiéndola por este promedio:
        time_diff = (obj['time'] - last_obj['time']).total_seconds() / 3600 / avg_time_diff


Dos objeto se consideran parte del mismo grupo si:
    (obj['time'] - last_obj['time']).total_seconds() / 3600 / avg_time_diff <= self.time_window
    time_window : es un parametro ajustable por ejemplo 3 horas. Si avg_time_diff = 1 y time_window =2,
                  objetos separados por dos horas pueden agruparse.


# Algoritmo de agrupamiento (greedy en tiempo)

    for obj in objects_sorted:
    matched = False
    for group in groups:
        last_obj = group[-1]
        time_diff = ...  # normalizado
        if time_diff <= self.time_window:
            distance_km = ...  # entre centroides
            if distance_km <= config.min_dist_same_object:
                group.append(obj)
                matched = True
                break
    if not matched:
        groups.append([obj])


Para cada objeto nuevo:
        Se recorren todos los grupos existentes.
        Se compara solo con el último objeto del grupo (asumiendo evolución continua).
        Si cumple ambas condiciones:
            Temporal: time_diff ≤ self.time_window (en pasos normalizados)
            Espacial: distancia ≤ min_dist_same_object (ej. 20 km)
             
        → Se añade al grupo.
         
    Si no encaja en ningún grupo → se crea un nuevo grupo (nuevo sistema).


Finalment para cada objeto 2D, genera un objeto 3D con propiedades agregadas:

    for group in groups:
                times = [obj['time'] for obj in group]
                time_idxs = [obj['time_idx'] for obj in group]
                centroids = [obj['centroid'] for obj in group]
                areas = [obj['area'] for obj in group]
                intensities = [obj['intensity_mean'] for obj in group]
                
                temporal_obj = {
                    'type': group[0]['type'],
                    'time_start': min(times),
                    'time_end': max(times),
                    'duration': len(group),
                    'time_points': times,
                    'time_indices': time_idxs,
                    'centroid_mean': np.mean(centroids, axis=0),
                    'centroid_trajectory': centroids,
                    'area_mean': np.mean(areas),
                    'area_max': max(areas),
                    'intensity_mean': np.mean(intensities),
                    'intensity_max': max(intensities),
                    'objects_2d': group,
                    'id': len(temporal_objects)
                }
                

time_start/time_end: Inicio y fin del sistema

duration: Número de timesteps que persistió

time_points: Lista de todos los tiempos

centroid_trajectory_geo: Trayectoria completa del centroide (lat/lon)

centroid_mean_geo: Posición promedio del sistema

area_mean/area_max: Tamaño promedio y máximo

intensity_mean/intensity_max: Intensidad promedio y pico

objects_2d: Referencia a los objetos 2D originales (para análisis posterior)

id: Identificador único

resolution: Resolución espacial (3 km o 10 km)


En teoría de grafos, esto se modela así: G=(V,E)

    V (Vértices o Nodos): Son tus objetos 2D individuales identificados en cada paso de tiempo (Oit\mathcal{O}_i^{t}Oit​). 
    Cada "mancha" de lluvia en una hora específica es un vértice.

    E (Aristas o Conexiones): Son las "líneas invisibles" que unen un objeto en el tiempo ttt con un objeto en el tiempo t+1. 
    Si existe una arista, significa que el algoritmo ha decidido que son el mismo sistema de precipitación evolucionando.

Un grupo temporal (una tormenta con ciclo de vida) no es más que una Componente Conexa en este grafo: un conjunto de vértices unidos por aristas.

Para que el algoritmo dibuje una línea entre Oit\mathcal{O}_i^{t}Oit​ y Ojt+1\mathcal{O}_j^{t+1}Ojt+1​, deben cumplirse dos condiciones 
simultáneas (una lógica AND). Esto está codificado en el método _group_temporal_objects de tu script:

Condición A: Proximidad Temporal (El tiempo no puede saltar)

El objeto nuevo debe aparecer dentro de una ventana de tiempo razonable respecto al último objeto del grupo. Si hay un hueco de 6 horas, 
asumimos que la tormenta anterior murió y esta es una nueva, aunque estén en el mismo lugar.


Condición B: Proximidad Espacial (La tormenta no puede teletransportarse)

El centroide de la nueva mancha de lluvia debe estar cerca del centroide de la mancha anterior. Si a las 14:00 la tormenta está en Madrid y 
a las 15:00 aparece una en Barcelona, el grafo no dibujará una arista entre ellas. Son entidades distintas.


Imagina 3 pasos de tiempo. El script procesa así:

    Tiempo t=1t=1t=1: Detecta el Objeto A. 
        Acción: Crea un nuevo grupo: Grupo 1 = [A]
    Tiempo t=2t=2t=2: Detecta el Objeto B. 
        Verificación: ¿Está B cerca en tiempo y espacio del último elemento de Grupo 1 (que es A)? SÍ.
        Acción: Añade B al grupo. Grupo 1 = [A, B]. (Se ha creado la arista A→BA \rightarrow BA→B).
    Tiempo t=3t=3t=3: Detecta el Objeto C (cerca de B) y el Objeto D (muy lejos, en otra provincia).
        Verificación para C: ¿Cerca de B? SÍ. Grupo 1 = [A, B, C]. (Arista B→CB \rightarrow CB→C).
        Verificación para D: ¿Cerca de B o C? NO.
        Acción para D: Crea un nuevo grupo. Grupo 2 = [D].

Al final, en lugar de tener 4 objetos 2D aislados (A, B, C, D), tu script devuelve 2 Entidades Espacio-Temporales:

    Entidad 1 (Ciclo de vida largo): Compuesta por [A, B, C].
    Entidad 2 (Ciclo de vida corto): Compuesta por [D].


Esto significa que el grafo se construye como una cadena (A→B→CA \rightarrow B \rightarrow CA→B→C), no como una malla completa 
donde C se compara con A y con B. ¿Es esto correcto? Sí, para seguimiento de tormentas es el enfoque estándar y más eficiente. 
Asume que la evolución es un proceso de Markov de primer orden: el estado de la tormenta en t+1t+1t+1 depende principalmente de 
su estado en ttt. Comparar con todos los anteriores sería computacionalmente costoso y meteorológicamente menos relevante 
(una tormenta de hace 3 horas ya no define directamente la posición de la actual).








##-----------------------------------------------------------------------------
## 11. Metodo _match_objects
##-----------------------------------------------------------------------------

*Metodo _match_objects: utiliza una estrategia de emparejamiento usando la matriz de interes 
que calcula que tan bien cada objeto pronosticado (fcst) coincide con uno observado (obs). 
Esto lo consigue llamando al metodo _calculate_interest():
    self.interest_matrix[i, j] = self._calculate_interest(fcst_obj, obs_obj)

Luego hace un emparejamiento Greedy, donde ordena todos los pares posibles por interes descendente
y selecciona el mejor par disponible, evitando duplicados:

    Ejemplo de Matriz de Interés:
	        Obs1	Obs2
    Fcst1	0.85	0.40
    Fcst2	0.30	0.75

    Emparejamientos: (Fcst1, Obs1) y (Fcst2, Obs2).

##-----------------------------------------------------------------------------
## 12. Metodo _calculate_interest
##-----------------------------------------------------------------------------

*Metodo _calculate_interest: este metodo evalua que tan bien coincide un objeto pronosticado (fcst) 
con uno observado (obs) mediante la puntuacion de interes (0-1) que combina 5 factores espaciotemporales.
Esta vez, la formula del interes es un promedio ponderado que incluye:

    interest_distance : mide que tan cercanos estan los centros de ambos objetos
    area_ratio : calcula una proporcion entre las areas, divide el area del menor objeto entre el area 
    del mayor
    overlap : mide la maxima coincidencia espacial en tiempos comunes
    interest_orientation : mide la distancia angular (respecto del eje x horizontal) entre los objetos 
    observados y pronosticados.
    temporal_match : Mide la fraccion de tiempos compartidos entre los objetos observados y pronosticados


    # 1. Distancia entre centroides
        distance = self._calculate_centroid_distance(fcst_obj, obs_obj)
        max_distance = np.sqrt(self.forecast.sizes['lat']**2 + self.forecast.sizes['lon']**2)
        interest_distance = np.exp(-distance / (0.2 * max_distance))

        Por ejemplo: 
        Si distance = 50 km y max_distance = 1000 km:
        interest_distance = e^(-50 / 200) ≈ 0.78.
    
        Luego de haber calculado la distancia euclideana entre los centroides, hace un decaimiento
        exponencial de la distancia para llevar el valor a un interest_distance

    # 2. Razón de áreas
        area_ratio = min(fcst_obj['area_mean'], obs_obj['area_mean']) / max(fcst_obj['area_mean'], obs_obj['area_mean'])
        
        Se divide el area menor de los objetos entre la mayor
        Por ejemplo:
        area_fcst = 100 km², area_obs = 80 km² → area_ratio = 80/100 = 0.8

    # 3. Solapamiento espaciotemporal
        overlap = self._calculate_spatiotemporal_overlap(fcst_obj, obs_obj)
        
        Mide la maxima coincidencia espacial en tiempos comunes usando:
        overlap = (píxeles en común) / (píxeles totales de ambos objetos)

        Por ejemplo:

        Si en un tiempo compartido:

        fcst: 10 píxeles, obs: 8 píxeles, común: 6 píxeles.
        overlap = 6 / (10 + 8 - 6) = 6/12 = 0.5.
        
        Pronóstico:    ■■■■■□□□
        Observación:   □□■■■■□□
        Superposición:   ■■■■ (4/7 ≈ 0.57)

    # 4. Orientación (solo para objetos no circulares)

        orientation_diff = self._calculate_orientation_difference(fcst_obj, obs_obj)
        interest_orientation = 1 - (orientation_diff / 90) if orientation_diff is not None else 0.5
        
        Mide la diferencia angular entre ejes mayores

        El valor de interes_orientation solo viene dado por una resta, entre 1 y la normalizacion del 
        valor de la diferencia angular (orientation_diff) entre 90 porque es la maxima diferencia de 
        orientacion que pueden tener dos objetos. Solo contribuye en un 10% al interes total porque 
        es menos critico que la posicion o el tamano de los objetos. 
        
    
    # 5. Coincidencia temporal
        temporal_match = self._calculate_time_overlap(fcst_obj, obs_obj)

        Mide la fraccion de tiempos compartidos


Finalmente el valor de interes es un promedio ponderado de pesos (weights) con los parametros de interes
que se calcularon en cada una de las clases. Los pesos son los siguientes:

weights = {
            'distance': 0.25,
            'area_ratio': 0.2,
            'overlap': 0.25,
            'orientation': 0.1,
            'temporal': 0.2
        }

Y el interes total ponderado:

# Interés total ponderado
        total_interest = (
            weights['distance'] * interest_distance +
            weights['area_ratio'] * area_ratio +
            weights['overlap'] * overlap +
            weights['orientation'] * interest_orientation +
            weights['temporal'] * temporal_match
        )

Es decir: 

total_interest = (
    0.25 * interest_distance +    # Distancia entre centroides
    0.20 * area_ratio +           # Similaridad de tamaño
    0.25 * overlap +              # Superposición espacial
    0.10 * interest_orientation + # Alineación angular
    0.20 * temporal_match         # Coincidencia temporal
)

Es decir, para un 100% de match entre cada uno de los objetos, se le esta dando una prioridad del:

total_interest = (weight₁ × metric₁) + (weight₂ × metric₂) + ...

25% A la distancia entre los centriodes (mayor de los pesos)
20% Al area que ocupan los objetos
25% A la superposicion espacial de las areas de los objetos (mayor de los pesos)
10% A la alineacion u orientacion de los objetos
20% A la coincidencia temporal

Los pesos reflejan que tan importante es cada factor en el problema, en el cual, algunos factores
(como la posicion) son mas objetivos para determinar coincidencias que otros factores (como la 
orientacion) pueden ser ruidosos o irrelevantes en ciertos contextos.

##-----------------------------------------------------------------------------
## 13. Metodo _calculate_centroid_distance
##-----------------------------------------------------------------------------

*Metodo _calculate_centroid_distance : Calcula la distancia euclideana entre los centroides
en indices de array, es decir:

    distance = sqrt((lat_fcst - lat_obs)² + (lon_fcst - lon_obs)²)  # Distancia euclidiana
    interest_distance = exp(-distance / (0.2 * max_distance))  # Decaimiento exponencial 

##-----------------------------------------------------------------------------
## 14. Metodo _calculate_spatiotemporal_overlap
##-----------------------------------------------------------------------------

*Metodo _calculate_spatiotemporal_overlap : Calcula la maxima superposicion espacial entre
un objeto pronosticado (fcst_obj) y uno observado (obs_obj) en los tiempos donde ambos coexisten.
Es clave para determinar si dos objetos representan el mismo fenomeno fisico.

    common_times = set(fcst_obj['time_points']).intersection(obs_obj['time_points'])
    Encuentra tiempos comunes donde ambos objetos existen. Por ejemplo:
       
        fcst_obj existe en [t1, t2, t3].

        obs_obj existe en [t2, t3, t4].

        common_times = {t2, t3}

    Itera sobre tiempos comunes, es decir, para cada tiempo t en common_times:

        Obtiene objetos 2D correspondientes:

        fcst_2d = next((o for o in fcst_obj['objects_2d'] if o['time'] == t), None)
        obs_2d = next((o for o in obs_obj['objects_2d'] if o['time'] == t), None) 

        Esto busca en las listas objects_2d (que contienen los objetos individuales en
        cada tiempo) aquellos que coincidan con el tiempo t.

        if fcst_2d and obs_2d:

        Si existen ambos objetos. Entonces calcula la superposicion espacial:
        
        coords_fcst = set([tuple(coord) for coord in fcst_2d['coords']])
        coords_obs = set([tuple(coord) for coord in obs_2d['coords']])
        intersection = len(coords_fcst & coords_obs)  # Píxeles/celdas comunes
        union = len(coords_fcst | coords_obs)         # Píxeles/celdas totales cubiertos
        overlap = intersection / union if union > 0 else 0

        Formula:
        overlap = Area de interseccion / Area de union​ = ∣A∩B∣ / ∣A∪B∣

        Actualizar maxima superposicion:

        max_overlap = max(max_overlap, overlap)

        Retiene el valor mas alto de superposicion encontrado en todos los tiempos comunes.

Ejemplo:

Datos de Entrada
    Objeto Pronosticado (fcst_obj):

        time_points: [t1, t2]

        objects_2d:
            En t1: coords = [(1,1), (1,2), (2,1)] (3 píxeles).

            En t2: coords = [(1,2), (2,2), (2,3)] (3 píxeles).

    Objeto Observado (obs_obj):
        time_points: [t1, t3]

        objects_2d:
            En t1: coords = [(1,1), (1,2)] (2 píxeles).

            En t3: coords = [(3,3)].

Procesamiento
    Tiempos comunes: Solo t1.
    En t1:

        coords_fcst = {(1,1), (1,2), (2,1)}.

        coords_obs = {(1,1), (1,2)}.

        Intersección: {(1,1), (1,2)} → 2 píxeles.

        Unión: {(1,1), (1,2), (2,1)} → 3 píxeles.

        overlap = 2/3 ≈ 0.67.
    Resultado:
    max_overlap = 0.67 (no hay otros tiempos comunes para comparar)
​
        
Para t1:

Pronóstico (A):   ■ ■ □
                  ■ □ □
Observación (B):  ■ ■ □

■ = Píxel en ambos (intersección).
□ = Píxel solo en A o B.

    Área de intersección (A ∩ B): 2 píxeles.

    Área de unión (A ∪ B): 3 píxeles.

    IoU: 2/3 ≈ 0.67.

##-----------------------------------------------------------------------------
## 15. Metodo _calculate_orientation_difference
##-----------------------------------------------------------------------------


*Metodo _calculate_orientation_difference : calcula la diferencia angular promedio entre las orientaciones
de un objeto pronosticado (fcst_obj) y uno observado (obs_obj). Es clave para evaluar si la forma y 
alineacion de los objetos coinciden.

    1. Verifica que existen ambos objetos (fcst_2d y obs_2d) en el tiempo t:

    for t in set(fcst_obj['time_points']).intersection(obs_obj['time_points']):
            fcst_2d = next((o for o in fcst_obj['objects_2d'] if o['time'] == t), None)
            obs_2d = next((o for o in obs_obj['objects_2d'] if o['time'] == t), None)

    2. Filtra objetos validos: 
    
    if fcst_2d and obs_2d and fcst_2d['eccentricity'] > 0.2 and obs_2d['eccentricity'] > 0.2: 

    Solo considera objetos con eccentricity > 0.2 (objetos elongados) porque la orientacion
    en objetos circulares no es significativa como por ejemplo las tormentas simetricas.


    La excentricidad de un objeto se calcula a partir de los momentos de inercia de sus pixeles/celdas,
    donde:
        0.0 : objeto perfetamente circular (tormenta simetrica)
        1.0 : objeto completamente lineal (frente estrecho)

    El umbral de 0.2 es un valor empirico utilizado para distinguir objetos no circulares (donde 
    la orientacion es relevante) de los circulares (donde la orientacion no tiene significado fisico).
    Por ejemplo:

        eccentricity ≈ 0.1:
            ■■■
            ■■■
            ■■■
        Tormenta redonda, orientacion indefinida

        eccentricity ≈ 0.8:
            ■■■■■■■■
            ■■
        Frente alargado, orientacion ~0°

    3. Calcula la diferencia angular:

    angle_diff = abs(fcst_2d['orientation'] - obs_2d['orientation'])
    angle_diff = min(angle_diff, 180 - angle_diff)  # Considerar simetría

    orientation : Angulo (grados) del eje mayor del objeto respecto al eje horizontal (0° = vertical, 
    90° = horizontal)

    simetría: La diferencia entre 30° y 210° es 180°, pero físicamente es la misma orientación 
    (por eso se usa min(angle_diff, 180 - angle_diff)).

    Por ejemplo:
        Si fcst_orientation = 30° y obs_orientation = 210°:

            abs(30 - 210) = 180° → min(180, 0) = 0°.

        Si fcst_orientation = 30° y obs_orientation = 50°:

            abs(30 - 50) = 20° → min(20, 160) = 20°

    4. La funcion devuelve un promedio de diferencias:

    return np.mean(orientation_diffs) if orientation_diffs else None

    Retorna el promedio de diferencias angulares (grados) si hay datos validos
    Retorna None si: 
        No hay tiempos comunes
        Todos los objetos son circulares (eccentricity ≤ 0.2)

    Por ejemplo:

    Datos de Entrada
    Objeto Pronosticado (fcst_obj):

        time_points: [t1, t2]

        objects_2d:

            En t1: orientation = 45°, eccentricity = 0.8 (objeto alargado).

            En t2: orientation = 60°, eccentricity = 0.1 (objeto circular, se ignora).

    Objeto Observado (obs_obj):

        time_points: [t1, t3]

        objects_2d:

            En t1: orientation = 225°, eccentricity = 0.7 (objeto alargado).

            En t3: orientation = 90°, eccentricity = 0.3.

    Procesamiento

        Tiempos comunes: Solo t1.

        En t1:

            Ambos objetos son no circulares (eccentricity > 0.2).

            angle_diff = abs(45° - 225°) = 180° → min(180, 0) = 0°.

        Resultado:
        orientation_diffs = [0°] → np.mean([0]) = 0°.

##-----------------------------------------------------------------------------
## 16. Metodo _calculate_time_overlap
##-----------------------------------------------------------------------------

*Metodo _calculate_time_overlap : mide la fraccion de tiempos compartidos. Utilizando: 

temporal_match = (tiempos comunes) / (tiempos totales únicos)

Por ejemplo: 

    fcst: [Día1, Día2, Día3], obs: [Día2, Día3, Día4].

    comunes = [Día2, Día3], totales = [Día1, Día2, Día3, Día4].

    temporal_match = 2/4 = 0.5.

##-----------------------------------------------------------------------------
## 17.  Metodo _calculate_metrics
##-----------------------------------------------------------------------------

* Metodo _calculate_metrics : calcula metricas claves para evaluar la calidad de los emparejamientos
entre los objetos observados y pronosticados, combinando medidas de interes espaciotemporal y habilidad
predictiva. El metodo computa tres grupos de metricas:
    Mediana del Maximo Interes (MMI) : 
    Gilbert Skill Score (GSS) :
    Estadisticas basicas :


1. Mediana del Maximo Interes (MMI) :

Evalua que tan bien coinciden los objetos en terminos de posicion, tamano y tiempo, usando la matriz 
de interes (interest_matrix):

    max_interest_fcst = np.max(self.interest_matrix, axis=1)  # Máximo interés por objeto pronosticado 
    (filas de la matriz de interes)
    max_interest_obs = np.max(self.interest_matrix, axis=0)   # Máximo interés por objeto observado
    (columnas de la matriz de interes)
    self.metrics['MMI_forecast'] = float(np.median(max_interest_fcst))  # Mediana (pronóstico)
    self.metrics['MMI_observed'] = float(np.median(max_interest_obs))   # Mediana (observación)
    self.metrics['MMI'] = float(np.median(np.concatenate([max_interest_fcst, max_interest_obs])))  # Mediana Global

Se usa la mediana por su robustez frente a valores extremos. Por ejemplo:

    Si MMI_forectast = 0.7 : el 50% de los objetos pronosticados tuvieron al menos una coincidencia
    con interes ≥ 0.7.

2. Gilbert Skill Score (GSS) :

El Gilbert Skill Score (GSS), tambien conocido como Equitable Threat Score (ETS) es una metrica
fundamental en verificacion de pronosticos que evaluan la habilidad predictiva ajustando por 
coincidencias aleatorias. A diferencia del Critical Success Index (CSI), el GSS penaliza correctamente
los aciertos que podrian ocurrir por puro azar, por lo que no se infla artificialmente en dominios con 
muchos objetos evitando sobrestimar la habilidad por efectos aleatorios.


Mide la habilidad predictiva ajustando por coincidencias aleatorias segun :

    GSS = (H - Hrandom)/(H + F + M - Hrandom)
donde:

H: Hits (emparejamientos correctos) : Los objetos que estan en las observaciones y que el modelo pronostico
F: False alarms (objetos pronosticados sin match) : Los objetos que el modelo pronostico y no son observados
M: Misses (objetos observados sin match) : Los objetos que son observados, pero que no fueron pronosticados
Hrandom: Hits aleatorios (coincidencias de azar) : Numero esperado de coincidencias por azar

En el caso de Hrandom surge como un problema del Critical Success Index dado por:

    CSI = (H)/(H + F + M)

el cual no distingue bien entre habilidad real y coincidencias aleatorias. Por ejemplo, si tanto el pronostico
como la observacion tienen muchos objetos, el CSI puede ser alto incluso si el modelo no tiene habilidad. 
Entonces como sloucion se resta Hrandom (lo que se esperaria por azar) para aislar la habilidad real del modelo.


Calculo de Hrandom :

La clave esta en estimar cuantas coincidencias o Hits (H) ocurririan aleatoriamente si los objetos pronosticdos
y observados se distrubuyen al azar en el espacio y el tiempo. Se usa:

    Hrandom = (Volumen pronosticado x Volumen obsrvado)/(Volumen total del dominio)

    # Calcular hits esperados por azar
        total_volume = self.forecast.sizes['lat'] * self.forecast.sizes['lon'] * self.forecast.sizes['time']
        obs_volume = sum(obj['area_mean'] * obj['duration'] for obj in self.observed_objects)
        fcst_volume = sum(obj['area_mean'] * obj['duration'] for obj in self.forecast_objects)
        
        expected_random_hits = (fcst_volume * obs_volume) / total_volume if total_volume > 0 else 0

    donde:

    1. Volumen pronosticado (fcst_volume) : suma las areas de todos los objetos pronosticados multiplicados
    por su duracion ∑(area_mean×duration) para cada fcst_obj y representa la huella espacio-temporal total 
    del pronostico.

    2. Volumen observado (obs_volume) : suma las areas de todos los objetos observados multiplicados por su
    duracion ∑(area_mean×duration) para cada obs_obj y representa la huella espacio-temporal total de la 
    observacion.

    3. Volumen total del dominio (total_volume) : es el espacio total disponible que multiplica la cantidad 
    de latitudes por la cantidad de longitudes por los pasos de tiempos en el dominio, es decir n_lat×n_lon×n_time. 
    Por ejemplo, si:

        n_lat = 10
        n_lon = 100
        n_time = 10

    Entonces:

    total_volume = 1000 celdas x 10 pasos de tiempo = 10 000 unidades de volumen.

Un ejemplo general para el Calculo de Hrandom seria:

    Pronostico : 5 objetos, area total = 500 km², duración total = 10 pasos → fcst_volume = 500 × 10 = 5000.
    Observacion : 4 objetos, area total = 400 km², duracion total = 8 pasos → obs_volume = 400 × 8 = 3200
    Dominio : 1000 km² × 20 pasos  total_volume → 20 000

Entonces:

Hrandom = (5000 × 3200)/2000 = 800 coincidencias aleatorias esperadas

Escenario	                    GSS	                  Explicación
Todos los hits son aleatorios	0.0	    H = Hrandom (modelo no informativo).
Sin falsos/fallos	            1.0	    F = M = 0 y H > Hrandom​ (emparejamiento perfecto).
Más falsos que hits reales	    < 0	    El modelo es engañoso (peor que adivinar al azar).


3. Gilbert Skill Score Object-Based (GSS_Obj-Based) :

Basado en (A.Davis et al. 2009)

Una metrica alternativa que resume el comportamiento del modelo es el Gilbert Skill Score basado en objetos.
A diferencia del GSS tradicional, GSS_Obj-Based difiere en la aplicacion tradicional en la verificacion de precipitacion porque la evaluacion se realiza para objetos en lugar de celdas individuales de la cuadricula.
Este guarda relacion con las metricas de verificacion tradicionales, pero conserva la ventaja de la perspectiva
basada en objetos.

El GSS_Obj-Based en este caso es derivado del numero de objetos simples matcheados o hits, el numero de objetos pronosticados y el numero de objetos observados sin match o misses como :

    GSS_Obj-Based = (N_m - epsilon)/(N_f + M - epsilon)

donde:

N_m : Numero de objetos matcheados (Hits)
N_f : Numero total del objetos pronosticados
M : Numero de objetos observados sin match (Misses)    
epsilon : Hits aleatorios (coincidencias de azar) : Numero esperado de coincidencias por azar


Se llega a la conclusion de que la fraccion efectiva del area ocupada para la coincidencia, f_A, es aproximadamente
el 5% del area total. Segun esta definicion, el area fraccionaria nula es el 95% del dominio. Con lo cual, en 
promedio, el 95% del dominio esta vacio; por lo tanto, un objeto pronosticado en esta area no tendria match.
Si se acepta que el 5% del dominio es la unidad de tamano efectiva de un objeto, entonces habria 19 unidades de 
tamano de ese tipo en el 95% del dominio vacio (es decir 19 × 5% = 95%). Es decir, habria 19 objetos nulos por cada
objeto observado

La definicion de objetos nulos (null objects) nos permite calcular el numero de matches que ocurriran por azar.
Este numero es generalmente expresado como: 

    epsilon = (N_f × N_o)/(N_f + M + D)

donde:

N_f : Numero total de objetos pronosticados
N_o : Nuemero total de objetos observados 
M : Numeros de objetos observados sin match (Misses)
D : es el numero de pronosticos nulos correctos (predecir correctamente que no ocurrira ningun objeto)


D se representa como el numero de observaciones nulas menos el numero de objetos pronosticados:

    D = [(1 - f_A)N_o]/[f_A] - N_f

donde:

f_A : la fraccion de area de un objeto observado 


Utilizando las expresiones anteiores, se puede calcular el GSS para la coincidencia de objetos. Como
se indica en A09, los valores tradicionales del GSS solian rondar por los 0.1. Una razon para la diferencia
entre los dos conjuntos de valores es que no se requiere la superposicion espacial de los objetos pronosticados
y observados para obtener una puntuacion positiva en el GSS basado en objetos, pero dicha superposicion es 
esencial en la aplicacion tradicional del GSS. Segun (A.David,. et al. 2009) ambos de los modelos ahora 
alcanzaron un GSS de 0.42.

##-------------------------------------------------------------------------
## 18. Metodo plot_matched_objects
##-------------------------------------------------------------------------

*Metodo plot_matched_objects : genera una figura de dos paneles (pronostico vs observacion) que muestra:

    1. Contornos de objetos emparejados (colores) y no emparejados (gris).
    2. Lineas conectando objetos emparejados entre paneles ????????
    3. IDs de objetos y valores de interes.
    4. Contexto geografico (lineas de costa, bordes ... etc)


1. Primero filtra objetos por pasos de tiempo:
Se seleccionan los objetos que existen en el tiempo especifico referido como argumento de la funcion
time_idx:

    target_time = self.forecast.time.sel(time=time_idx).values
    fcst_objs = [obj for obj in self.forecast_objects if target_time in obj['time_points']]
    obs_objs = [obj for obj in self.observed_objects if target_time in obj['time_points']] 

2. Luego se identifican matches para ese tiempo:
Donde para cada match global, verifica si ambos objetos exiten en este tiempo especifico

    current_matches = []
    for match in self.matches:
        fcst_obj = next((o for o in fcst_objs if o['id'] == match['forecast_id']), None)
        obs_obj = next((o for o in obs_objs if o['id'] == match['observed_id']), None)
        if fcst_obj and obs_obj:
            current_matches.append({'forecast': fcst_obj, 'observed': obs_obj, 'interest': match['interest']})


3. Se configura la Figura con Cartopy:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                              subplot_kw={'projection': ccrs.PlateCarree()},
                              sharex=True, sharey=True)

Crea dos subplots con proyeccion geografica PlateCarre y sharex=True, sharey=True asegura que ambos paneles 
tengan la misma escala

4. Añade el contexto geografico:

    for ax in [ax1, ax2]:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        # Cuadrícula con etiquetas
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

Se añaden las lineas de costa y bordes de paises.
Se definen los limites geograficos basados en las latitudes de los datos
Cofigura la cuadricula con etiquetas solo en ejes izquierdo e inferior

5. Crea una leyenda de matches:

    legend_elements = []
    for i, match in enumerate(current_matches):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colors[i % len(colors)],
                                        markersize=10, 
                                        label=f'Match {i+1} (Int: {match["interest"]:.2f})'))
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=min(5, len(current_matches)), bbox_to_anchor=(0.5, -0.05))


##-------------------------------------------------------------------------
## 19. Metodo _draw_objects_with_contours
##-------------------------------------------------------------------------  

*Metodo _draw_objects_with_contours: dibuja los contornos de los objetos emparejados(colores) y no
emparejados(grises):

    # 1. Dibujar objetos no emparejados (gris)
    matched_ids = [m[obj_type]['id'] for m in matches]
    unmatched_objs = [o for o in objects if o['id'] not in matched_ids]

    for obj in unmatched_objs:
        self._draw_single_object_contour(ax, obj, 'lightgray')
        # Añadir ID

    # 2. Dibujar objetos emparejados (colores)
    for i, match in enumerate(matches):
        obj = match[obj_type]
        color = colors[i % len(colors)]
        self._draw_single_object_contour(ax, obj, color)
        # Añadir ID
        
        # 3. Conectar objetos emparejados (solo para pronóstico)
        if obj_type == 'forecast':
            paired_obj = match['observed']
            # Dibujar línea punteada entre centroides 


Primero dibuja los objetos no emparejados (fondo), luego los emparejados (primer plano) y las lineas
punteadas entre los objetos emparejados (solo desde el panel de pronostico).

Se calculan los centroides geograficos: 

    centroid_lon = np.mean([self.forecast.lon.values[self._safe_index(c[1], len(self.forecast.lon.values))] 
                      for c in obj['centroid_trajectory']])
    centroid_lat = np.mean([self.forecast.lat.values[self._safe_index(c[0], len(self.forecast.lat.values))] 
                      for c in obj['centroid_trajectory']])

Con lo cual se convierten indices de array a coordenadas geograficas (lon/lat) y usa el promedio de
la trayectoria del centroide para mayor estabilidad.

##-------------------------------------------------------------------------
## 20. Metodo _draw_single_object_contour
##-------------------------------------------------------------------------

*Metodo _draw_single_object_contour : este metodo es el corazon de objetos en MODE, con el cual se convierte
datos abstractos (coordenadas de pixeles) en representaciones visuales significativas (contornos geograficos).


obj_2d = next((o for o in obj['objects_2d'] 
              if o['time'] == pd.Timestamp(obj['time_points'][0])), None)

Encuentra el objeto especifico para el primer paso de tiempo de la trayectoria (time_points[0]). 
plot_matched_objects




##-----------------------------------------------------------------------------
## Analisis Estadistico dentro de MODE
##-----------------------------------------------------------------------------


Analizar la distribución interna de intensidad de lluvia dentro de los objetos identificados, no solo los valores promedios.

Métodos Implementados:

_calculate_object_intensity()


##-------------------------------------------------------------------------
## 1. Metodo _calculate_object_intensity
##-------------------------------------------------------------------------

*Metodo _calculate_object_intensity: extrae los valores reales de precipitacion para cada pixel que compone cada 
objeto.

    Toma un objeto 2D en un tiempo específico

    Accede a las coordenadas de píxeles del objeto (coords_pixel)

    Para cada coordenada (y, x), extrae el valor de precipitación del campo suavizado

    Almacena todos los valores en intensity_values del objeto

Ejemplo:

    Si un objeto tiene 50 píxeles, obtiene 50 valores de precipitación

    Permite calcular estadísticas robustas (cuartiles) no sensibles a valores extremos


##-------------------------------------------------------------------------
## 2. Metodo analyze_precipitation_quantiles
##------------------------------------------------------------------------- 


*Metodo analyze_precipitation_quantiles: analiza la distribucion de precipitacion por cuartiles dentro de los objetos.

Para cada objeto de WRF y GPM:

    Se recopilan todos valores de intensidad de todos los pasos de tiempo de los objetos observados y pronosticados

    Calcula cuartiles Q1(25%), Q2(corresponde con la mediana 50%) y Q3(75%)


    Para los pares emparejados igualmente calcula los cuartiles y compara las diferencias entre los pronosticados
    menos los observados:
        Calcula q1_diff = Q1_WRF - Q1_GPM, etc.

 
##-------------------------------------------------------------------------
## 2.1. Metodo plot_quantile_analysis
##-------------------------------------------------------------------------

##-------------------------------------------------------------------------
## 2.1.1 Metodo plot_quantile_analysis_
##-------------------------------------------------------------------------

*Metodo plot_quantile_analysis: Visualiza el analisis por cuartiles tanto de forma independiente (generando graficos
individuales) como general e influye varios graficos de analisis:

Gráfico 1: Distribución de Cuartiles por Tipo de Objeto
    Se trata de un Boxplot comparando WRF vs GPM para Q1, Q2, Q3

    Donde se muestra:

        Cajas: Representan el rango intercuartílico (Q1-Q3)

        Línea en caja: Mediana (Q2)

        Bigotes: Rango de datos (excluyendo outliers)

    Se puede interpretar de la siguiente manera:

        Si cajas de WRF están más altas: WRF sobreestima intensidad

        Si bigotes más largos en WRF: Mayor variabilidad en el modelo

        Solapamiento de cajas: Similitud en distribución interna

    Por ejemplo:

    GPM: Q1=2.1mm, Q2=4.3mm, Q3=7.8mm
    WRF: Q1=3.5mm, Q2=6.2mm, Q3=10.1mm
    → WRF sobreestima en todos los cuartiles

Gráfico 2: Diferencias de Cuartiles vs Interés
    Se trata de un Scatter plot: interés del match vs diferencia de cuartiles

    Donde se muestra:

        Cada punto es un par emparejado

        Eje X: Interés del match (0-1)

        Eje Y: Diferencia WRF-GPM en mm

    Se puede interpretar de la siguiente manera:

        Puntos arriba de línea y=0: WRF sobreestima

        Puntos cerca de interés=1 con diferencia≈0: Mejores matches

        Tendencia: Si diferencias disminuyen con interés alto → buen comportamiento del modelo

Gráfico 3: Diferencia de Máximos vs Ratio de Área
Se trata de un Scatter: ratio área vs diferencia máximos, coloreado por interés

    Donde se muestra:

        Relación entre tamaño del objeto y error en intensidad máxima

    Se puede interpretar de la siguiente manera:

        Cuadrante superior derecho (área>1, diferencia>0): WRF crea objetos más grandes y más intensos

        Cuadrante inferior izquierdo (área<1, diferencia<0): WRF subestima tamaño e intensidad

        Colores cercanos a amarillo: Matches de alta calidad

Gráfico 4: Distribución de Diferencias por Cuartil
Se trata de Histogramas de diferencias para Q1, Q2, Q3

    Donde se muestra:

        Frecuencia de diferentes magnitudes de error

    Se puede interpretar de la siguiente manera:

        Centrado en 0: Sin sesgo sistemático

        Desplazado a derecha: Sesgo positivo (sobreestimación)

        Ancho de distribución: Variabilidad del error



##-------------------------------------------------------------------------
## 3. Metodo analyze_temporal_persistence
##-------------------------------------------------------------------------

*Metodo analyze_temporal_persistence: Evaluar el tiempo que persisten los sistemas convectivos u objetos en el modelo  
contra las observaciones.

Se calcula el numero de pasos temporales que existe cada objeto y los separa por categorias, es decir, objetos emparejados y no emparejados de WRF y GPM.

Luego se calcula el coeficiente de correlacion en pares emparejados.


##-------------------------------------------------------------------------
## 3.1. Metodo plot_temporal_persistence
##-------------------------------------------------------------------------

Gráfico 1: Boxplot de Persistencia Temporal
    Se generan 6 boxplots comparando diferentes categorias:
        WRF total: todos los objetos del modelo
        GPM total: todos los objetos observados
        WRF Matched: objetos de wrf que encontraron match
        GPM Matched: objetos GPM emparejados
        WRF Unmatched: objetos wrf sin match
        GPM Unmatched: objetos GPM sin match

    Se interpreta como:
        Se compara por ejemplo WRF total con GPM total para ver si el modelo esta reproduciendo correctamente la vida
        de los sistemas en pasos temporales (en nuestro caso, cada paso temporal es de 1 hora). Por ejemplo:
        WRF Total: mediana=3 pasos (9 horas)
        GPM Total: mediana=2 pasos (6 horas)
        → WRF tiene persistencia excesiva

        Se compara Matched y Unmatched los objetos que persisten mas se emparejan mejor?

        Si WRF Matched es mayor que GPM Matched entonces el modelo, por ejemplo, alarga la vida de los objetos 
        o sistemas convetivos que detecta MODE


Gráfico 2: Duración de Pares Emparejados



##-------------------------------------------------------------------------
## 4. Metodo analyze_displacement_distortion
##-------------------------------------------------------------------------

*Metodo analyze_displacement_distortion: caracteriza errores de posicion, tamaño y forma de los objetos 

Muestra la relacion de duracion entre WRF y GPM para los matches donde:
    Relacion 1:1 es la ideal (puntos sobre la linea diagonal)
    Los puntos coloreados por interes del match

Se interpreta como:
    Puntos sobre la linea diagonal WRF mas persistente
    Puntos bajo la diagonal GPM mas persistente
    Puntos con valores de interes altos cerca de la diagonal buena correspondencia temporal

##-------------------------------------------------------------------------
## 4.1 Metodo plot_displacement_distortion
##-------------------------------------------------------------------------

*Metodo plot_displacement_distortion: caracterizar errores de posicion, tamano y forma de los objetos convectivos.

Se calculan:
    Desplazamiento km, es decir, la distancia entre centroides de objetos emparejados:
        displacement_km = sqrt(Δlat² + Δlon²) * 111 #Distancia Euclidea pero en km no en pixeles de rejilla

    Ratio de Areas (area_WRF/area_GPM) donde:
        Mayor que 1 : WRF tiene los objetos mas grandes
        Menor que 1: GPM tiene los objetos mas grandes

    Diferencia de Orientacion, es decir la diferencia angular promedio(grados) solo para objetos no 
    circulares, es decir (excentricidad mayor que 0.2).

    Radio de Excentricidad como: 
        eccentricity_ratios.append(fcst_ecc_mean / obs_ecc_mean if obs_ecc_mean > 0 else 1)
        Es decir, (Exc_wrf/Exc_GPM) para cada objeto que tuvo match

        Si por ejemplo:
        Es 1 entonces los objetos tienen la misma forma
        Si es mayor que 1 entonces WRF tiene los objetos mas alargados
        Si es menor que 1 entonces WRF tiene los objetos mas circulares


##-------------------------------------------------------------------------
## 4.1 Metodo plot_displacement_distortion
##-------------------------------------------------------------------------

*Metodo plot_displacement_distortion: genera los graficos que caracterizan los errores de posicion, tamano y forma de 
los objetos.

Se generan varios graficos:

Gráfico 1: Desplazamiento vs Interés

    Los puntos a la izquierda muestran un desplzamiento pequeno, que indican buena posicion entre los centroides, que
    por supuesto a mayor desplazamiento menor valores de interes.

Por ejemplo:
    Si la mayoria de los puntos con interes mayor a 0.7 tienen desplazamiento menor a 30km, entonces generalmente el 
modelo tiene buena habilidad para la posicion.

Gráfico 2: Ratio de Áreas vs Interés

Se dibujaron lineas de referencia en x=1 donde las areas son iguales y en y=0.7 donde el interes es bueno.

Se puede interpretar como:

    Puntos a la derecha de x=1 WRF sobreestima el tamano de los objetos
    Puntos a la izquiera de x=1 WRF subestima el tamano
    Los puntos con buen interes cercanos a x=1 indican mejor correspondencia de tamano


Gráfico 3: Diferencia de Orientación vs Interés

    Los puntos izquierda (diferencia pequeña): Misma orientación

    Límite físico: Diferencias >90° físicamente imposibles (se usa min(diff, 180-diff))

    Objetos circulares: No se incluyen (excentricidad baja)

La orientación indica la dirección de los sistemas convectivos (ej: líneas de turbonada)


Gráfico 4: Ratio de Excentricidad vs Interés

Ratio ≈ 1: Misma forma en WRF y GPM










##-------------------------------------------------------------------------
## Acceso a los scripts
##-------------------------------------------------------------------------


Estructura:

MODE_verification/
├── run_mode_verification.py         # Script principal (verificación MODE)
├── sensitivity_analysis.py          # Script para análisis de sensibilidad
├── statistical_analysis.py          # Script de análisis estadístico
├── config.py
├── data_loader.py
├── preprocessor.py
├── visualization.py
├── mode_verifier.py
└── utils.py



En el caso del Analisis de sensibilidad se puede correr varias formas:

    1.Desde el scrip princial que ejecuta la verificacion de MODE:

        Solamente se ejucata el estudio de sensibilidad
        python3 ./run_mode_verification --sensitivity-only

        Se ejecuta la verificacion completa y el estudio de sensibilidad
        python3 ./run_mode_verification.py --sensitivity

    2.Directamente desde el script que corre la verificacion:
        pyhton3 ./run_sensitivity_analysis.py
















