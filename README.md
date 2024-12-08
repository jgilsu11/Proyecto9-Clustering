# Proyecto9-Clustering



## Entrenamiento de modelos predictivos de los ingresos por ventas en función del tipo de cliente
![Modelos predictivos de los ingresos por ventas en función del tipo de cliente](https://github.com/jgilsu11/Proyecto7-PrediccionCasas/blob/main/Imagen/imagen%20alquiler.webp)  
  

***Descripción:***
El proyecto 9 consiste en la especificación e iteración de modelos predictivos hasta obtener un modelo predictivo óptimo para la predicción de los ingresos por ventas en función del tipo de cliente haciendo uso de archivos.py y Jupyter notebook.

Las técnicas usadas durante el proyecto son en su mayoría aquellas enseñadas durante la novena semana de formación (Preprocesamiento(Gestión de nulos, Duplicados,Encoding, Estandarizacióny Gestión de Outliers) , Clustering, generación y entrenamiento de modelos).

Adicionalmente, se usaron recursos obtenidos mediante research en documentación especializada, vídeos de YouTube e IA como motor de búsqueda y apoyo al aprendizaje.


***Estructura del Proyecto:***

El desarrollo del proyecto se gestionó de la siguiente manera:

- _En Primer lugar_, haciendo uso de JupyterNotebook como primer paso donde realizar ensayos con el código.  

- _En Segundo Lugar_, se creó una presentación basada en los datos.

- _Finalmente_, se realizó la documentación del proyecto en un archivo README (documento actual).

Por todo lo anterior, el usuario tiene acceso a:

        ├── datos/                                       # Donde se guardan los csv que se van generando en cada modelo de cada cluster 
        ├── Imagen/                                      # Imagen para su uso en el readme       
        ├── Notebooks/                                   # Notebooks de Jupyter donde se han ido desarrollando los modelos con su preprocesamiento (clusters incluidos)     
        ├── src/                                         # Scripts (.py)
        ├── README.md                                    # Descripción del proyecto
                 
        
***Requisitos e Instalación🛠️:***

Este proyecto usa Python 3.11.9 y bibliotecas que se necesitarán importar al principio del código como:
- [pandas](https://pandas.pydata.org/docs/)
- [numpy](https://numpy.org/doc/2.1/)
- [matplotlib](https://matplotlib.org/stable/index.html)
- [matplotlib-inline](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html)
- [seaborn](https://seaborn.pydata.org/)
- [requests](https://requests.readthedocs.io/en/latest/)
- [sys](https://docs.python.org/3/library/sys.html)
- [os](https://docs.python.org/3/library/os.html)
- [sklearn](https://scikit-learn.org/stable/)
- [tqdm](https://tqdm.github.io/)
- [warnings](https://docs.python.org/3/library/warnings.html)
- [pandas.options.display](https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html)


**Resumen de resultados:**    
- Se identificaron 3 tipos de clientes distintos:  
    - Clientes procedentes de zonas geográficas donde no hay mucho volúmen de pedidos (cluster 0)  
    - Clientes procedentes de zonas geográficas donde hay mucho volúmen de pedidos (cluster 2)  
    - Clientes procedentes de ciudades de EEUU (zona con gran volúmen de pedidos) cuyas ciudades son de bajo volúmen de pedidos  

- Se concluyó que las mejores variables para predecir los ingresos por ventas son:  
    - Las sub-categorías de los productos  
    - Las categorías de los productos  
    - La ciudad del usuario que realiza el pedido  
    - El país del usuario que realiza el pedido  
    - Las cuantías de descuento aplicados a los pedidos      
  
***Aportación al Usuario🤝:***

El doble fin de este proyecto incluye tanto el propio aprendizaje y formación como la intención de crear modelos predictivos de los ingresos por ventas en función del tipo de cliente para poder ayudar a la empresa a mejorar.


***Próximos pasos:***

En un futuro, se recomienda ser más exhaustivo y variado en el preprocesamiento, así como incluir el "Ship Cost" como variable explicativa en los modelos y especificar más modelos para poder hacer un predicción más precisa. La herramientas que más útiles pueden ser son el uso de otras formas de machine learning, inteligencia artificial u otras opciones.
