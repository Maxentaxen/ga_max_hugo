# Hämtning och analys av sattelitbilder med ConvLSTM

Inuti image_downloader defineras hämtningen av sattelitbilder från eumetsat.int. Detta är Hugo Larsson del av gymnasiearbetet "Kortsiktiga väderprognoser med ML".

Inuti conv_lstm_satellite definieras ML-modellen och dess träningsruting samt bildgenerering. Detta är Max Niemis del av gymnasiearbetet "Kortsiktiga väderprognoser med ML".


## Projektstruktur

```
conv_lstm_satellite
├──checkpoints
|   └── checkpoint_epoch_50.pth  # Senaste versionen av modellen, används i predict.py
├── data                         # Bilderna sparas och hämtas härifrån
|   ├── 2020
|   ├── 2021
|   ├── 2022
|   ├── 2023
|   ├── 2024
|   └── 2025
├── predictions
|   ├──  diffs                     # Differensbilder som sparas efter att man kör predict.py
|   ├──  preds                     # Bilderna som genereras när man kör predict.py
|   └── targets                    # Målbilden som kopieras hit när man kör predict.py
├── src
|   ├── checkpoints # Här sparas nätverket efter varje epok
│   ├── dataset
│   │   └── satellite_dataset.py  # Egen klass för att skapa datasetet
│   ├── models
│   │   └── convlstm_network.py     # Nätverkets definition
|   ├── predict.py                  # Kör nätverket vid en viss tidspunkt
│   ├── train.py                    # Modellens träningsrutin
|   ├── viewpreds.py                # Visar alla bilder i /predictions i ett rutnät
│   └── utils
│       └── __init__.py            # init-fil
image_downloader
└──downloader.py                         # Program för att ladda ner bilder mellan två datum
```
För att köra predict.py, navigera till /src och kör `python predict.py ÅÅÅÅ/MM/DD/HH `, valfria argument: (`--cmap` följt av valfritt cmap-värde från matplotlib, `--p` följt av 1 eller 0 för lite ascii i början)  

Bildhämtningen i downloader.py konfigureras via parametrar på rad 6–26 där användaren kan ange WMS-endpoint (WMS_URL), bildlager (LAYER), tidsperiod och tidssteg för nedladdning (START_TIME, END_TIME, TIME_STEP), startindex för filnumrering (count), samt geografiskt bounding box (BBOX) och bildens upplösning (WIDTH, HEIGHT).
## License

This project is licensed under the MIT License. See the LICENSE file for more details.


https://docs.google.com/document/d/1ORn3F6_LzKLvk1_01Ww31dBzyG5ocg5WCOi-uu-L2bA/edit?tab=t.0
