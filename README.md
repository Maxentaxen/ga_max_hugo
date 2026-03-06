# Satellitbildanalys med ConvLSTM

Detta är Max Niemis del av gymnasiearbetet "Kortsiktiga väderprognoser med ML". Här definieras ML-modellen och dess träningsruting samt bildgenerering.


## Projektstruktur

```
conv_lstm_satellite
├──checkpoints
|   └── checkpoint_epoch_50.pth  # Senaste versionen av modellen
├── data                         # All data (45154 undermappar)
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
└──prog.py                         # Program för att ladda ner bilder mellan två datum
```
För att köra predict.py, navigera till /src och kör `python predict.py ÅÅÅÅ/MM/DD/HH `, valfria argument: (`--cmap` följt av valfritt cmap-värde från matplotlib, `--p` följt av 1 eller 0 för lite ascii i början)  

## License

This project is licensed under the MIT License. See the LICENSE file for more details.


https://docs.google.com/document/d/1ORn3F6_LzKLvk1_01Ww31dBzyG5ocg5WCOi-uu-L2bA/edit?tab=t.0
