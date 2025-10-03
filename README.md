# Satellitbildanalys med ConvLSTM

Detta är Max Niemis del av gymnasiearbetet "Kortsiktiga väderprognoser med ML". Här definieras ML-modellen och dess träningsruting samt dataöverfaring till och bort.


## Projektstruktur

```
conv_lstm_satellite
├── src
│   ├── dataset
│   │   └── satellite_dataset.py  # Egen klass för att skapa datasetet
│   ├── models
│   │   └── convlstm_network.py     # Nätverkets definition
│   ├── train.py                    # Modellens träningsrutin
│   └── utils
│       └── __init__.py            # init-fil
└── config.yaml                     # Konfigurationsfil för hyperparametrar samt diverse filsökvägar
```


## License

This project is licensed under the MIT License. See the LICENSE file for more details.


https://docs.google.com/document/d/1ORn3F6_LzKLvk1_01Ww31dBzyG5ocg5WCOi-uu-L2bA/edit?tab=t.0
