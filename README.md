## SIF_partitioning

This repository shows the codes & data of the SIF partitioning project, in which we show that neural networks informed by SIF observations, NN_SIF, can successfully be used to partition ecosystem carbon fluxes and retrieve the GPP–SIF relationship at the ecosystem scale.

### Structure of the repository

`
.
├── README.md
├── codes
│   ├── NNSIF_partitioning_whole_process.ipynb
│   └── utils
│       ├── NNSIF_model.py
│       ├── plot.py
│       └── preprocess.py
└── input_data
    ├── Readme.txt
    └── test_data_SCOPE.csv
`


* The `NNSIF_partitioning_whole_process.ipynb` file shows the complete process to partition CO2 fluxes using the NN_SIF model, which includes: 1) data preprocessing, 2) model training, 3) model prediction, and 4) model evaluation.

* The directory `./codes/utils/` stores the detailed functions for data preprocessing (preprocess.py), model training (NNSIF_model.py) and figure plotting (plot.py).

* The directory `./input_data/` provides the input data which can be used to run and test the above codes.


### Example data
We provide an example of data in order to try and test the codes (`./input_data/test_data_SCOPE.csv`). The data were obtained based on SCOPE simulations, which can provide the "truth" value of GPP and ER. The explanation for each variable in the example data can be found in `./input_data/Readme.txt`. These data are provided only and exclusively for the test of the NN_SIF codes, and can not be used for other applications. 


### Project Dependencies

* Python 3.6+
* Tensorflow 2.2.0
* Keras version: 2.3.0