## SIF_partitioning

This repo shows the code & data the SIF partitioning project, in which we show that neural networks informed by SIF observations, $NN_{SIF}$, can successfully be used to partition ecosystem carbon fluxes and retrieve the GPP–SIF relationship at the ecosystem scale.

### Code structure

* The `NNSIF_partitioning.ipynb` file shows the complete process to preprocess data, train the $NN_{SIF}$ model, and evaluate the trained model.

* The directory `./utils/` stores functions for model training and data preprocessing.

* The directory `./data/` stores the SCOPE simulation data, which are needed to run the above codes.


### Project Dependencies

* Python 3.6+
* Tensorflow 2.2.0
* Keras version: 2.3.0