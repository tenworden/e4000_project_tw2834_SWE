# e4000_project_tw2834_SWE
E4000 Project: Predicting the SWE using Transformer and LSTM


# Instructions 

1. Each model implementation has its notebook. There are two models, thus two notebooks for Transformer and LSTM.
2. The Main.ipynb notebook contains the summary of the implementation of two models. Please refer to the individual notebooks for detail. Thus all models are loaded in that notebook to plot the output and predicted output.
3. Under the directory /best_models, you will find all the optimized model weights.
4. Use the requirement.txt to install the package requirements. The file also contains the version of each package.


```  
./
├── LSTM.ipynb
├── Main.ipynb
├── README.md
├── Transformer.ipynb
├── best_models
│   └── transformer_best_model.h5
├── images
│   ├── The_transformer_encoders_decoders.png
│   ├── Transformer_pred.png
│   ├── Transformer_test_pred.png
│   ├── Transformer_train_pred.png
│   ├── data_tab.png
│   ├── embed.png
│   ├── encoder.png
│   ├── lstm-1.png
│   ├── lstm_cell.png
│   ├── lstm_loss.png
│   ├── lstm_pred.png
│   ├── lstm_test_pred.png
│   ├── lstm_train_pred.png
│   ├── multi-head-attention.png
│   ├── obs.png
│   ├── positional.png
│   ├── rnn.png
│   ├── rnn1.png
│   ├── swe_globe.png
│   ├── table.png
│   ├── transformer-1.png
│   ├── transformer.jpeg
│   ├── transformer_loss.png
│   └── transformer_model.png
├── tx_data
│   ├── cmc_sdepth_mly_1998_v01.2.txt
│   ├── cmc_sdepth_mly_2020_v01.2.txt
│   ├── cmc_sdepth_mly_clim_1998to2012_v01.2.txt
│   ├── cmc_swe_mly_clim_1998to2012_v01.2.txt
│   ├── data_filtered_test.txt
│   ├── data_filtered_train.txt
│   └── data_filtered_val.txt
└── utils
    ├── TransformerArchitecture.py
    ├── __pycache__
    │   ├── TransformerArchitecture.cpython-39.pyc
    │   ├── lstm.cpython-39.pyc
    │   ├── lstm_utils.cpython-39.pyc
    │   └── transformer_utils.cpython-39.pyc
    ├── lstm.py
    ├── lstm_utils.py
    └── transformer_utils.py

5 directories, 44 files
```  