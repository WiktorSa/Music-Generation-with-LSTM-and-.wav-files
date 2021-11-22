
# Music Generation with LSTM and .wav files

Music Generation with LSTM and .wav files is a deep learning project where the model trains on .wav files and later produces new music based on what it has learned

More info about a project is in the Abstract.pdf  
More info about the training and validation is in the Experiments.pdf  
Link to the files: https://drive.google.com/drive/folders/1L1Ij3y4saLyWctX2h_57knfYE9687jGB?usp=sharing  


## Acknowledgements

 - ["Music generation with deep learning" by Vasanth Kalingeri and Srikanth Grandhe](https://arxiv.org/abs/1612.04928)
  
## Installation

Clone a repository with

```bash
  git clone https://github.com/WiktorSa/Music-Generation-with-LSTM-and-.wav-files my-project
  cd my-project
```

Install all the dependencies with
```bash
  pip install -r requirements.txt
```


    
## Documentation

To generate training data using default parameters put .wav files in raw_audio directory and run

```bash
  python generate_data.py
```

To train your model with default parameters run

```bash
  python train_model.py
```

To generate music with default parameters put .wav files in seeds directory and run

```bash
  python generate_songs.py
```

To see what parameters you can tune run

```bash
  python generate_data.py --help
  python train_model.py --help
  python generate_songs.py --help
```
## FAQ

#### I've got a memory error during training

If you get this error try to reduce the batch size.

#### The training is taking too long

Because of the size of the input and the number of parameters in the model, the model will take a long time to process one batch of data.
To shorten the training time we recommend that you use a smaller dataset and reduce the number of epochs.

  
## Authors

- [@wiktorsadowy](https://github.com/WiktorSa)

  