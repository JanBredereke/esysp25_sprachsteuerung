# KI

Classification of spoken words to be able to control a remote-control car. Supported words are "left", "right", "forward", "stop", and "back".

## Data

[Google Drive](https://drive.google.com/drive/folders/1o2vysWZpr3EBviqZPcHCt85LuCLRrPBp?usp=sharing)

The data is also located in the repository under the "data" folder.

## Installation (Windows)

First install Python 3.8.10 (https://www.python.org/downloads/windows/)

There are two external dependencies you need to install before pienv.
 - [ffmpeg](https://ffmpeg.org/ or https://www.gyan.dev/ffmpeg/builds/) for loading MP3s using PyDub. The \bin folder needed to be added to the PATH
 - [portaudio](https://people.csail.mit.edu/hubert/pyaudio/) for inference using a microphone on a development machine.

Pipenv is used to install dependencies so install pipenv first:
```
pip install pipenv
```

After that you can use pipenv install in the root of the project and all dependencies:
```
pipenv install
```
You may need to manually install a dependency for the finn library as it is not currently available on PyPI.
To do so plese clone the finn repository:
```
git clone https://github.com/Xilinx/finn/
```
After that you need to copy the file
```
/finn/src/finn/util/visualization.py
```
into your virtual environment under the path
```
<your_virtual_environment_path>\Lib\site-packages\finn\util\
```


Start pipenv:
```
pipenv shell
```

If you want to use CUDA to enable GPU support you should install PyTorch with CUDA (otherwise PyTorch is already installed with the dependencies and you can skip this step):

```
pip3 install torch==1.13.1+cu117 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

## Start the training

There are different parameter with which you can start EmbedsKi.py. 
You should start with synthesising audio data and training:

```
python EmbedsKi.py --synthesis --train
```

If you want to train a specific model, you can specify it with the parameter ``--modelName``. Then the model is selected from the collection of different models that exists. If you don't use this parameter, the default value is "DefaultModel". <br/>
You can specify multiple model names a once, so to train multiple models, you just need to do data preprocessing once. If you append ``_X`` to the model name, where "X" is a placeholder for any word, character or number, a different version of the model is created in a different directory. This is for the case, that you just want to change a few parameter but not the whole architecture of the model.<br/>
The models you can train are defined in the ``neuralNetwork/Model.py`` file.

### Select device for training
In normal use of this project, the device for the training is used automatically. If possible, the GPU is used for the training, otherwise the CPU. To be able to train the model on the CPU by choice, you can add the parameter ``--cpu`` to the training.

## Verify the model
Because the training failed frequently because of a wrong model structure, I added the parameter ``--verify`` to check the structure once before training, to lose less time. With this parameter, all models are forwarded once, to be sure, this is not the reason of failure. Models that fail this check, are not trained

## Observing training

Other than the log outputs training data is also written into the "runs" folder using tensorboard.
See [Tensorboard](https://www.tensorflow.org/tensorboard).

```
tensorboard --logdir=runs
```

## Test the trained model
To test the trained model on an audio file, you can use the following command:

```
python EmbedsKi.py --test --audioFile=<path_to_audio_file>
```

## Use the trained model

Now you can test the trained model with:

```
python EmbedsKi.py --test
```

The system will now start to record 4 seconds of audio snippets and use the model to classify the spoken word. The result will be printed to the console. If you want to use a specific model, you can add the parameter ``--modelName`` with the name of the model you want to use.

## Add another Model
To add another model, just create a new class in the ``neuralNetwork/Model.py`` file and extend it from the SuperModel class. In the ``__init__()`` function you need to specify the layers of the network. The name of the model is the classname.

## Export the Model
To export any model to finn-onnx format, use the ``--convert`` param in combination with a model name: eg ``--modelName=DefaultModel``.<br/>
If there is an error saying
```
TypeError: export() got an unexpected keyword argument 'enable_onnx_checker'
```
comment line 77 of the ``manager.py`` of the brevitas library and try again. <br/>
To get a preview of the converted model, you can add the parameter ``--showOnnx`` to the script. A Netron instance will start and is available on ``127.0.0.1:8080``.

## Convert an Audiofile to an npy-File

To do the conversion use ``--convertToNpy`` in combination with ``--audioFile``. The result will be an file like: ``outputInt8_x.npy``. To use this to test the interference on the PYNQ-Board you further need to convert it. Use ``convertToint4.py`` where you pass the file as a parameter to the Skript.

## Troubleshooting

- If you use CUDA and you get the error "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!" you need to delete all existing checkpoints.
