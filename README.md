# Controlled Language Generation of Framed Issues

This repo contains the code corresponding to my bachelor's thesis Controlled Language Generation of Framed Issues.

## Setup/Installation

This code was tested on Python 3.6. However, newer versions should work too.

To run the code in this repository, you need to create three different Python environments.

For all three environments, first install pytorch according to your system specifications following https://pytorch.org/get-started/locally/

Then, for the first one, which is used for training and generation of CTRL and BART models and overall evaluation, use the requirements file in the root folder. For the second one, which uses adapter-transformers instead of base transformers, is used for training silver-corpus models and generating the silver samples itself. The corresponding requirements file can be found in [silver/](silver/)

The implementation of FUDGE is based on the original code by Kevin Yang and Dan Klein. For setup instructions, please refer to https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation

## Model Training and Evaluation

### Training

To train the BART and CTRL models, in [train_generate/](train_generate/), run train_bart.py and train_ctrl.py respectively. Most arguments are currently still hard-coded into the files, so if you want to change file paths or model hyperparameters, please refer to the code.

To train FUDGE, please refer to the instructions at https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation and use the newly added task names ftopic and fframe. To change the predictor architectures and forward logic, please change their respective code in main.py and model.py

You can find a sample input file in the data folder to understand the required input format.

### Generation

To generate issue-framed sentences for the issue and frame classes in your test data with the BART and CTRL models, in [train_generate/](train_generate/), run gen_bart.py and gen_ctrl.py respectively. To change file paths and generation strategy, you have to modify the code as well.

To generate sentences with FUDGE, refer to the original repository for instructions and use the new script evaluate_frame.py .
Rhyme predictor:

### Evaluation

To evaluate the generated sentences against a gold sample (same ordering as input to generation script is important!) use the scripts eval_bart.py, eval_ctrl.py and eval_fudge.py in [eval/](eval/). Check the top of the files to set paths to input and gold samples.

## Silver Corpus Generation

### Train frame classifier

To train a frame classification adapter, run train_adapter_model in [silver/](silver/) with framing data labeled with frame values from 0 to 15 as in the data sample file.

### Generate silver samples

There are two different scripts for generating silver data. One is configured to use the Reuters Corpus Volume 1 as input. To use it, place the rcv1 data in rct-input and run silver-generation-pipeline-rcv.py.

The other is configured to extract content from WebArchive files. To use it, run silver-generation-pipeline.py and supply the name of a warc file in the folder pipeline-input as an argument (without the .warc extension!). This then outputs the extracted silver samples as a file of the same name into the folder pipeline-output.
