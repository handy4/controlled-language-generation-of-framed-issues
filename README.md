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

To generate outputs, run:

```
python -u evaluate_formality.py --ckpt ckpt/formality/predictor_gyafc_entertainment_music/model.pth.tar --dataset_info ckpt/formality/predictor_gyafc_entertainment_music/dataset_info --in_file formality_data/fisher_test_oracle.es --model_path ckpt/formality/marian_finetune_fisher > formality_preds.log
```

The above command generates predictions using the Marian model finetuned on the Fisher dataset; remove the `--model_path` argument to get predictions with the un-finetuned Marian model from HuggingFace (referred to as 0-shot in the paper)

Then evaluate metrics using:

```
python eval_formality_metrics.py --pred formality_preds.log --ref formality_data/test.noid.cleaned_0 formality_data/test.noid.cleaned_1 --ckpt ckpt/formality/test_evaluator_gyafc_family_relationships/model.pth.tar --dataset_info ckpt/formality/test_evaluator_gyafc_family_relationships/dataset_info
```

### Training your own predictors

Example command below. (Reminder: you need to go get the GYAFC dataset following the instructions in https://github.com/raosudha89/GYAFC-corpus.)

```
python -u main.py --task formality --data_dir train_data/GYAFC_Corpus/Entertainment_Music --save_dir ckpt/formality/formality_retrain_predictor --num_workers 20 --batch_size 32 --epoch_max_len 1000000 --validation_freq 1 --lr 2e-5 --epochs 20 > formality_retrain_predictor.log
```

(The test-time formality evaluator is trained in the same way, just using the Family/Relationships half of the GYAFC dataset.)

The same evaluation commands as before will work; just modify the paths in the command to point to `model_best.pth.tar`, `dataset_info`, and `rhyme_info` from your newly trained ckpt folders. 

## Running FUDGE on your own data

The code has been refactored so that the iambic (poetry), rhyme (poetry), newline (poetry), future word (topic), and formality (machine translation) are controlled by the `--task` flag to `main.py`. You should add your task as another option here, then modify the data processing in `data.py` and the model in `model.py` as needed for your task. (In `data.py` you probably won't need all the entries of the tuple that is expected of the loader; you can just put dummy entries in the ones you don't need.) You might also need to modify the loss computation in the `train` and `validate` functions in `main.py`. You'll probably want to write new evaluation scripts, though the existing poetry/topic/formality ones are hopefully helpful as references. 

Alternatively, the general FUDGE framework is pretty simple, so you could always try reimplementing things yourself. A few additional details based on questions I've received: 

(1) The formality task setup is likely closest to what you want if you're just trying to run the simplest form of FUDGE (take a language model, and use a classifier to optimize toward a single attribute) although you may need to swap out the Marian translation model/tokenizer we use. 

(2) When you construct your training data, if you have an example in your data e.g. "This movie is great!" for positive sentiment, you want to learn on all the pairs (This, +), (This movie, +), (This movie is, +), etc., as that's one of the main points of our approach. 

(3) For computational efficiency, we first filter the base model's next token probabilities down to the top 200 (Sec. 3.1 in the paper), before adding the classifier logits. This way you only need to evaluate your classifier on 200 continuations. Then afterward, you filter down again to whatever top-k/greedy/nucleus sampling you're using for evaluation (we use top-k with k=10 for poetry and topic, greedy for formality). 

(4) You can use a pretrained LM backbone instead of a simple LSTM backbone for the predictor as well. This should work better when your dataset is smaller. 
