# Adding Romanian Diacritics using Bidirectional LSTM

Diacritics are part of the Romanian identity but are usually dismissed in colloquial speech to favour faster typing. In this project I am proposing a faster alternative to inserting them, using *Bidirectional Long-Short Term Memory* Artificial Neural Networks.

## Understanding the problem

* There are 5 types of diacritics in the romanian language. (Ș /ʃ/, Ă /ə/, Ț /t͡s/, Â /ɨ/, Î /ɨ/ and their lowercase parts)
* Comma-below (ș and ț) versus cedilla (ş and ţ) --  * Many printed and online texts still incorrectly use "s with cedilla" and "t with cedilla". * [[Wikipedia:en@Romanian_alphabet]](https://en.wikipedia.org/wiki/Romanian_alphabet)
* According to the 1993 reform, the choice between î and â is thus again based on a rule that is neither strictly etymological nor phonological, but positional and morphological. The sound is always spelled as â, except at the beginning and the end of words, where î is to be used instead. Exceptions include proper nouns where the usage of the letters is frozen, whichever it may be, and compound words, whose components are each separately subjected to the rule (e.g. ne- + îndemânatic → neîndemânatic "clumsy", not *neândemânatic). [[Wikipedia:en@Romanian_alphabet]](https://en.wikipedia.org/wiki/Romanian_alphabet#%C3%8E_versus_%C3%82)

### Output Targets

- Ă ă (a with breve)
- Â â (a with circumflex)
- Î î (i with circumflex)
- Ș ș (s with comma)
- Ț ț (t with comma)
- Not Diacritic (Ignore / Discard)

Now that we understand the problem at hand, we need to tinker around with the inputs and outputs of our ANN. Since we don't really need 6 outputs, we can simplify our targets using the following format:

| ă or ş or ț 	| î 	| â 	| Not diacritic 	|
|:-----------:	|:-:	|:-:	|:-------------:	|
|      1      	| 0 	| 0 	|       0       	|
|      0      	| 1 	| 0 	|       0       	|
|      0      	| 0 	| 1 	|       0       	|
|      0      	| 0 	| 0 	|       1       	|

## Usage
```
usage: main.py [-h] [--weights WEIGHTS] [--epochs EPOCHS]
               [--timeseries TIMESERIES] [--lstmSize LSTMSIZE]
               [--dropout DROPOUT] [--train TRAIN] [--test TEST]
               action

positional arguments:
  action                Test or Train

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     A filename where to save our weights
  --epochs EPOCHS       The number of epochs to train the neural network for.
  --timeseries TIMESERIES
                        Size of a single timeseries
  --lstmSize LSTMSIZE   The size of our first LSTM layer
  --dropout DROPOUT     Percent Dropout
  --train TRAIN         Train dataset
  --test TEST           Test dataset
```

To train
```
python main.py train --train train.txt --test test.txt --epochs 200
```

For HTTP Backend
```
python main.py serve --weights save.hf
```