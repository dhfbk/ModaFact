# ModaFact

This repo provides an inference script for joint Event Factuality and Modality detection in Italian.
The full contribution is documented in our paper [ModaFact: Multi-paradigm Evaluation for Joint Event Modality and Factuality Detection](https://aclanthology.org/2025.coling-main.425/)

In order to tag your texts, you need to:

- download the fine-tuned model from our HF repo: [ModaFact fine-tuned model](https://huggingface.co/dhfbk/modafact-ita). The model has been fine-tuned on the [ModaFact dataset](https://huggingface.co/datasets/dhfbk/modafact-ita), using [mT5-xxl](https://huggingface.co/google/mt5-xxl) as a base model.
- install the requirements in you env via `pip install -r requirements.txt`
- add your Huggingface API key by modifying the script (line 7)

The `predictor.py` script takes as input a text file with one sentence per line (see the test_example.txt file) and outputs a text file with the corresponding annotation, one string per line (corresponding to all annotated spans in the sentence).

Example:

Input (one line): "Per chiarire la questione la Santa Sede autorizzò il prelievo di campioni del legno che vennero datati attraverso l'utilizzo del metodo del carbonio-14."

Output (one line): "chiarire=POSSIBLE-POS-FUTURE-FINAL | autorizzò=CERTAIN-POS-PRESENT/PAST | prelievo=UNDERSPECIFIED-POS-FUTURE-CONCESSIVE | datati=CERTAIN-POS-PRESENT/PAST | utilizzo=CERTAIN-POS-PRESENT/PAST"


The script takes 7 (mandatory) arguments:

- `input_file`: path to input file
- `output_path`: path to the folder where to save ouput file
- `model_checkpoint`: path to the fine-tuned model (previously downloaded, see above)
- `model_name`: name of base model, in this case must be `google/mt5-xxl`
- `batch_size`: number of sentences to be processed in each batch
- `max_in_len`: max input length. The model is set to pad to the longest sentence (in each batch).
- `max_out_len`: max length of the produced output sequence. 

Example:

`python3 predictor.py test_examples.txt /output/ /path/to/finetuned/model/ google/mt5-xxl 2 50 80`



## Caveat

- In order to use this script, you need a GPU with at least 24 GB RAM. You can reset the device (line 10) if you want to execute it on CPU (not tested, though!)
- Since the base model is approx. 52 GB, you may want to cache it, in order not to download it at each execution. By default, the model is NOT cached. Please refer to the comments in the script about how to do it (lines 49 and 66).

- Be aware that `bitsandbytes`, used in the script, might be sensitive to the CUDA version. The script has been tested on a Nvidia A40, with CUDA 12.4 and `bitsandbytes==0.41.0`. 

- The model has been fine-tuned with a max input length of 172 and a max output length of 185 (longest sequences in ModaFact). Longer input sequences might not be fully processed.


## Reference

If you use this repo, please consider citing ModaFact's paper:

```
@inproceedings{rovera-etal-2025-modafact,
    title = "{M}oda{F}act: Multi-paradigm Evaluation for Joint Event Modality and Factuality Detection",
    author = "Rovera, Marco  and
      Cristoforetti, Serena  and
      Tonelli, Sara",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.425/",
    pages = "6378--6396",
    abstract = "Factuality and modality are two crucial aspects concerning events, since they convey the speaker`s commitment to a situation in discourse as well as how this event is supposed to occur in terms of norms, wishes, necessity, duty and so on. Capturing them both is necessary to truly understand an utterance meaning and the speaker`s perspective with respect to a mentioned event. Yet, NLP studies have mostly dealt with these two aspects separately, mainly devoting past efforts to the development of English datasets. In this work, we propose ModaFact, a novel resource with joint factuality and modality information for event-denoting expressions in Italian. We propose a novel annotation scheme, which however is consistent with existing ones, and compare different classification systems trained on ModaFact, as a preliminary step to the use of factuality and modality information in downstream tasks. The dataset and the best-performing model are publicly released and available under an open license."
}

```
