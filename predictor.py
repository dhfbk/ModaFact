from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import nltk, torch, argparse, os, logging
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
nltk.download('punkt')
access_token = "" # HERE YOUR HUGGINGFACE API KEY

# set device
device = torch.device('cuda:0')

# parse args
parser = argparse.ArgumentParser()

#  Arguments
parser.add_argument("input_file", metavar="input-file", help="Path to file containing input sentences, one sentence per line.")
parser.add_argument("output_path", metavar="output-path", help="Path to output folder where results will be saved.")
parser.add_argument("model_checkpoint", metavar="model-checkpoint", help="Path to fine-tuned model.")
parser.add_argument("model_name", metavar="model_name", help="Name of pre-trained LLM.")
parser.add_argument("batch_size", metavar="batch-size", help="Number of samples to be processed in each batch.")
parser.add_argument("max_in_len", metavar="max-in-len", help="Max input length.")
parser.add_argument("max_out_len", metavar="max-out-len", help="Max output length.")

args = parser.parse_args()

input_file = args.input_file
output_path = args.output_path
model_checkpoint = args.model_checkpoint
model_name = args.model_name
batch_size = int(args.batch_size)
max_in_len = int(args.max_in_len)
max_out_len = int(args.max_out_len)

# log arguments
logging.info(f"Input file: {args.input_file}")
logging.info(f"Output path: {args.output_path}")
logging.info(f"Model Checkpoint: {args.model_checkpoint}")
logging.info(f"Model name: {model_name}")
logging.info(f"Batch size: {batch_size}")
logging.info(f"Max input sequence length: {args.max_in_len}")
logging.info(f"Max output sequence length: {args.max_out_len}")

# loading data
logging.info("Loading data...")
data = open(input_file, 'r').read().splitlines()

logging.info("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_name,
											#cache_dir='',
											token=access_token
										)  # Since the model is heavy, you can cache it by specifying the argument cache_dir="path_to_cache_directory"
tokenizer.pad_token_id = tokenizer.eos_token_id

# define tokenizer and base model
bnb_config = BitsAndBytesConfig(
			load_in_8bit=True,
			bnb_8bit_use_double_quant=True,
			bnb_8bit_quant_type="nf4",
			bnb_8bit_compute_dtype=torch.bfloat16
			)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
			model_name,
			quantization_config=bnb_config,
			device_map='auto',
			#cache_dir='',
			token = access_token
		) # Since the model is heavy, you can cache it by specifying the argument cache_dir="path_to_cache_directory"


# Load fine-tuned model
logging.info("Loading fine-tuned model...")
model = PeftModel.from_pretrained(base_model, model_checkpoint)

logging.info("PREDICTING...")
i = 0
j = batch_size
processed = 0
all_predictions = []

while i < len(data):

	sentence_batch = data[i:j]

	model_inputs = tokenizer(sentence_batch, max_length=max_in_len, padding='longest', is_split_into_words=False, return_tensors='pt').to(device)
	outputs = model.generate(input_ids=model_inputs.input_ids, max_new_tokens=max_out_len)

	for output in outputs:
		decoded = tokenizer.decode(output, max_length=max_out_len, skip_special_tokens=True)
		all_predictions.append(decoded)
		logging.info(decoded)
	processed += len(sentence_batch)
	logging.info(f"Processed: {processed} samples.")

	i += batch_size
	j += batch_size


logging.info(f"Processed {len(all_predictions)} sentences.")
logging.info(f"Saving predictions at {output_path}...")

model_name = model_name.replace('/', '_') # !

os.makedirs(output_path, exist_ok=True) # create output folder if it doesn't exist

with open(output_path+'prediction__'+model_name+'.txt', 'w') as outfile:
	outfile.writelines([pred + '\n' for pred in all_predictions])

logging.info("Done!")
