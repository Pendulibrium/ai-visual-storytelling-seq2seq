import commands
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# Calculating Meteor score
dataset_type = 'train'
results_model_dir = "./results/2018-02-03_14:00:00-2018-02-04_06:15:29/"
beam_size = 1

if beam_size > 1:
    hypotheses_filename = results_model_dir + "hypotheses_" + dataset_type + "_beam" + str(beam_size) + ".txt"
else:
    hypotheses_filename = results_model_dir + "hypotheses_" + dataset_type + ".txt"

references_filename = "./results/original_" + dataset_type + ".txt"

status, output_meteor = commands.getstatusoutput(
    "java -Xmx2G -jar nlp/meteor-1.5.jar " + hypotheses_filename + " " + references_filename + " -t hter -l en -norm")

if beam_size > 1:
    text_file = open(results_model_dir + "meteor_" + dataset_type + "_beam" + str(beam_size), "w")
else:
    text_file = open(results_model_dir + "meteor_" + dataset_type, "w")
text_file.write(output_meteor)
text_file.close()

# Calculating BLEU score
status, output_bleu = commands.getstatusoutput(
    "perl ./nlp/multi-bleu.perl " + references_filename + " < " + hypotheses_filename)

if beam_size > 1:
    text_file = open(results_model_dir + "bleu_" + dataset_type + "_beam" + str(beam_size), "w")
else:
    text_file = open(results_model_dir + "bleu_" + dataset_type, "w")
text_file.write(output_bleu)
text_file.close()
