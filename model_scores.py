import commands
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# Calculating Meteor score
dataset_type = 'valid'
results_model_dir = "./results/2018-01-29_00:37:26-2018-01-31_00:01:42/"
hypotheses_filename = results_model_dir + "hypotheses_" + dataset_type + ".txt"
references_filename = "./results/original_" + dataset_type + ".txt"

status, output_meteor = commands.getstatusoutput(
    "java -Xmx2G -jar nlp/meteor-1.5.jar " + hypotheses_filename + " " + references_filename + " -l en -norm")

text_file = open(results_model_dir + "meteor_" + dataset_type, "w")
text_file.write(output_meteor)
text_file.close()

# Calculating BLEU score
status, output_bleu = commands.getstatusoutput(
    "perl ./nlp/multi-bleu.perl " + references_filename + " < " + hypotheses_filename)
text_file = open(results_model_dir + "bleu_" + dataset_type, "w")
text_file.write(output_bleu)
text_file.close()
