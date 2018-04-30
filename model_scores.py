import commands
from util import util


# Calculating Meteor score
dataset_type = 'valid'
results_model_dir = "./results/2018-02-09_15:30:08-2018-02-10_01:04:10/"

hypotheses_sentences_filename = results_model_dir + "hypotheses_" + dataset_type + ".txt"
hypotheses_filename = results_model_dir + "hypotheses_story_" + dataset_type + ".txt"

util.sentences_to_story(hypotheses_sentences_filename, hypotheses_filename)

references_filename = "./results/original_story_" + dataset_type + ".txt"

# Calculating METEOR
status, output_meteor = commands.getstatusoutput(
    "java -Xmx2G -jar nlp/meteor-1.5.jar " + hypotheses_filename + " " + references_filename + " -t hter -l en -norm")


text_file = open(results_model_dir + "meteor_story_" + dataset_type, "w")
text_file.write(output_meteor)
text_file.close()

# Calculating BLEU score
status, output_bleu = commands.getstatusoutput(
    "perl ./nlp/multi-bleu.perl " + references_filename + " < " + hypotheses_filename)


text_file = open(results_model_dir + "bleu_story_" + dataset_type, "w")
text_file.write(output_bleu)
text_file.close()
