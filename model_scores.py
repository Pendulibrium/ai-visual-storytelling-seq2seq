import commands
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


# Meteor scoring
hypotheses_filename = "./results/hypotheses_2018-01-18_17:39:24-2018-01-20_18:50:39_train.txt"
references_filename = "./results/original_train.txt"

status, output = commands.getstatusoutput(
    "java -Xmx2G -jar nlp/meteor-1.5.jar " + hypotheses_filename + " " + references_filename + " -l en -norm")

text_file = open("./results/meteor_2018-01-18_17:39:24-2018-01-20_18:50:39_train.txt", "w")
text_file.write(output)
text_file.close()

#Bleu scoring

with open(references_filename) as f:
    references = f.readlines()
references = [x.strip() for x in references]
references = [[x.split()] for x in references]

with open(hypotheses_filename) as f:
    hypotheses = f.readlines()
hypotheses = [x.strip().split() for x in hypotheses]


print(corpus_bleu(references, hypotheses) * 100)




