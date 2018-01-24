import commands

# Meteor scoring
hypotheses_filename = "./results/hypotheses_2018-01-20_22:10:16-2018-01-21_09:53:21_valid.txt"
references_filename = "./results/original_valid.txt"
status, output = commands.getstatusoutput(
    "java -Xmx2G -jar nlp/meteor-1.5.jar " + hypotheses_filename + " " + references_filename + " -l en -norm")

text_file = open("./results/meteor_2018-01-20_22:10:16-2018-01-21_09:53:21_valid.txt", "w")
text_file.write(output)
text_file.close()

#Bleu scoring
#Implemented scoring with nltk corpus method
