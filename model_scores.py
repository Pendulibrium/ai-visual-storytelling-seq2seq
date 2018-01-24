import commands

# Meteor scoring
hypotheses_filename = "./results/hypotheses_2018-01-18_17:39:24-2018-01-20_18:50:39_valid.txt"
references_filename = "./results/original_valid.txt"
status, output = commands.getstatusoutput(
    "java -Xmx2G -jar nlp/meteor-1.5.jar " + hypotheses_filename + " " + references_filename + " -l en -norm")

text_file = open("./results/meteor_2018-01-18_17:39:24-2018-01-20_18:50:39_valid.txt", "w")
text_file.write(output)
text_file.close()

#Bleu scoring
