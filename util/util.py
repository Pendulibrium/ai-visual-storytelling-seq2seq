def seconds_to_formatted_string(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02dh" % (h, m, s)


def sentences_to_story(open_file_path, save_file_path, num_sentences=5):
    with open(open_file_path) as fp:
        i = 1
        stories = []
        story = ""
        for line in fp:
            line = line.replace("\n", "")
            story = story + line + " "
            if i % 5 == 0:
                stories.append(story)
                story = ""
            i += 1
        # print stories

        stories_file = open(save_file_path, "w")
        stories_with_new_line = map(lambda x: x + "\n", stories)
        stories_file.writelines(stories_with_new_line)
        stories_file.close()
