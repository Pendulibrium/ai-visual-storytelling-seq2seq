def seconds_to_formatted_string(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02dh" % (h, m, s)
