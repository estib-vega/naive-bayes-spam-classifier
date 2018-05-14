def extract_free_string_from_mail(mail_name):
    file = open(mail_name, "r", encoding="utf-8", errors="ignore")

    # here we will store the clean text
    # only letters
    content = []

    shouldAppend = True
    for raw_line in file:
        # for each line in the file
        line = raw_line.strip()
        if len(line) > 0:
            for c in line:
                # for each character in the line
                if shouldAppend:
                    #initially we should be able to append
                    if ord(c) != 60:
                        if ord(c) >= 65 and ord(c) <= 90:
                            # if the character is a lowercase character or...
                            content.append(c)
                        elif ord(c) >= 97 and ord(c) <= 122:
                            # the character is a uppercase letter
                            content.append(c)
                        else:
                            # else we append a space, if there isn't already a space
                            # at the las position of content
                            if content[::-1] != ' ' : content.append(' ')
                    else:
                        # if the character is '<' then we stop until 
                        # we find the closure '>'
                        shouldAppend = False
                else:
                    if ord(c) == 62:
                        # closure '>' found, start appending again
                        shouldAppend = True

    return "".join(content)