

def substr_match(str_span, locs):
    for l in locs:
        match = l.find(str_span)
        if match >= 0:
            return True

    return False