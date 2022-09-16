import unicodedata

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )



print(unicode_to_ascii('d'))

unicodedata.numeric('g')