checker = [
    'xxx', 'xxxx', 'zz', 'b', 'd', 'oop', 'ayn', 'ad', 't', '[laughter]',
    '<se>', 'analytical<anal>', 'c', 'oaxaca', 'p', 'ca', 'ci', 'comme', 'mani',
    'pedi', 'x', 'ba', '<calm>calms', '<s>still', 'y', 'z', 'de', 'g', 'ws',
    '<deep', 'breath>'
]



def find_odd_words(data):

    weird_words = []
    for i, strings in enumerate(data):
        for p, v in enumerate(strings):
            if isinstance(v, str):
                if v in checker:
                    weird_words.append(f"{v}:{i, p}")
            else:
                inter_v = v.split()
                for k in inter_v:
                    if k in checker:
                        weird_words.append(f"{v}:{i, p}")

