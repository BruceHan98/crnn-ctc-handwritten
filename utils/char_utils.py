
def get_char_dict(char_dict_path):
    char_dict = dict()
    with open(char_dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.strip()
            if char:
                char_dict[char] = len(char_dict)
    return char_dict


def char2id(char, char_dict):
    try:
        return char_dict[char]
    except Exception as e:
        print(e)


def id2char(id_, char_dict):
    try:
        return list(char_dict.keys())[list(char_dict.values()).index(id_)]
    except Exception as e:
        print(e)
