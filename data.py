import string


def encoding_dic():
    encoding = {}
    decoding = {}
    for i, v in enumerate(string.ascii_lowercase):
        encoding[v] = i + 1
        decoding[i + 1] = v
    encoding["."] = 0
    decoding[0] = "."
    return encoding, decoding


def name_to_input(name, context_size, dic):
    name = "." * context_size + name + "."
    input_context = []
    target = []
    for i in range(0, len(name) - context_size):
        window = []
        for char in name[i : i + context_size]:
            window.append(dic[char])
        input_context.append(window)
        target_char = name[i + context_size]
        target.append(dic[target_char])
    return input_context, target


def load_data(filename, context_size):
    with open(filename) as file:
        names = [line.rstrip().lower() for line in file]
    input_context = []
    target = []
    encoding, _ = encoding_dic()
    for name in names:
        inputs, targets = name_to_input(name, context_size, encoding)
        input_context.extend(inputs)
        target.extend(targets)
    return input_context, target
