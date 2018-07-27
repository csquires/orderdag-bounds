import yaml


def save_list(l, filename):
    with open(filename, 'w') as f:
        for item in l:
            f.write('%s\n' % item)


def load_list(filename):
    l = []
    with open(filename) as f:
        for line in f.readlines():
            l.append(line.rstrip())
    return l


def save_yaml(o, filename):
    return yaml.dump(o, open(filename, 'w'), indent=2)


def load_yaml(filename):
    return yaml.load(open(filename))


