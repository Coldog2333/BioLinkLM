import os


def get_n_line(filename, dump=False):
    if os.path.exists(filename + '.nline'):
        n_line = int(open(filename + '.nline').read())
    else:
        print(f"{filename + '.nline'} not found. Counting lines...")
        n_line = int(os.popen('wc -l %s' % filename).read().split(' ')[-2])
        if dump:
            open(filename + '.nline', 'w').write(str(n_line))
    return n_line
