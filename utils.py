def print_info(string, filename='vit_info.txt'):
    print(string)
    with open(filename, 'a') as f:
        f.write(string + '\n')