
def main():
    with open('train.txt', encoding='utf-8') as f, open('train_modified.txt', 'w', encoding='utf-8') as g:
        for line in f.readlines()[1:]:
            new_line = line.split('\t', 1)[1].rsplit('\t', 1)[0]
            g.write(new_line + '\n')


if __name__ == '__main__':
    main()
