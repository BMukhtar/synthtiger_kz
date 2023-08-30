import collections

def count_chars(filename, chunk_size=1024*1024):
    char_count = collections.Counter()
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        while chunk := f.read(chunk_size):
            char_count.update(chunk)
    return char_count

filename = './data/archive/kk.txt'
char_count = count_chars(filename)

# Print the 10 most common characters and their counts
print(char_count.most_common(10))

#%%
print(char_count)