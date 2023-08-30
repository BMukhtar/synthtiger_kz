import random

# Read words from kk_dict.txt
with open('resources/corpus/kk_dict.txt', 'r', encoding='utf-8') as f:
    words = f.read().splitlines()

# Define the symbols
symbols = '0123456789!\'#$%()*+/,-.:;<=>? €АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЁабвгдежзийклмнопрстуфхцчшщъыьэюяёӘҒҚҢӨҰҮІҺәғқңөұүіһ'

# Define maximum and minimum sequence length
max_n = 80
min_n = 2

# Initialize character counts
char_counts = {char: 1 for char in symbols}

# Filter out words that use different symbols
words = [word for word in words if all(char in symbols for char in word)]

# Define a function to get a random character with probability inversely proportional to its count
def get_random_char():
    min_count = min(char_counts.values())
    candidates = [char for char, count in char_counts.items() if count == min_count]
    return random.choice(candidates)

# Generate corpus
corpus = []
for i in range(200_000):  # Adjust the range as needed
    sequence = ''
    target_length = random.randint(min_n, max_n)
    while len(sequence) < target_length:
        if random.random() < 0.8:  # Adjust the probability as needed
            word = random.choice(words)
            for char in word:
                sequence += char
                char_counts[char] += 1
                if len(sequence) == target_length:
                    break
            if len(sequence) < target_length:
                sequence += ' '
                char_counts[' '] += 1
        else:
            symbol = get_random_char()
            sequence += symbol
            char_counts[symbol] += 1
    corpus.append(sequence)

# Write corpus to corpus.txt
path = "./resources/corpus/kz_corpus_generated.txt"
with open(path, 'w', encoding='utf-8') as f:
    for sequence in corpus:
        f.write(sequence + '\n')

print(f'Corpus generated and written to {path}')