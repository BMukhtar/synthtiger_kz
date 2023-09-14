import random
import math
from datetime import datetime, timedelta

# Define the symbols
afterword_symbols = "!?.,:;"
numbers = "0123456789"
other_symbols = "'#()<>+-/*=%$"
space_symbol = ' '
all_letters = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЁабвгдежзийклмнопрстуфхцчшщъыьэюяёӘҒҚҢӨҰҮІҺәғқңөұүіһ'
all_symbols = numbers + afterword_symbols + other_symbols + space_symbol + all_letters
print(all_symbols)

# Define maximum and minimum sequence length
max_n = 25
word_count = 1_000_000

# Initialize character counts
char_counts = {char: 1 for char in all_symbols}

# Read words from kk_dict.txt
with open('resources/corpus/kk_dict.txt', 'r', encoding='utf-8') as f:
    words = f.read().splitlines()

# Filter out words that use different symbols
words = [word for word in words if all(char in char_counts for char in word)]

def handle_less_frequent_letters():
    # Filter out lowercase letters from char_counts
    chars = set([l.lower() for l in all_letters])
    lowercase_counts = {k: v for k, v in char_counts.items() if k in chars}

    # Sort the dictionary by frequency (value), and take the 10 less frequent lowercase letters
    sorted_counts = sorted(lowercase_counts.items(), key=lambda x: x[1])
    least_frequent_chars = [char for char, count in sorted_counts[:10]]

    # Generate a random word length between 3 and 10
    word_length = random.randint(3, 10)

    # Create the word by randomly selecting characters from the least_frequent_chars list
    word = ''.join(random.choice(least_frequent_chars) for _ in range(word_length))

    return word

def get_scaled_number(min = 1, max = 100_000):
    log_rand = random.uniform(math.log(min), math.log(max))
    rand_val = math.exp(log_rand)

    # Round to the nearest integer
    return str(round(rand_val))


# Function to get random number with additional formatting
def get_random_number():
    number = get_scaled_number()
    decimal_point = "." + str(random.randint(1, 100))
    post_fix = random.choice(["$", "%", "", "", decimal_point])
    return random.choice(["", "", "-"]) + str(number) + post_fix


def handle_math_expression():
    return space_symbol + get_scaled_number() + random.choice(
        ["+", "-", '/', '*']) + get_scaled_number() + "=" + get_scaled_number()


def handle_hashtag():
    return "#" + random.choice(words)

def handle_date():
    start_date = datetime(1900, 1, 1)
    end_date = datetime(2099, 12, 31)

    delta = end_date - start_date
    random_days = random.randint(0, delta.days)

    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime('%Y-%m-%d')

def handle_time():
    return str(random.randint(1, 24)) + ':' + str(random.randint(1, 59))


def handle_other():
    n = random.randint(1, 10)
    if n == 1:
        return handle_hashtag()
    if n == 2:
        return random.choice(["(", ""]) + random.choice(words) + random.choice([")", ""])
    if n == 3:
        return "<" + random.choice(words) + ">"
    if n == 4:
        tik = random.choice(["'", "''"])
        return tik + random.choice(words) + tik
    if n == 5:
        return random.choice(words) + "-"
    if n == 6:
        return handle_date()
    if n == 7:
        return handle_time()
    if n == 8:
        return random.choice(["(", ""]) + get_scaled_number(1, 3000) + "-" + get_scaled_number(1, 3000) + random.choice([")", ""])
    if n == 9:
        return random.choice(words) + "-" + random.choice(words)
    if n == 10:
        return "т.б."

def get_candidate() -> str:
    word = random.choice(words)
    if random.random() < 0.8:
        return word

    choice = random.random()
    # Append a special symbol
    if choice < 0.2:
        return random.choice(words) + random.choice(afterword_symbols)
    # Append a number
    elif choice < 0.4:
        number = get_random_number()
        return number
    elif choice < 0.45:
        return handle_math_expression()
    elif choice < 0.6:
        return handle_less_frequent_letters()
    # Append other symbols
    else:
        return handle_other()


def main():
    # Generate corpus
    corpus = []
    for i in range(word_count):  # Adjust the range as needed
        sequence = ''
        target_length = max_n
        remaining_words = random.randint(1, 4)

        while len(sequence) < target_length:
            # Append a special symbol
            candidate = space_symbol + get_candidate()
            if len(sequence + candidate) > target_length:
                break

            sequence = sequence + candidate

            # Trim sequence if it exceeds target length
            sequence = sequence[:target_length]
            remaining_words -= 1
            if remaining_words <= 0:
                break

        if random.random() < 0.2:
            sequence = sequence.capitalize()

        if len(sequence) > 0:
            sequence = sequence.strip()
        for char in sequence:
            char_counts[char] += 1
        corpus.append(sequence)

    # Write corpus to corpus.txt
    path = "./resources/corpus/kz_corpus_generated.txt"
    with open(path, 'w', encoding='utf-8') as f:
        for sequence in corpus:
            f.write(sequence + '\n')

    print(f'Corpus generated and written to {path}')


if __name__ == "__main__":
    main()
