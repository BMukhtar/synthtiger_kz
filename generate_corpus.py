import random
import math
from datetime import datetime, timedelta
import string
from typing import Dict

# Define the symbols
afterword_symbols = "!?.,:;"
numbers = "0123456789"
other_symbols = string.punctuation + "«»…£€¥№°—"
space_symbol = ' '
kazakh_letters = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЁабвгдежзийклмнопрстуфхцчшщъыьэюяёӘҒҚҢӨҰҮІҺәғқңөұүіһ'
english_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
all_letters = kazakh_letters + english_letters
all_symbols = numbers + other_symbols + space_symbol + all_letters
assert len(all_symbols) == len(set(all_symbols))
print(all_symbols)

# Define maximum and minimum sequence length
max_n = 40
word_count = 1_000_000

# Initialize character counts
char_counts = {char: 1 for char in all_symbols}

# Read words from kk_dict.txt
with open('resources/corpus/kk_dict.txt', 'r', encoding='utf-8') as f:
    words = f.read().splitlines()

# Read words from kk_dict.txt
with open('resources/corpus/mjsynth.txt', 'r', encoding='utf-8') as f:
    english_words = f.read().splitlines()


# Read words from kk_dict.txt
with open('resources/corpus/russian.txt', 'r', encoding='utf-8') as f:
    russian_words = f.read().splitlines()

# Filter out words that use different symbols
words = [word for word in words if all(char in char_counts for char in word)]

# Filter out words that use different symbols
english_words = [word for word in english_words if all(char in char_counts for char in word)]

# Filter out words that use different symbols
russian_words = [word for word in russian_words if all(char in char_counts for char in word)]

all_words = words + english_words + russian_words

def main():
    # Generate corpus
    corpus = []
    for i in range(word_count):  # Adjust the range as needed
        sequence = ''
        target_length = max_n
        # give more priority to less words
        remaining_words = random.choice([1, 1, 1, 1, 1, 2])

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

def build_inverted_index(words, all_letters):
    # Initialize the inverted index
    inverted_index = {char: [] for char in all_letters if char == char.lower()}

    # Populate the inverted index
    for word in all_words:
        word = word.lower()
        for char in set(word):  # Use set to avoid duplicate entries for the same word
            if char in inverted_index:
                inverted_index[char].append(word)

    # remove empty entries
    inverted_index = {char: words for char, words in inverted_index.items() if len(words) > 0}
    return inverted_index

# Build the inverted index
inverted_index = build_inverted_index(words, all_letters)

def get_banking_related_number():
    """Generate a random banking-related number with various banking formats."""
    
    currencies = ["$", "€", "₸", "₽", "£", "¥"]  # Dollar, Euro, Tenge, Ruble, Pound, Yen
    currency_names = ["USD", "EUR", "KZT", "RUB", "GBP", "JPY"]

    banking_formats = [
        # Card number format (XXXX XXXX XXXX XXXX)
        lambda: f"{get_scaled_number(1000, 9999)} {get_scaled_number(1000, 9999)} {get_scaled_number(1000, 9999)} {get_scaled_number(1000, 9999)}",
        
        # IBAN format (IBAN XXXX XXXX XXXX XXXX)
        lambda: f"IBAN {get_scaled_number(1000, 9999)} {get_scaled_number(1000, 9999)} {get_scaled_number(1000, 9999)} {get_scaled_number(1000, 9999)}",
        
        # Account number format (8 to 12 digits)
        lambda: f"{get_scaled_number(10000000, 999999999999)}",
        
        # Currency formats with different symbols (e.g., $1,234.56, €567.89)
        lambda: f"{random.choice(currencies)}{get_scaled_number(100, 1000000):,}.{random.randint(0, 99):02d}",  # Currency with symbol
        lambda: f"{get_scaled_number(100, 1000000):,} {random.choice(currency_names)}",  # Currency with name
        
        # Percentage format (interest rates or other percentages)
        lambda: f"{random.randint(1, 100)}%",
        
        # Loan or balance amounts (e.g., $10,000.00 or €500,000.00)
        lambda: f"{random.choice(currencies)}{get_scaled_number(1000, 1000000):,}.{random.randint(0, 99):02d}",
        
        # Transaction or reference ID formats (XXX-XXXX-XXX-XXXX)
        lambda: f"{random.randint(100, 999)}-{random.randint(1000, 9999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        
        # SWIFT/BIC code format (8 to 11 alphanumeric characters)
        lambda: ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.choice([8, 11]))),
        
        # Routing numbers or similar identifiers (6 to 9 digits)
        lambda: f"{random.randint(100000, 999999999)}",
        
        # Transaction amount in various currencies and formats
        lambda: f"{get_scaled_number(1000, 100000)} {random.choice(currency_names)}",
        
        # Date formats relevant to banking (e.g., YYYY-MM-DD, DD/MM/YYYY)
        lambda: handle_date(),
        lambda: (datetime.now() - timedelta(days=random.randint(0, 3650))).strftime('%d/%m/%Y'),  # European date format (last 10 years)
        lambda: (datetime.now() - timedelta(days=random.randint(0, 3650))).strftime('%m-%d-%Y'),  # US date format (last 10 years)
        
        # Time format (e.g., 12:34)
        lambda: handle_time()
    ]
    
    return random.choice(banking_formats)()


def handle_english_word():
    word = random.choice(english_words)
    n = random.random()
    if n < 0.1:
        return word.capitalize()
    elif n < 0.2:
        return word.upper()
    return word

def get_least_frequent_chars(char_counts, all_letters, threshold=10):

    # unite upper and lower case counts
    united_counts = {}
    for char, count in char_counts.items():
        if char.lower() in united_counts:
            united_counts[char.lower()] += count
        else:
            united_counts[char.lower()] = count

    # get sorted list of chars by count
    sorted_chars = sorted(united_counts.items(), key=lambda x: x[1])

    return [char for char, count in sorted_chars if char in all_letters and char in inverted_index][:threshold]


def handle_less_frequent_letters():
    # choose letters type: english or kazakh with 0.5 probability
    if random.random() < 0.5:
        sample = english_letters
    else:
        sample = kazakh_letters
    least_frequent_chars = get_least_frequent_chars(char_counts, sample)
    # Randomly select a less frequent character
    char = random.choice(least_frequent_chars)
    
    # Randomly select a word from the list of words containing this character
    return random.choice(inverted_index[char])

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
    return (space_symbol + get_scaled_number()
            + random.choice(["+", "-", '/', '*'])
            + get_scaled_number()
            + random.choice(["=", ">", "<"])
            + get_scaled_number())


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

def get_random_word():
    word = random.choice(all_words)
    n = random.random()
    if n < 0.1:
        return word.capitalize()
    elif n < 0.2:
        return word.upper()
    return word

def handle_random_symbol():
    filtered = other_symbols.replace("\\", "")
    return random.choice([random.choice(words), "", "", ""]) + random.choice(filtered)

def handle_other():
    n = random.randint(1, 10)
    if n == 1:
        return handle_hashtag()
    if n == 2:
        return random.choice(["(", ""]) + get_random_word() + random.choice([")", ""])
    if n == 3:
        return "«" + get_random_word() + "»"
    if n == 4:
        tik = random.choice(["'", "''"])
        return tik + get_random_word() + tik
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
    if random.random() < 0.8:
        return get_random_word()

    choice = random.random()
    # Append a special symbol
    if choice < 0.3:
        return random.choice(words) + random.choice(afterword_symbols)
    # Append a number
    elif choice < 0.4:
        number = get_random_number()
        return number
    elif choice < 0.45:
        return handle_math_expression()
    elif choice < 0.6:
        return handle_less_frequent_letters()
    elif choice < 0.7:
        return handle_random_symbol()
    # Append other symbols
    else:
        return handle_other()


if __name__ == "__main__":
    main()
