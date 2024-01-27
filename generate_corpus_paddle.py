

# Define the symbols
afterword_symbols = "!?.,:;"
numbers = "0123456789"
other_symbols = """
    '#()<>+-/*=%$»«"
""".strip()
space_symbol = ' '
kazakh_letters = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЁабвгдежзийклмнопрстуфхцчшщъыьэюяёӘҒҚҢӨҰҮІҺәғқңөұүіһ'
english_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
all_letters = kazakh_letters + english_letters
all_symbols = numbers + afterword_symbols + other_symbols + space_symbol + all_letters
print(all_symbols)

# print each symbol in a new line on a new file
with open('resources/corpus/symbols.txt', 'w', encoding='utf-8') as f:
    for symbol in all_symbols:
        f.write(symbol + '\n')
