def merge_abab_patterns(char_array):
    result = []
    i = 0
    n = len(char_array)

    while i < n:
        if i + 3 < n and char_array[i] != char_array[i + 1]:
            if char_array[i] == char_array[i + 2] and char_array[i + 1] == char_array[i + 3]:
                result.extend([char_array[i], char_array[i + 1]])
                i += 4
                continue

        result.append(char_array[i])
        i += 1

    return result





def find_longest_sequence(characters):
    from collections import defaultdict

    char_segments = []
    total_occurrences = defaultdict(int)
    n = len(characters)
    i = 0

    while i < n:
        if characters[i] == '+':
            i += 1
            continue

        current_char = characters[i]
        length = 0

        while i < n and characters[i] == current_char:
            length += 1
            i += 1

        char_segments.append([current_char, length])
        total_occurrences[current_char] += length

    # initial combination
    combined_char_segments = []
    idx = 0
    for char, length in char_segments:
        if not (total_occurrences[char]>=3 and any(segment[0] == char and segment[1] >= 2 for segment in char_segments)):
            del char_segments[idx]
        idx = idx + 1

    # comnbine adjacent chars which are the same
    prev = None
    previdx = -1
    for i in range(len(char_segments)):
        c = char_segments[i][0]
        if c != prev:
            combined_char_segments.append([c, char_segments[i][1]])
            prev = c
            previdx = previdx + 1
        else:
            combined_char_segments[previdx][1] += char_segments[i][1]

    results = []
    for char, length in combined_char_segments:
        if length >= 7:
            results.append(char)
            results.append(char)
        else:
            results.append(char)
    results = merge_abab_patterns(results)

    if(len(results) <= 3):
        results = []
        for char, length in combined_char_segments:
            if length >= 9:
                results.append(char)
                results.append(char)
                results.append(char)
            elif length >= 7:
                results.append(char)
                results.append(char)
            else:
                results.append(char)

    return results

# Example usage
def example():
    input_1 = ['+', '+', '2', '2', '2', '2', '2', '2', 'V', 'V', 'W', 'V', '8', '8', '8', '8', 'Y', 'Y', 'Y', 'Y', 'Y', 'V', 'V', 'V', 'V', 'V', 'V', '+', '+']
    input_2 = ['+', '+', '+', 'C', 'G', 'G', 'G', 'G', 'G', 'C', 'G', 'G', 'C', 'X', 'X', 'X', 'X', 'B', 'B', 'B', 'B', 'B', 'M', 'M', 'M', 'M', '+', '+', '+']

    output_1 = find_longest_sequence(input_1)
    output_2 = find_longest_sequence(input_2)

    print(output_1)  # Expected: ['2', 'V', '8', 'Y', 'V']
    print(output_2)  # Expected: ['G', 'G', 'X', 'B', 'M']


# example()