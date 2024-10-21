import json


def convert_word_to_char_bio(entry, label_map):
    sentence = entry['sentence']
    labels = entry['labels']
    char_labels = ['O'] * len(sentence)

    char_index = 0
    for label in labels:
        word = label['word']
        tag = label['label']

        word_len = len(word)
        if tag.startswith('B-'):
            for i in range(word_len):
                if i == 0:
                    char_labels[char_index + i] = label_map['B-' + tag[2:]]
                else:
                    char_labels[char_index + i] = label_map['I-' + tag[2:]]
        else:
            for i in range(word_len):
                char_labels[char_index + i] = label_map[tag]

        char_index += word_len

    return {'sentence': sentence, 'labels': char_labels}


# Load data from JSON file
with open('tagged_sentences.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create label map
label_map = {}
label_count = 0

# Add "O" label to the map
label_map['O'] = 0

for entry_list in data:
    for entry in entry_list:
        labels = entry['labels']
        for label in labels:
            tag = label['label']
            if tag.startswith('B-') and 'B-' + tag[2:] not in label_map:
                label_count += 1
                label_map['B-' + tag[2:]] = label_count
                label_count += 1
                label_map['I-' + tag[2:]] = label_count
            elif tag not in label_map:
                label_count += 1
                label_map[tag] = label_count

# Print label map
print("Label Map:")
for tag, label_id in label_map.items():
    print(f"{tag}: {label_id}")

# Convert data
converted_data = []
for entry_list in data:
    converted_entry_list = [convert_word_to_char_bio(entry, label_map) for entry in entry_list]
    converted_data.append(converted_entry_list)

# Remove trailing 'O' labels
for entry_list in converted_data:
    for entry in entry_list:
        labels = entry['labels']
        # Reverse iterate to find the last non-'O' label
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] != 0:  # 'O' label ID is 0
                entry['labels'] = labels[:i + 1]  # Keep up to the last non-'O' label
                break

# Clean up 'O' labels from the end of each sentence
for entry_list in converted_data:
    for entry in entry_list:
        labels = entry['labels']
        # Remove all 'O's
        entry['labels'] = [label_id for label_id in labels if label_id != 'O']

# Print the first entry for debugging
print("\nExample of converted data:")
print(converted_data[0])

# Save to new JSON file
with open('converted_tagged_sentences.json', 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)
# Note: Saving to JSON file is not required as per the request
