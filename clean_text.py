import re

def clean_gutenberg_text(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Define the start and end markers as literal strings
    start_literal = "*** START OF THE PROJECT GUTENBERG EBOOK Pride and Prejudice ***"
    end_literal = "*** END OF THE PROJECT GUTENBERG EBOOK Pride and Prejudice ***"

    # Escape the literal strings for use in regex
    start_marker_escaped = re.escape(start_literal)
    end_marker_escaped = re.escape(end_literal)

    # Extract content between START and END markers
    match = re.search(f"{start_marker_escaped}(.*?){end_marker_escaped}", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        print("Start or end marker not found. Cleaning entire file.")
        # If markers are not found, proceed with the whole text (or handle error)

    # Remove illustration and copyright lines using string methods
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line.startswith("[Illustration:") and not stripped_line.startswith("[_Copyright"):
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Remove any remaining single underscores (used for italics)
    text = text.replace("_", "")

    # Remove multiple blank lines, leaving only single blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace from each line
    text = "\n".join([line.strip() for line in text.splitlines()])

    # Remove multiple blank lines again after stripping lines
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Strip leading/trailing whitespace from the whole text
    text = text.strip()

    # Save cleaned text
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Cleaned text saved to {output_file}")

# Example usage: 
clean_gutenberg_text("E:\\Mema-Lab\\Menari\\Menari-core\\pride_and_prejudice.txt", "E:\\Mema-Lab\\Menari\\Menari-core\\pride_and_prejudice_clean_ready.txt")
