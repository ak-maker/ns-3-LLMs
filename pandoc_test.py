import re
import subprocess

def preprocess_rst(input_rst, output_rst, include_file):
    """
    Preprocess the .rst file to resolve `.. include::` directives and replace placeholders.
    :param input_rst: The input .rst file path.
    :param output_rst: The output preprocessed .rst file path.
    :param include_file: The path to the `replace.txt` file with replacements.
    """
    try:
        # Read the replacement file (replace.txt)
        replacements = {}
        with open(include_file, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.match(r'\.\. \|(\w+)\| replace:: (.*)', line)
                if match:
                    placeholder, replacement = match.groups()
                    replacements[f'|{placeholder}|'] = replacement.strip()

        # Read the main .rst file
        with open(input_rst, 'r', encoding='utf-8') as file:
            content = file.read()

        # Replace placeholders like |ns3| with their values from replace.txt
        for placeholder, replacement in replacements.items():
            content = content.replace(placeholder, replacement)

        # Resolve `.. include:: replace.txt`
        # Since replace.txt is already processed, we don't need to inline its content here.
        content = re.sub(r'\.\. include::\s*(\S+)', '', content)

        # Write the preprocessed content to a new file
        with open(output_rst, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"Preprocessing completed: {input_rst} -> {output_rst}")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")

def rst_to_md_with_pandoc(input_rst, output_md):
    """
    Convert the preprocessed .rst file to .md using Pandoc.
    :param input_rst: The input preprocessed .rst file path.
    :param output_md: The output .md file path.
    """
    try:
        # Run pandoc to convert .rst to .md
        subprocess.run(['pandoc', input_rst, '-f', 'rst', '-t', 'gfm', '-o', output_md], check=True)
        print(f"Conversion successful: {input_rst} -> {output_md}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
    except FileNotFoundError:
        print("Please make sure Pandoc is installed and in your PATH.")

# Example usage:

input_rst_file = 'enhancements.rst'  # Replace with your .rst file path
replace_txt_file = 'replace.txt'  # Replace with your replace.txt file path
output_preprocessed_rst_file = 'preprocessed_example.rst'
output_md_file = 'output.md'

# Step 1: Preprocess the .rst file
preprocess_rst(input_rst_file, output_preprocessed_rst_file, replace_txt_file)

# Step 2: Convert the preprocessed .rst file to .md using Pandoc
rst_to_md_with_pandoc(output_preprocessed_rst_file, output_md_file)
