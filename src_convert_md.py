# import os
# import re
# import subprocess
#
# # Define default replacements to be used when no replace.txt is present
# DEFAULT_REPLACEMENTS = {
#     '|ns3|': '*ns-3*',
#     '|ns2|': '*ns-2*'
# }
#
#
# def preprocess_rst_content(input_rst, include_file):
#     """
#     Preprocess the .rst file content to resolve `.. include::` directives and replace placeholders.
#     :param input_rst: The input .rst file path.
#     :param include_file: The path to the `replace.txt` file with replacements.
#     :return: Preprocessed content as a string.
#     """
#     try:
#         # Read the replacement file (replace.txt) if it exists
#         replacements = DEFAULT_REPLACEMENTS.copy()
#         if os.path.exists(include_file):
#             with open(include_file, 'r', encoding='utf-8') as file:
#                 for line in file:
#                     match = re.match(r'\.\. \|(\w+)\| replace:: (.*)', line)
#                     if match:
#                         placeholder, replacement = match.groups()
#                         replacements[f'|{placeholder}|'] = replacement.strip()
#
#         # Read the main .rst file
#         with open(input_rst, 'r', encoding='utf-8') as file:
#             content = file.read()
#
#         # Skip inline references like [turkmani]_ or [cost231]_
#         # These are treated as references and should not be replaced
#         content = re.sub(r'\[(.*?)\]_', r'\[\1\]_', content)
#
#         # Replace placeholders like |ns3| with their values from replace.txt or default replacements
#         for placeholder, replacement in replacements.items():
#             content = content.replace(placeholder, replacement)
#
#         # Resolve `.. include:: replace.txt` directives (remove them since we're handling manually)
#         content = re.sub(r'\.\. include::\s*(\S+)', '', content)
#
#         return content
#
#     except Exception as e:
#         print(f"An error occurred during preprocessing: {e}")
#         return None
#
#
# def rst_to_md_with_pandoc(preprocessed_content, output_md):
#     """
#     Convert the preprocessed .rst content to .md using Pandoc.
#     :param preprocessed_content: Preprocessed .rst content as a string.
#     :param output_md: The output .md file path.
#     """
#     try:
#         # Run pandoc to convert .rst content to .md
#         process = subprocess.Popen(['pandoc', '-f', 'rst', '-t', 'gfm', '-o', output_md], stdin=subprocess.PIPE)
#         process.communicate(input=preprocessed_content.encode('utf-8'))
#         print(f"Conversion successful -> {output_md}")
#     except subprocess.CalledProcessError as e:
#         print(f"Conversion failed: {e}")
#     except FileNotFoundError:
#         print("Please make sure Pandoc is installed and in your PATH.")
#
#
# def copy_file_content(input_file, output_file, language):
#     """
#     Copy the content of a file and wrap it in a code block with the specified language for .md format.
#     :param input_file: The input file path.
#     :param output_file: The output .md file path.
#     :param language: The programming language to be added to the code block.
#     """
#     try:
#         with open(input_file, 'r', encoding='utf-8') as file:
#             content = file.read()
#
#         with open(output_file, 'w', encoding='utf-8') as file:
#             # Add the language to the code block
#             file.write(f"```{language}\n{content}\n```")
#         print(f"Copied {input_file} -> {output_file}")
#     except Exception as e:
#         print(f"An error occurred copying {input_file}: {e}")
#
#
# def process_all_files(src_dir, dest_dir):
#     """
#     Process all files in the src_dir, convert .rst files to .md using Pandoc,
#     and copy other files (e.g., .cc, .h, .py, CMakeLists.txt) to .md files in dest_dir.
#     :param src_dir: Source directory containing files.
#     :param dest_dir: Destination directory for .md files.
#     """
#     # Ensure the destination directory exists
#     os.makedirs(dest_dir, exist_ok=True)
#
#     # Walk through all files in the source directory
#     for root, dirs, files in os.walk(src_dir):
#         for file in files:
#             file_path = os.path.join(root, file)
#             relative_path = os.path.relpath(file_path, src_dir)
#
#             # Process .rst files using Pandoc and replace .rst extension with .md
#             if file.endswith(".rst"):
#                 md_file_path = os.path.join(dest_dir, relative_path.replace(".rst", ".md"))
#             else:
#                 # For non-rst files, just append ".md"
#                 md_file_path = os.path.join(dest_dir, relative_path + ".md")
#
#             # Ensure output directories exist
#             os.makedirs(os.path.dirname(md_file_path), exist_ok=True)
#
#             # Process .rst files using Pandoc
#             if file.endswith(".rst"):
#                 # Check if a replace.txt exists in the same directory as the .rst file
#                 replace_txt = os.path.join(root, "replace.txt")
#                 preprocessed_content = preprocess_rst_content(file_path, replace_txt)
#
#                 if preprocessed_content:
#                     rst_to_md_with_pandoc(preprocessed_content, md_file_path)
#
#             # For .cc, .h, .py, CMakeLists.txt, copy content with appropriate language in code block
#             elif file.endswith('.cc') or file.endswith('.h'):
#                 copy_file_content(file_path, md_file_path, 'cpp')  # Use 'cpp' for C++ files
#             elif file.endswith('.py'):
#                 copy_file_content(file_path, md_file_path, 'python')  # Use 'python' for Python files
#             elif file == 'CMakeLists.txt':
#                 copy_file_content(file_path, md_file_path, '')  # No specific language for CMakeLists.txt
#
#
# # Example usage:
#
# src_directory = r'E:\python\ns-3-rag\ns-3-dev-git\src'  # Replace with your source directory containing all files
# dest_directory = r'E:\python\ns-3-rag\ns-3-dev-git\src_markdown'  # Correct destination directory for .md files
#
# # Process all files in the source directory and convert them to .md
# process_all_files(src_directory, dest_directory)
import os
import re
import subprocess
import shutil

DEFAULT_REPLACEMENTS = {
    '|ns3|': '*ns-3*',
    '|ns2|': '*ns-2*'
}

def preprocess_rst_content(input_rst, include_file):
    """
    Preprocess the .rst file content to resolve `.. include::` directives and replace placeholders.
    :param input_rst: The input .rst file path.
    :param include_file: The path to the `replace.txt` file with replacements.
    :return: Preprocessed content as a string.
    """
    try:
        # Read the replacement file (replace.txt) if it exists
        replacements = DEFAULT_REPLACEMENTS.copy()
        if os.path.exists(include_file):
            with open(include_file, 'r', encoding='utf-8') as file:
                for line in file:
                    match = re.match(r'\.\. \|(\w+)\| replace:: (.*)', line)
                    if match:
                        placeholder, replacement = match.groups()
                        replacements[f'|{placeholder}|'] = replacement.strip()

        # Read the main .rst file
        with open(input_rst, 'r', encoding='utf-8') as file:
            content = file.read()

        # Skip inline references like [turkmani]_ or [cost231]_
        # These are treated as references and should not be replaced
        content = re.sub(r'\[(.*?)\]_', r'\[\1\]_', content)

        # Replace placeholders like |ns3| with their values from replace.txt or default replacements
        for placeholder, replacement in replacements.items():
            content = content.replace(placeholder, replacement)

        # Resolve `.. include:: replace.txt` directives (remove them since we're handling manually)
        content = re.sub(r'\.\. include::\s*(\S+)', '', content)

        return content

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None


def rst_to_md_with_pandoc(preprocessed_content, output_md):
    """
    Convert the preprocessed .rst content to .md using Pandoc.
    :param preprocessed_content: Preprocessed .rst content as a string.
    :param output_md: The output .md file path.
    """
    try:
        # Run pandoc to convert .rst content to .md
        process = subprocess.Popen(['pandoc', '-f', 'rst', '-t', 'gfm', '-o', output_md], stdin=subprocess.PIPE)
        process.communicate(input=preprocessed_content.encode('utf-8'))
        print(f"Conversion successful -> {output_md}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
    except FileNotFoundError:
        print("Please make sure Pandoc is installed and in your PATH.")

def process_all_rst_files(src_dir, dest_dir):
    """
    处理 src_dir 下的所有 .rst 文件，转换为 .md，并保存在 dest_dir 中。
    其他文件和目录将按原样复制，保持结构不变。
    :param src_dir: 包含 .rst 文件的源目录。
    :param dest_dir: 用于保存 .md 文件的目标目录。
    """
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 遍历源目录的所有文件和子目录
    for root, dirs, files in os.walk(src_dir):
        # 计算相对于源目录的路径
        rel_path = os.path.relpath(root, src_dir)
        dest_root = os.path.join(dest_dir, rel_path)
        os.makedirs(dest_root, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)

            if file.endswith(".rst"):
                # 将 .rst 文件转换为 .md
                dest_file_md = os.path.splitext(dest_file)[0] + ".md"

                # 检查是否存在与 .rst 文件同目录的 replace.txt
                replace_txt = os.path.join(root, "replace.txt")

                # 预处理 .rst 内容
                preprocessed_content = preprocess_rst_content(src_file, replace_txt)

                if preprocessed_content:
                    # 使用 Pandoc 将预处理后的内容转换为 .md
                    rst_to_md_with_pandoc(preprocessed_content, dest_file_md)
            else:
                # 直接复制其他文件
                shutil.copy2(src_file, dest_file)
                print(f"已复制：{src_file} -> {dest_file}")

    # 复制空目录
    for root, dirs, files in os.walk(src_dir):
        for dir in dirs:
            src_dir_path = os.path.join(root, dir)
            rel_dir_path = os.path.relpath(src_dir_path, src_dir)
            dest_dir_path = os.path.join(dest_dir, rel_dir_path)
            if not os.path.exists(dest_dir_path):
                os.makedirs(dest_dir_path)

# 示例用法：

src_directory = r'E:\python\ns-3-rag\ns-3-dev-git\src'  # 请替换为您的源目录
dest_directory = r'E:\python\ns-3-rag\ns-3-dev-git\src_markdown'  # 请替换为您的目标目录

# 处理所有 .rst 文件并转换为 .md，其他文件直接复制
process_all_rst_files(src_directory, dest_directory)