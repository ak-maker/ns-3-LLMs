# import os
#
# def convert_file_to_md(src_file, dest_file):
#     """
#     Convert the given source file (.py, .cc, .sh, CMakeLists.txt) to markdown without any extra content or headers.
#     :param src_file: The source file path.
#     :param dest_file: The destination markdown file path.
#     """
#     try:
#         with open(src_file, 'r', encoding='utf-8') as file:
#             content = file.read()
#
#         # Write content to the .md file
#         with open(dest_file, 'w', encoding='utf-8') as md_file:
#             # Add a markdown code block with the appropriate language identifier
#             md_file.write("```" + get_file_extension(src_file) + "\n")
#             md_file.write(content)
#             md_file.write("\n```\n")
#
#         print(f"Converted: {src_file} -> {dest_file}")
#
#     except Exception as e:
#         print(f"Error converting {src_file}: {e}")
#
#
# def get_file_extension(file_path):
#     """
#     Return the appropriate markdown language identifier for the code block based on the file extension.
#     :param file_path: The file path to check.
#     :return: The markdown language identifier (e.g., python, cpp, sh).
#     """
#     if file_path.endswith('.py'):
#         return 'python'
#     elif file_path.endswith('.cc'):
#         return 'cpp'
#     elif file_path.endswith('.sh'):
#         return 'sh'
#     else:
#         return ''  # For CMakeLists.txt or other files
#
#
# def process_all_code_files(src_dir, dest_dir):
#     """
#     Process all .py, .cc, .sh, and CMakeLists.txt files in the src_dir, convert them to .md,
#     and save in the dest_dir with the same directory structure, without adding any headers.
#     :param src_dir: Source directory containing code files.
#     :param dest_dir: Destination directory for markdown files.
#     """
#     # Ensure the destination directory exists
#     os.makedirs(dest_dir, exist_ok=True)
#
#     # Walk through all files in the source directory
#     for root, dirs, files in os.walk(src_dir):
#         for file in files:
#             if file.endswith((".py", ".cc", ".sh", "CMakeLists.txt")):
#                 src_file = os.path.join(root, file)
#                 relative_path = os.path.relpath(src_file, src_dir)
#
#                 # Determine the output markdown file path
#                 md_file = os.path.join(dest_dir, relative_path + ".md")
#
#                 # Ensure output directories exist
#                 os.makedirs(os.path.dirname(md_file), exist_ok=True)
#
#                 # Convert the file to markdown without any directory information or headers
#                 convert_file_to_md(src_file, md_file)
#
#
# # Example usage:
#
# src_directory = r'E:\python\ns-3-rag\ns-3-dev-git\examples'  # Replace with your source directory
# dest_directory = r'E:\python\ns-3-rag\ns-3-dev-git\examples_markdown'  # Replace with your destination directory for markdown
#
# # Process all .py, .cc, .sh, and CMakeLists.txt files in the source directory and convert them to .md
# process_all_code_files(src_directory, dest_directory)
import os
import re
import subprocess
import shutil

def preprocess_rst_content(input_rst, include_file):
    """
    预处理 .rst 文件内容，解决 `.. include::` 指令并替换占位符。
    :param input_rst: 输入的 .rst 文件路径。
    :param include_file: 包含替换内容的 `replace.txt` 文件路径。
    :return: 预处理后的内容字符串。
    """
    try:
        # 读取替换规则（replace.txt）如果存在
        replacements = {}
        if os.path.exists(include_file):
            with open(include_file, 'r', encoding='utf-8') as file:
                for line in file:
                    match = re.match(r'\.\. \|(\w+)\| replace:: (.*)', line)
                    if match:
                        placeholder, replacement = match.groups()
                        replacements[f'|{placeholder}|'] = replacement.strip()

        # 读取主 .rst 文件
        with open(input_rst, 'r', encoding='utf-8') as file:
            content = file.read()

        # 替换类似 |ns3| 的占位符
        for placeholder, replacement in replacements.items():
            content = content.replace(placeholder, replacement)

        # 处理 `.. include:: replace.txt`，此处假设已经处理了替换
        content = re.sub(r'\.\. include::\s*(\S+)', '', content)

        return content

    except Exception as e:
        print(f"预处理时出错：{e}")
        return None

def rst_to_md_with_pandoc(preprocessed_content, output_md):
    """
    使用 Pandoc 将预处理后的 .rst 内容转换为 .md。
    :param preprocessed_content: 预处理后的 .rst 内容字符串。
    :param output_md: 输出的 .md 文件路径。
    """
    try:
        # 使用 pandoc 进行转换
        process = subprocess.Popen(['pandoc', '-f', 'rst', '-t', 'gfm', '-o', output_md], stdin=subprocess.PIPE)
        process.communicate(input=preprocessed_content.encode('utf-8'))
        print(f"转换成功：{output_md}")
    except subprocess.CalledProcessError as e:
        print(f"转换失败：{e}")
    except FileNotFoundError:
        print("请确保已安装 Pandoc 并已添加到系统 PATH 中。")

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

src_directory = r'E:\python\ns-3-rag\ns-3-dev-git\examples'  # 请替换为您的源目录
dest_directory = r'E:\python\ns-3-rag\ns-3-dev-git\examples_markdown'  # 请替换为您的目标目录

# 处理所有 .rst 文件并转换为 .md，其他文件直接复制
process_all_rst_files(src_directory, dest_directory)