import os

# 定义要遍历的基目录列表
base_dirs = [
    r'E:\python\ns-3-rag\ns-3-dev-git\doc',
    r'E:\python\ns-3-rag\ns-3-dev-git\examples',
    r'E:\python\ns-3-rag\ns-3-dev-git\src'
]

# 定义输出文件路径
output_file = r'E:\python\ns-3-rag\ns-3-dev-git\combined_markdown_output.md'

# 定义要忽略的文件扩展名集合
ignore_extensions = {'.dia', '.png', '.pcap', '.pdf', '.eps', '.dot', '.seqdiag',
                     '.fad', '.gp', '.svg', '.jpg', '.jpeg', '.gif', '.bmp', '.ico'}

# 定义文件扩展名与编程语言的映射关系
extension_language_map = {
    '.py': 'python',
    '.cc': 'cpp',
    '.cpp': 'cpp',
    '.cxx': 'cpp',
    '.h': 'cpp',
    '.hpp': 'cpp',
    '.sh': 'bash',
    '.gnuplot': 'gnuplot',
    '.txt': '',
    '.conf': '',
    '.m': 'matlab',
    '.click': '',
    '.pl': 'perl',
    '.ns_movements': '',
    '.ns_params': '',
    '.params': '',
    '.reflog': '',
    '.plt': 'gnuplot',
    '.xml': 'xml',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
}

# 定义项目根目录
project_root = r'E:\python\ns-3-rag\ns-3-dev-git'

# 打开输出文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    # 遍历每个基目录
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            for filename in files:
                # 忽略名为 replace.txt 的文件
                if filename == 'replace.txt':
                    continue

                filepath = os.path.join(root, filename)
                extension = os.path.splitext(filename)[1]
                # 如果文件扩展名在忽略列表中，跳过
                if extension.lower() in ignore_extensions:
                    continue
                try:
                    # 读取文件内容
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as infile:
                        content = infile.read()
                except Exception as e:
                    print(f"无法读取文件 {filepath}: {e}")
                    continue

                # 如果文件内容为空，跳过该文件
                if not content.strip():
                    print(f"文件为空，跳过：{filepath}")
                    continue

                # 获取相对于项目根目录的相对路径
                relative_path = os.path.relpath(filepath, project_root)
                relative_path = relative_path.replace('\\', '/')

                # 构建新的标题
                dir_path = os.path.dirname(relative_path)
                file_extension = os.path.splitext(filename)[1]  # 包含点，如 '.md'、'.py'

                if extension == '.md':
                    # 对于 Markdown 文件，尝试从文件内容中提取第一个一级标题
                    lines = content.splitlines()
                    heading_found = False
                    heading_text = ''
                    for line in lines:
                        stripped_line = line.strip()
                        if stripped_line.startswith('# '):  # 找到第一个以 '# ' 开头的标题
                            heading_text = stripped_line[2:].strip()
                            heading_found = True
                            break
                    if not heading_found:
                        # 如果没有找到一级标题，使用文件名（包括扩展名）
                        heading_text = filename
                    # 构建新标题，包含扩展名
                    new_header = f"# /{dir_path}/{heading_text}"
                    new_header = new_header.replace('\\', '/')

                    # 替换文件中的起始标题
                    header_changed = False
                    new_lines = []
                    for line in lines:
                        if not header_changed and line.strip().startswith('#'):
                            new_lines.append(new_header)
                            header_changed = True
                        else:
                            new_lines.append(line)
                    if not header_changed:
                        new_lines.insert(0, new_header)
                    content = '\n'.join(new_lines)
                else:
                    # 非 Markdown 文件，标题中使用文件名（包括扩展名）
                    file_name_with_ext = filename
                    new_header = f"# /{dir_path}/{file_name_with_ext}"
                    new_header = new_header.replace('\\', '/')

                    # 删除结尾的空行
                    content_lines = content.splitlines()
                    while content_lines and not content_lines[-1].strip():
                        content_lines.pop()
                    content = '\n'.join(content_lines)

                    # 获取语言标识
                    language = extension_language_map.get(extension.lower(), '')

                    # 包裹代码块并添加标题
                    content = f"{new_header}\n\n```{language}\n{content}\n```\n"

                # 将内容写入输出文件
                outfile.write(content)
                outfile.write('\n\n')  # 文件间添加空行
