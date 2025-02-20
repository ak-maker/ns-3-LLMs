import os

# 定义各个目录的路径
doc_markdown_dir = r"E:\python\ns-3-rag\ns-3-dev-git\doc_markdown"
examples_markdown_dir = r"E:\python\ns-3-rag\ns-3-dev-git\examples_markdown"
combined_src_file = r"E:\python\ns-3-rag\ns-3-dev-git\combined_src_markdown.md"
output_file = r"E:\python\ns-3-rag\merged_markdown_output.md"

# 函数用于读取目录中的所有markdown文件并合并内容
def merge_markdown_files(directory):
    merged_content = ""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    merged_content += f"\n\n# {file}\n\n"  # 添加文件名作为标题
                    merged_content += f.read()
    return merged_content

# 合并doc_markdown和examples_markdown中的内容
doc_content = merge_markdown_files(doc_markdown_dir)
examples_content = merge_markdown_files(examples_markdown_dir)

# 读取combined_src_markdown.md的内容
with open(combined_src_file, "r", encoding="utf-8") as f:
    combined_src_content = f.read()

# 合并所有内容
final_content = doc_content + "\n\n" + examples_content + "\n\n" + combined_src_content

# 将合并后的内容写入输出文件
with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_content)

print(f"Markdown内容已成功合并并保存到 {output_file}")
