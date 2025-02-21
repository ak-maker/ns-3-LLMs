import json
import tiktoken
import re


def calculate_token_size(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def extract_code_block(content):
    pattern = r'```(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    return matches[0] if matches else None


def should_exclude_comment(comment):
    lower_comment = comment.lower()
    exclude_patterns = [
        'copyright',
        'author',
        'license',
        'spidx',
        '#include',
        'original author'
    ]
    return any(pattern in lower_comment for pattern in exclude_patterns)


def process_rocket_fuel(content):
    seen_cities = set()
    kept_lines = []
    lines = content.split('\n')

    for line in lines:
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            loc1_full = parts[0]
            city1 = loc1_full.split(',')[0] if ',' in loc1_full else loc1_full
            loc2_full = parts[1]
            city2 = loc2_full.split(',')[0] if ',' in loc2_full else loc2_full

            if city1 not in seen_cities:
                seen_cities.add(city1)
                kept_lines.append(line)
                continue

            if city2 not in seen_cities:
                seen_cities.add(city2)
                kept_lines.append(line)

        except Exception as e:
            print(f"Warning: Error processing line: {line}")
            continue

    return kept_lines


def process_special_files(content, filename):
    if 'RocketFuel_toposample' in filename:
        return process_rocket_fuel(content)

    if filename in [
        '/src/mobility/examples/default.ns_movements',
        '/src/topology-read/examples/Inet_toposample.txt',
        '/src/topology-read/examples/Orbis_toposample.txt'
    ]:
        lines = content.split('\n')
        return lines[:20]

    return content.split('\n')


def create_reduced_content(filename, orig_line_num, content_lines):
    content = '\n'.join(content_lines)
    reduced_content = f"# {filename}\n\nOut of embedding size, original line number in chunk.jsonl: {orig_line_num}\n```\n{content}\n```"
    if calculate_token_size(reduced_content) > 8000:
        while content_lines and calculate_token_size(reduced_content) > 8000:
            content_lines.pop()
            content = '\n'.join(content_lines)
            reduced_content = f"# {filename}\n\nOut of embedding size, original line number in chunk.jsonl: {orig_line_num}\n```\n{content}\n```"
    return reduced_content


def truncate_comments_by_token_size(comments, filename, orig_line_num):
    output_truncated_path = "./filter_commentsoutofsize.txt"

    if not comments:
        return comments

    reduced_content = create_reduced_content(filename, orig_line_num, comments)
    kept_comments = comments[:len(reduced_content.split('\n')) - 4]  

    if len(kept_comments) < len(comments):
        with open(output_truncated_path, 'a', encoding='utf-8') as f:
            truncated_count = len(comments) - len(kept_comments)
            f.write(f"filename: {filename}\n")
            f.write(
                f"original lines of comments: {len(comments)}, reserved lines: {len(kept_comments)}, truncated lines: {truncated_count}\n")
            f.write("The truncated lines of comments: \n")
            for comment in comments[len(kept_comments):]:
                f.write(f"{comment}\n")
            f.write("\n")

    return kept_comments

def extract_comments_for_display(code_block):
    if code_block is None:
        return []

    comments = []
    lines = code_block.split('\n')
    i = 0
    in_multiline_comment = False
    current_multiline_comment = []
    in_docstring = False
    current_docstring = []

    while i < len(lines):
        original_line = lines[i]
        line = original_line.strip()

        if line.startswith('#include'):
            i += 1
            continue

        if '#' in line and not line.startswith('#include'):
            comment_start = line.index('#')
            before_hash = line[:comment_start]
            if not (before_hash.count('"') % 2 == 1 or before_hash.count("'") % 2 == 1):
                if not should_exclude_comment(line):
                    comments.append(original_line)
            i += 1
            continue

        if (line.startswith('"""') or line.startswith("'''")) and not in_docstring:
            in_docstring = True
            current_docstring = [original_line]
            if line.count('"""') == 2 or line.count("'''") == 2:
                if not should_exclude_comment(line):
                    comments.append(original_line)
                in_docstring = False
            elif line.endswith('"""') or line.endswith("'''"):
                if not should_exclude_comment(line):
                    comments.append(original_line)
                in_docstring = False
            i += 1
            continue
            
        if in_docstring:
            current_docstring.append(original_line)
            if line.endswith('"""') or line.endswith("'''"):
                complete_docstring = '\n'.join(current_docstring)
                if not should_exclude_comment(complete_docstring):
                    comments.extend(current_docstring)
                current_docstring = []
                in_docstring = False
            i += 1
            continue

        if '//' in line and not line.startswith('#include'):
            if not should_exclude_comment(line):
                comments.append(original_line)
            i += 1
            continue

        if '/*' in line and not in_multiline_comment:
            in_multiline_comment = True
            current_multiline_comment = [original_line]
            if '*/' in line:  # 单行的完整注释
                if not should_exclude_comment(line):
                    comments.append(original_line)
                current_multiline_comment = []
                in_multiline_comment = False
            i += 1
            continue

        if in_multiline_comment:
            current_multiline_comment.append(original_line)
            if '*/' in line:
                complete_comment = '\n'.join(current_multiline_comment)
                if not should_exclude_comment(complete_comment):
                    comments.extend(current_multiline_comment)
                current_multiline_comment = []
                in_multiline_comment = False
            i += 1
            continue

        i += 1

    return comments


def count_comment_lines(code_block):
    if code_block is None:
        return 0

    count = 0
    lines = code_block.split('\n')
    i = 0
    in_multiline_comment = False
    in_docstring = False

    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith('#include'):
            i += 1
            continue

        if '#' in line and not line.startswith('#include'):
            comment_start = line.index('#')
            before_hash = line[:comment_start]
            if not (before_hash.count('"') % 2 == 1 or before_hash.count("'") % 2 == 1):
                if not should_exclude_comment(line):
                    count += 1
            i += 1
            continue

        if (line.startswith('"""') or line.startswith("'''")) and not in_docstring:
            in_docstring = True
            if not should_exclude_comment(line):
                count += 1
            if line.count('"""') == 2 or line.count("'''") == 2:
                in_docstring = False
            elif line.endswith('"""') or line.endswith("'''"):
                in_docstring = False
            i += 1
            continue

        if in_docstring:
            if not should_exclude_comment(line):
                count += 1
            if line.endswith('"""') or line.endswith("'''"):
                in_docstring = False
            i += 1
            continue

        if '//' in line and not line.startswith('#include'):
            if not should_exclude_comment(line):
                count += 1
            i += 1
            continue

        if '/*' in line and not in_multiline_comment:
            in_multiline_comment = True
            if not should_exclude_comment(line):
                count += 1
            if '*/' in line: 
                in_multiline_comment = False
            i += 1
            continue

        if in_multiline_comment:
            if '*/' in line:
                in_multiline_comment = False
                if line.strip() == '*/':  
                    i += 1
                    continue
            if not should_exclude_comment(line):
                count += 1
            i += 1
            continue

        i += 1

    return count


def extract_filename(content):
    first_line = content.split('\n')[0]
    if first_line.startswith('# /'):
        return first_line[2:].strip()
    return first_line.strip()


def process_jsonl():
    input_file_path = r"E:\python\ns-3-rag\data\chunk.jsonl"
    output_token_count_path = r"E:\python\ns-3-rag\data\filter_token_count_result.txt"
    output_filtered_path = r"E:\python\ns-3-rag\data\filtered_chunk.jsonl"
    output_less_than_10_path = r"./filter_comments_lessthan10.txt"

    max_token_size = 8000
    over_limit_lines = []
    less_than_10_comments = []

    original_lines = []
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for idx, line in enumerate(infile, 1):
            original_lines.append((idx, line))

    open(output_filtered_path, 'w', encoding='utf-8').close()

    for orig_line_num, line in original_lines:
        try:
            content = json.loads(line)
            filename = extract_filename(content)
            token_count = calculate_token_size(content)

            if token_count > max_token_size:
                code_block = extract_code_block(content)

                if code_block is not None:
                    if any(name in filename for name in [
                        'default.ns_movements',
                        'Inet_toposample.txt',
                        'Orbis_toposample.txt',
                        'RocketFuel_toposample'
                    ]):
                        processed_lines = process_special_files(code_block, filename)
                        reduced_content = create_reduced_content(filename, orig_line_num, processed_lines)
                        with open(output_filtered_path, 'a', encoding='utf-8') as filtered_outfile:
                            filtered_outfile.write(json.dumps(reduced_content) + '\n')
                        continue

                    comment_count = count_comment_lines(code_block)

                    if comment_count >= 10:
                        comments = extract_comments_for_display(code_block)
                        truncated_comments = truncate_comments_by_token_size(comments, filename, orig_line_num)
                        reduced_content = create_reduced_content(filename, orig_line_num, truncated_comments)
                        with open(output_filtered_path, 'a', encoding='utf-8') as filtered_outfile:
                            filtered_outfile.write(json.dumps(reduced_content) + '\n')

                        over_limit_lines.append({
                            "filename": filename,
                            "line_number": orig_line_num,
                            "type": "comments>=10",
                            "comment_count": comment_count
                        })
                    else:
                        truncated_content = create_reduced_content(filename, orig_line_num, content.split("\n"))
                        with open(output_filtered_path, 'a', encoding='utf-8') as filtered_outfile:
                            filtered_outfile.write(json.dumps(truncated_content) + '\n')
                        less_than_10_comments.append({
                            "filename": filename,
                            "line_number": orig_line_num,
                            "comment_count": comment_count
                        })
                else:
                    over_limit_lines.append({
                        "filename": filename,
                        "line_number": orig_line_num,
                        "type": "non-code-block"
                    })
            else:
                with open(output_filtered_path, 'a', encoding='utf-8') as filtered_outfile:
                    filtered_outfile.write(line)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {orig_line_num}: {str(e)}")
            continue

    with open(output_token_count_path, 'w', encoding='utf-8') as f:
        f.write("The following is information about files exceeding 8000 tokens:\n\n")
        for item in over_limit_lines:
            if "type" in item and item["type"] == "comments>=10":
                f.write(
                    f"Filename: {item['filename']}, Original Line Number: {item['line_number']}, Comment Count: {item['comment_count']} (Comment Count >= 10)\n")
            else:
                f.write(f"Filename: {item['filename']}, Original Line Number: {item['line_number']} (Not a code block or other case)\n")
        f.write(f"\nTotal number of lines exceeding 8000 tokens: {len(over_limit_lines)}\n")

    with open(output_less_than_10_path, 'w', encoding='utf-8') as f:
        f.write("The following files have fewer than 10 comments:\n\n")
        for item in less_than_10_comments:
            f.write(f"Filename: {item['filename']}, Original Line Number: {item['line_number']}, Comment Count: {item['comment_count']}\n")
        f.write(f"\nTotal number of files with fewer than 10 comments: {len(less_than_10_comments)}\n")


    print(f"""Processing complete:
        - Total exceeding token limit: {len(over_limit_lines)}
        - Number of files with fewer than 10 comments: {len(less_than_10_comments)}
        Results have been written to the corresponding file""")


if __name__ == "__main__":
    process_jsonl()
