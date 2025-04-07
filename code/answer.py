""" an rag demo running in the terminal
"""
import warnings

warnings.simplefilter("ignore", UserWarning)
""" an rag demo running in the terminal
"""
import tiktoken
from tiktoken import encoding_for_model, get_encoding
import typer
from typing_extensions import Annotated
from litellm import completion
from config import cfg
from pathlib import Path
from vectordb import VectorDB
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from functools import partial
import json
from preprocess import log, LOGGING_HELP
from tool import load_tools
import logging
import logging
from pathlib import Path
from typing import Annotated
import typer
from rich.prompt import Prompt
import re
import subprocess
from pathlib import Path

enable_watchdog = False  # Declare a global variable
path = "E:\python\\ns-3-rag\\ns-3-dev-git"
conversation_history = []

app = typer.Typer(help=__doc__)
console = Console()


def execute_function_call(message, tools):
    to_call = message.tool_calls[0].function
    if to_call.name in tools:
        query = json.loads(to_call.arguments)["query"]
        tool = tools[to_call.name]
        results = tool._run(query)
    else:
        results = f"Error: function {message.tool_calls[0].function.name} does not exist"
    return results


import os
import subprocess
import time
import threading


def update_plt_file(plt_path, dat_absolute_path):
    """
    Updates the .plt file to replace the relative .dat file path with an absolute path.

    :param plt_path: Path to the .plt file.
    :param dat_absolute_path: Absolute path to the .dat file.
    """
    with open(plt_path, "r") as file:
        lines = file.readlines()

    # Update the line containing the .dat file path
    updated_lines = []
    for line in lines:
        if "plot" in line and ".dat" in line:
            line = f'plot "{dat_absolute_path}" index 0 title "Emitter Interarrival Time" with linespoints\n'
        elif "set output" in line and ".png" in line:
            # Ensure the output PNG file has the full path
            png_path = os.path.join(os.path.dirname(plt_path), "dynamic-emitter-plot.png")
            line = f'set output "{png_path}"\n'
        updated_lines.append(line)

    # Save the updated .plt file
    with open(plt_path, "w") as file:
        file.writelines(updated_lines)


def monitor_and_execute_gnuplot(directory, interval=5):
    """
    Monitors a directory for new .plt files, updates their .dat file paths, and executes them using gnuplot.

    :param directory: Path to the directory to monitor.
    :param interval: Time interval (in seconds) between checks.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory.")

    print(f"Monitoring directory: {directory} for new .plt files...")
    seen_files = set(os.listdir(directory))

    while True:
        current_files = set(os.listdir(directory))
        new_files = current_files - seen_files

        for file in new_files:
            if file.endswith(".plt"):
                plt_path = os.path.join(directory, file)
                dat_filename = file.replace(".plt", ".dat")
                dat_path = os.path.join(directory, dat_filename)

                # print(f"New .plt file detected: {file}. Updating and executing gnuplot...")

                # Check if the associated .dat file exists
                if not os.path.exists(dat_path):
                    print(f"Warning: Required data file '{dat_filename}' is missing. Skipping execution.")
                    continue

                # Update the .plt file with the absolute path to the .dat file
                update_plt_file(plt_path, dat_path)

                # Execute the updated .plt file
                try:
                    subprocess.run(["gnuplot", plt_path], check=True)
                    # print(f"Execution of {file} completed.")
                except subprocess.CalledProcessError as e:
                    print("")

        # Update the seen files
        seen_files = current_files
        time.sleep(interval)


def start_monitoring_gnuplot(directory, interval=5):
    """
    Starts the gnuplot monitoring function in a separate thread.

    :param directory: Directory to monitor.
    :param interval: Time interval between checks.
    """
    monitor_thread = threading.Thread(target=monitor_and_execute_gnuplot, args=(directory, interval), daemon=True)
    monitor_thread.start()
    print("Directory monitoring for .plt files started in the background.")


def save_conversation_history_and_calculate_tokens(history, output_file):
    # Save the conversation history to a text file
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in history:
            f.write(f"{entry['role']}: {entry['content']}\n")

    # Concatenate all conversation history for token calculation
    full_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in history])

    # Get the tokenizer for the model
    tokenizer = get_encoding("cl100k_base")

    # Calculate the token count
    token_count = len(tokenizer.encode(full_text))

    # Append the token count to the file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\nToken Count: {token_count}\n")

    # print(f"Conversation history saved to {output_file}. Token count: {token_count}")


@app.command("demo")
def demo(
        docs_jsonl_sionna: Annotated[Path, typer.Argument(help="A JSONL file for Sionna documents")],
        embed_jsonl_sionna: Annotated[Path, typer.Argument(help="A JSONL file for Sionna embeddings")],
        docs_jsonl_ns3: Annotated[Path, typer.Argument(help="A JSONL file for NS-3 documents")],
        embed_jsonl_ns3: Annotated[Path, typer.Argument(help="A JSONL file for NS-3 embeddings")],
        vectordb_sionna: Annotated[str, typer.Option(help="Name of the Sionna database")] = "sionna_db",
        vectordb_ns3: Annotated[str, typer.Option(help="Name of the NS-3 database")] = "ns-3_db",
        rerank: Annotated[bool, typer.Option(help="whether or not rerank the retrieved contexts")] = False,
        rebuild: Annotated[
            bool, typer.Option(help="if true, rebuild the database from docs_jsonl and embed_jsonl")] = False,
        top_k: Annotated[int, typer.Option(help="number of contexts to retrieve")] = cfg.get("top_k", 1),
        llm: Annotated[str, typer.Option(help="which LLM to use")] = cfg.get("llm"),
        logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
        use_watchdog: Annotated[bool, typer.Option(help="Enable Watchdog automation for script execution")] = False,
):
    global enable_watchdog
    enable_watchdog = Prompt.ask(
        "Do you want to enable automation for script processing? ([bold yellow]yes/no[/])",
        default="no"
    ).lower() == 'yes'

    if enable_watchdog:
        start_monitoring_gnuplot(path)
        log.info("Automation is enabled.")
    else:
        log.info("Automation is disabled.")

    cfg['llm'] = llm
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")

    # create database
    sionna_db = VectorDB(vectordb_sionna, rerank=rerank)
    ns3_db = VectorDB(vectordb_ns3, rerank=rerank)

    if rebuild:
        log.info("Rebuilding databases...")
        assert docs_jsonl_sionna.exists(), f"Input docs_jsonl ({docs_jsonl_sionna}) doesn't exist."
        assert embed_jsonl_sionna.exists(), f"Input embed_jsonl ({embed_jsonl_sionna}) doesn't exist."
        assert docs_jsonl_ns3.exists(), f"Input docs_jsonl ({docs_jsonl_sionna}) doesn't exist."
        assert embed_jsonl_ns3.exists(), f"Input embed_jsonl ({embed_jsonl_sionna}) doesn't exist."
        sionna_db.rebuild(docs_jsonl_sionna, embed_jsonl_sionna)  # 重建 Sionna 数据库
        ns3_db.rebuild(docs_jsonl_ns3, embed_jsonl_ns3)  # 重建 NS-3 数据库

    log.info("Databases initialized.")

    # 初始化 Chat 对象
    sionna_chat = Chat(sionna_db, top_k=top_k)
    ns3_chat = Chat(ns3_db, top_k=top_k)

    # Main loop
    while True:
        # Prompt the user
        question = Prompt.ask(
            "Enter your question (convert c++ script into python version by input [bold yellow]convert[/]; quit by stroking [bold yellow]q[/] with [bold yellow]enter[/]).",
            default="List the supported features of the mesh module, such as Peering Management Protocol, Hybrid Wireless Mesh Protocol, and 802.11e compatible airtime link metric."
        )
        # Normalize user input for checking special commands
        question_stripped = question.strip().lower()

        domain = determine_domain(question, llm)

        if question_stripped == "q":
            # User wants to quit
            break

        elif question_stripped == "convert":
            # User wants to convert a C++ file to Python
            # ns-3 only
            cxx_path_str = Prompt.ask("Enter your c++ script absolute path")
            path_obj = Path(cxx_path_str).expanduser().resolve()

            if not path_obj.is_file():
                log.error(f"File not found: {path_obj}")
            else:
                # Read the file
                file_content = path_obj.read_text(encoding="utf-8")
                # Escape newlines
                file_content_escaped = file_content.replace("\n", "\\n")
                # Build the final prompt
                final_prompt = (
                    f"turn this ns3 c++ script into its python version: {file_content_escaped}"
                )
                # Send to chat
                print(final_prompt)
                # chat.send(final_prompt) # old
                ns3_chat.send(final_prompt)

        if domain.replace("'", "") == "sionna":
            log.info("Sionna domain detected. Querying Sionna database...")
            sionna_chat.send(question)
        elif domain.replace("'", "") == "ns-3":
            log.info("NS-3 domain detected. Querying NS-3 database...")
            ns3_chat.send(question)
        elif domain.replace("'", "") == "both":
            log.info("Both NS-3 and Sionna are relevant. Asking user for query order...")
            # **let user choose high → low or low → high**
            query_order = Prompt.ask(
                "Your question relates to both NS-3 (high-level network) and Sionna (low-level PHY)."
                "\n[bold green]1[/]: Start with NS-3 (high-level network) → then Sionna (low-level PHY)"
                "\n[bold blue]2[/]: Start with Sionna (low-level PHY) → then NS-3 (high-level network)",
                choices=["1", "2"],
                default="default to 2"
            )
            query_order = "high-low" if query_order == "1" else "low-high"

            # **执行 ordered query**
            final_answer = execute_ordered_query(question, sionna_chat, ns3_chat, query_order, llm)
        else:
            log.warning("Could not determine the domain. Returning GPT response...")
            response = answer_with_gpt(question, llm)
            show(response, title="[green]" + cfg.get("llm"))


def execute_ordered_query(original_question, sionna_chat, ns3_chat, query_order, llm):
    """
    先查询 high-level (NS-3) 然后 low-level (Sionna)，或反之。
    但第一个查询前要改写问题，让其符合该 level 的需求。
    """
    if query_order == "high-low":
        high_level_question = generate_specific_question(original_question, None, "ns-3", query_order, llm)
        high_level_response = ns3_chat.send(high_level_question)

        refined_low_level_question = generate_specific_question(original_question, high_level_response, "sionna",
                                                                query_order, llm)
        low_level_response = sionna_chat.send(refined_low_level_question)

        return f"**NS-3 Perspective:**\n{high_level_response}\n\n**Sionna Perspective:**\n{low_level_response}"

    elif query_order == "low-high":
        low_level_question = generate_specific_question(original_question, None, "sionna", query_order, llm)
        low_level_response = sionna_chat.send(low_level_question)

        refined_high_level_question = generate_specific_question(original_question, low_level_response, "ns-3",
                                                                 query_order, llm)
        high_level_response = ns3_chat.send(refined_high_level_question)

        return f"**Sionna Perspective:**\n{low_level_response}\n\n**NS-3 Perspective:**\n{high_level_response}"

    else:
        log.warning("Invalid query order. Defaulting to high-low.")
        return execute_ordered_query(original_question, sionna_chat, ns3_chat, "high-low", llm)


def generate_specific_question(original_question, reference_answer, target_domain, query_order, llm):
    """
    让 GPT 生成基于 `reference_answer` 的更具体问题，以适应 `target_domain` (`ns-3` / `sionna`)。
    `query_order` 决定是否是 `high → low` 还是 `low → high`。
    """
    if reference_answer:
        # **第二轮问题 (结合第一轮回答)**
        prompt = f"""
        You are refining a technical question for a layered network simulation.

        **Understanding query order (`{query_order}`):**
        - `"high → low"` means we first asked about **high-level network behaviors (NS-3)** and the response is `First response`. Now, refine the question to focus on **low-level physical interactions (Sionna)** that explain how these high-level behaviors are implemented.
        - `"low → high"` means we first asked about **low-level physical interactions (Sionna)** and the response is `First response`. Now, refine the question to explore how these interactions affect **higher-level network behavior (NS-3)**.

        **User's original question:**
        "{original_question}"

        **First response (`{query_order.split('-')[0]}` perspective):**
        "{reference_answer}"

        **Your task:**
        - You need to summary the `First response` content and add the content at the end of the generated question so the model answers the rewrite question has relevant information to refer to. Make sure to extract the key factors comprehensively.
        - Rewrite the question so that it explores `{target_domain}`'s role in the scenario.
        - Ensure that the new question logically extends from the **first response**.

        **Generate the refined question.** The key word {target_domain} must appears in the rewrite question.
        """
    else:
        # **第一轮问题 (无 `reference_answer`)**
        prompt = f"""
        You are refining a user's question for a specific network simulation domain.

        **Understanding query order (`{query_order}`):**
        - `"high → low"` means we want to ask **NS-3 (network level)** first, then move to **Sionna (physical level)**.
        - `"low → high"` means we want to ask **Sionna (physical level)** first, then move to **NS-3 (network level)**.

        **User's original question:**
        "{original_question}"

        Given the query order: `{query_order}`, rewrite the question to be **most relevant for {target_domain} to 
        finish the first step**. The key word {target_domain} must appears in the rewrite question. 

        If you want to use "high → low" or "low → high" these two phases, make sure you explains the meaning in your refined question!
        """

    response = completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=llm,
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        custom_llm_provider="openai",
        stream=True
    )

    # 处理流式响应
    prompt_answer = ""
    for chunk in response:
        if hasattr(chunk.choices[0].delta, 'content'):
            prompt_answer += chunk.choices[0].delta.content or ""

    prompt_answer = prompt_answer.strip().lower()
    print(prompt_answer)
    log.debug(f"LLM domain response: {prompt_answer}")
    return prompt_answer


def determine_domain(question: str, llm: str) -> str:
    """
    判断问题属于哪个领域（Sionna 或 NS-3）。
    """
    if "sionna" in question.lower():
        return "sionna"
    elif "ns-3" in question.lower() or "ns3" in question.lower():
        return "ns-3"
    elif "ns-3" in question.lower() and "ns3" in question.lower():
        return "both"

    # 使用 GPT 进一步判断
    domain_prompt = f"""You are an AI assistant. Your task is to classify the following question into one of three domains: 'sionna' or 'ns-3' or 'both'. If the question does not belong to either domain, respond with 'unknown'.

Question: {question}

Respond with either 'sionna', 'ns-3', 'both' or 'unknown'. Do not include additional explanations.
"""
    log.info(f"Querying LLM {llm} for domain classification...")
    response = completion(
        messages=[
            {"role": "user", "content": domain_prompt}
        ],
        model=llm,
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        custom_llm_provider="openai",
        stream=True
    )

    # 处理流式响应
    domain = ""
    for chunk in response:
        if hasattr(chunk.choices[0].delta, 'content'):
            domain += chunk.choices[0].delta.content or ""

    domain = domain.strip().lower()
    print(domain)
    log.debug(f"LLM domain response: {domain}")
    return domain


def answer_with_gpt(question: str, llm: str) -> str:
    """
    使用 GPT 直接回答问题（未能判断领域时）。
    """
    log.info(f"Using GPT model {llm} to answer the question...")
    response = completion(
        messages=[
            {"role": "user", "content": question}
        ],
        model=llm,
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        custom_llm_provider="openai",
        stream=True
    )

    return response  # 直接返回响应对象供 show() 函数处理


class Chat:
    def __init__(self, vecdb, top_k):
        self.messages = [{
            "role": "system",
            "content": (
                f"You are tasked with answering a question based on the context of a simulation framework"
                f" for wireless communication networks. The user cannot view the context."
                f" Please provide a comprehensive and self-contained answer."
                f" Ensure that your code is fully functional, with all input parameters pre-filled,"
                f" to minimize the need for further user interaction."
            )
        }]
        self.tools, self.jsons = \
            load_tools(['python_code_interpreter'])
        self.add_context = partial(answer, db=vecdb, top_k=top_k, llm_func=lambda x: x)
        self.enable_watchdog = enable_watchdog  # Default value, updated in the demo method

    def send(self, question):
        # if not self.messages:
        question = self.add_context(question)
        self.messages.append({
            "role": "user", "content": question
        })
        self.get()

    def get(self):
        response = completion(
            messages=self.messages,
            # tools=self.jsons,
            # tool_choice=tool_choice,
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
            model=cfg.get("llm"),
            custom_llm_provider="openai",
            stream=True,
        )
        show(response, title="[green]" + cfg.get("llm"))
        content = response.response_uptil_now
        self.messages.append({
            "role": "assistant", "content": content
        })
        conversation_history.append({"role": "ASSISTANT", "content": content})  # Add the assistant's response
        save_conversation_history_and_calculate_tokens(conversation_history,
                                                       "E:\python\\ns-3-rag\code\prompt_analysis.txt")
        output_path = Path("E:\python\\ns-3-rag\code\my_code.cc")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        # If Watchdog is enabled, process the content
        if self.enable_watchdog:
            log.info("Processing content automatically...")
            self.process_content(content)

        return content

    def process_content(self, content):
        # Determine the type of code snippet: C++ or Python
        cpp_code_match = re.search(r"```cpp\n(.*?)\n```", content, re.DOTALL)
        python_code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)

        if cpp_code_match:
            code = cpp_code_match.group(1)
            script_name = "my_generated_code.cc"
            file_path = Path(f"{path}/scratch/{script_name}")
            language = "C++"
        elif python_code_match:
            code = python_code_match.group(1)
            script_name = "my_generated_script_python.py"
            file_path = Path(f"{path}/scratch/{script_name}")
            language = "Python"
        else:
            log.warning("No valid code snippet found in the content.")
            return

        # Save the extracted code to the appropriate directory
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            log.info(f"{language} code snippet saved to {file_path}")
        except Exception as e:
            log.error(f"Failed to save {language} code snippet: {e}")
            return

        # Run the script using the ns-3 run command
        try:
            ns3_repo_path = Path(path)
            with open("output.log", "w+") as output_file:
                result = subprocess.run(
                    ["stdbuf", "-oL", "./ns3", "run", f"scratch/{script_name}"],
                    cwd=ns3_repo_path,
                    stdout=output_file,
                    stderr=output_file,
                    text=True
                )

            # Read the full contents of output.log to capture everything
            with open("output.log", "r") as output_file_r:
                full_output = output_file_r.read()

            # Check the return code to determine success or failure
            if result.returncode == 0:
                log.info(f"{language} script executed successfully. Full Output:\n{full_output}")
            else:
                log.error(f"{language} script execution failed with return code "
                          f"{result.returncode}. Full Output:\n{full_output}")

        except Exception as e:
            log.error(f"Failed to execute the {language} script: {e}")


def show(response, title="[green]" + cfg.get("llm")):
    cache = ""
    with Live(Panel(Markdown(cache), title=title), console=console, refresh_per_second=20) as live:
        for chunk in response:
            item = chunk.choices[0].delta.content or ""
            cache += item
            live.update(Panel(Markdown(cache), title=title))
    return


def answer(question, db, top_k=1, llm_func=lambda x: x):
    # Prepare the conversation history if it exists
    if len(conversation_history) > 0:
        conversation = "\n".join(
            [f"{entry['role']}: {entry['content']}" for entry in conversation_history]
        )
        conversation_part = (
            f"=== CONVERSATION HISTORY ===\n"
            f"{conversation}\n"
            f"=== END OF CONVERSATION HISTORY ===\n\n"  # Alert at the end of the conversation
        )
    else:
        conversation_part = ""  # No conversation history to add

    # Handle the case where top_k == 0
    if top_k == 0:
        prompt = (
            f"{conversation_part}"
            f"QUESTION: {question}\n"
        )
        # Append the current question to the conversation history
        conversation_history.append({"role": "USER", "content": question})
        save_conversation_history_and_calculate_tokens(conversation_history,
                                                       "E:\python\\ns-3-rag\code\prompt_analysis.txt")

        # Token calculation
        tokenizer = get_encoding("cl100k_base")  # Use 'cl100k_base' tokenizer
        num_tokens = len(tokenizer.encode(prompt))

        # Save the prompt and token count to a file
        with open("E:\python\\ns-3-rag\code\generated_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
            f.write(f"\n\nTOKEN COUNT: {num_tokens}")

        return llm_func(prompt)

    con_and_question = (
        f"{conversation_part}"
        f"QUESTION: {question}\n"
    )
    # Query the database for the top-k relevant documents
    # Path to save the file
    path = "E:\python\\ns-3-rag\code\con_and_question.txt"

    # Writing the string to a text file
    with open(path, "w") as file:
        file.write(con_and_question)
    context = db.query(con_and_question, top_k)
    documents = context['documents']
    docs = [f'Context {i}: {doc}' for i, doc in enumerate(documents)]
    ctx = '\n\n'.join(docs)

    # Combine the conversation history (if any), context, and the question
    prompt = (
        f"{conversation_part}"
        f"=== CONTEXT ===\n\n"
        f"{ctx}\n\n"
        f"=== QUESTION ===\n{question}\n"
    )

    # Append the current question to the conversation history
    conversation_history.append({"role": "USER", "content": question})

    # Save the conversation history and calculate tokens
    save_conversation_history_and_calculate_tokens(conversation_history, "E:\python\\ns-3-rag\code\prompt_analysis.txt")

    # Token calculation
    tokenizer = get_encoding("cl100k_base")  # Use 'cl100k_base' tokenizer
    num_tokens = len(tokenizer.encode(prompt))

    # Save the prompt and token count to a file
    with open("E:\python\\ns-3-rag\code\generated_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
        f.write(f"\n\nTOKEN COUNT: {num_tokens}")

    return llm_func(prompt)


def aug_answer(question, db, top_k=1, llm_func=lambda x: x):
    context = db.query(question, top_k)
    docs = [f'Context {i}: {doc}' for i, doc in enumerate(context['documents'])]
    # docs = context['documents'][0]
    ctx = '\n'.join(docs)
    prompt = (
        f"You will answer a question given the context related to ns-3, a discrete-event network simulator for Internet systems. "
        f"Note that the user cannot see the context. You shall provide a complete answer.\n"
        f"CONTEXT:\n"
        f"{ctx}\n"
        f"QUESTION: {question}\n"
    )
    return llm_func(prompt)


@app.command("batch")
def batch(
        input_jsonl: Annotated[Path, typer.Argument(help="a jonsl file stores line of question")],
        output_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of llm response")],
        vectordb: Annotated[str, typer.Option(help="name of the database")] = cfg.get("vectordb", "vectordb"),
        rerank: Annotated[bool, typer.Option(help="whether or not rerank the retrieved contexts")] = False,
        rebuild: Annotated[
            bool, typer.Option(help="if true, rebuild the database from docs_jsonl and embed_jsonl")] = False,
        docs_jsonl: Annotated[Path, typer.Option(help="a jsonl file stores line of doc")] = None,
        embed_jsonl: Annotated[Path, typer.Option(help="a jsonl file stores line of embedding")] = None,
        llm: Annotated[str, typer.Option(help="which LLM to use")] = cfg.get("llm"),
        top_k: Annotated[int, typer.Option(help="number of contexts to retrieve")] = cfg.get("top_k", 1),
        logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    cfg['llm'] = llm
    print(cfg['llm'])
    import os, tempfile
    from parallel_request import cli
    # initialize logging
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")
    # create database
    db = VectorDB(vectordb, rerank)
    if rebuild:
        assert docs_jsonl, f"Input docs_jsonl ({docs_jsonl}) doesn't exist."
        assert embed_jsonl, f"Input embed_jsonl ({embed_jsonl}) doesn't exist."
        db.rebuild(docs_jsonl, embed_jsonl)
    add_context = partial(answer, db=db, top_k=top_k, llm_func=lambda x: x)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        log.info(f"{tmp.name} created for temperal storage")
        with open(input_jsonl, 'r') as reader:
            for line in reader:
                print(line)
                question = json.loads(line.strip())  # Parse the line as JSON

                # Log the question into the global conversation history
                question = add_context(question)
                packed = json.dumps([
                    {
                        "role": "system",
                        "content": (
                            f"You are tasked with answering a question based on the context of a simulation framework"
                            f" for wireless communication networks. The user cannot view the context."
                            f" Please provide a comprehensive and self-contained answer."
                            f" Ensure that your code is fully functional, with all input parameters pre-filled,"
                            f" to minimize the need for further user interaction."
                        )
                    },
                    {"role": "user", "content": question},
                ])
                tmp.write(f"{packed}\n".encode("utf-8"))
        tmp.close()
        cli(tmp.name, output_jsonl,
            cfg.get('llm'), cfg.get('base_url'), cfg.get('api_key'),
            max_attempts=30,
            )
    finally:
        tmp.close()
        os.unlink(tmp.name)


if __name__ == "__main__":
    app()
