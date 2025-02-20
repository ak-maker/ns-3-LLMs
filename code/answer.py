""" an rag demo running in the terminal
"""
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


class Chat:
    def __init__(self, vecdb, top_k):
        self.messages = [{
            "role": "system",
            "content": (
                f"You are tasked with answering a question based on the context of \'Sionna,\'"
                f" a novel Python package for wireless simulation. The user cannot view the context."
                f" Please provide a comprehensive and self-contained answer."
                f" Ensure that your code is fully functional, with all input parameters pre-filled,"
                f" to minimize the need for further user interaction."
            )
        }]
        self.tools, self.jsons = \
            load_tools(['python_code_interpreter'])
        self.add_context = partial(answer, db=vecdb, top_k=top_k, llm_func=lambda x: x)

    def send(self, question):
        # if not self.messages:
        question = self.add_context(question)
        self.messages.append({
            "role": "user", "content": question
        })
        content = self.get()
        return content

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
        return content


def show(response, title="[green]" + cfg.get("llm")):
    cache = ""
    with Live(Panel(Markdown(cache), title=title), console=console, refresh_per_second=20) as live:
        for chunk in response:
            item = chunk.choices[0].delta.content or ""
            cache += item
            live.update(Panel(Markdown(cache), title=title))
    return


def answer(question, db, top_k=1, llm_func=lambda x: x):
    if top_k == 0:
        prompt = f"QUESTION: {question}\n"
        return llm_func(prompt)
    context = db.query(question, top_k)
    documents = context['documents']
    docs = [f'Context {i}: {doc}' for i, doc in enumerate(documents)]
    ctx = '\n\n'.join(docs)
    prompt = (
        f"CONTEXT:\n\n"
        f"{ctx}\n\n"
        f"QUESTION: {question}\n"
    )
    return llm_func(prompt)


def aug_answer(question, db, top_k=1, llm_func=lambda x: x):
    context = db.query(question, top_k)
    docs = [f'Context {i}: {doc}' for i, doc in enumerate(context['documents'])]
    # docs = context['documents'][0]
    ctx = '\n'.join(docs)
    prompt = (
        f"You will answer a question given the context related to sionna, a novel python package for wirless simulation. "
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
    from parallel_request_old_before_dm_change import cli
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
                question = json.loads(line)
                question = add_context(question)
                packed = json.dumps([
                    {"role": "system", "content": (
                        f"You are tasked with answering a question based on the context of \'Sionna,\'"
                        f" a novel Python package for wireless simulation. The user cannot view the context."
                        f" Please provide a comprehensive and self-contained answer."
                        f" Ensure that your code is fully functional, with all input parameters pre-filled,"
                        f" to minimize the need for further user interaction."
                    )},
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


# @app.command("demo_before") # 之前的，没有实现多个rag选择
# def demo(
#     docs_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of doc")],
#     embed_jsonl: Annotated[Path, typer.Argument(help="a jsonl file stores line of embedding")],
#     vectordb: Annotated[str, typer.Option(help="name of the database")] = cfg.get("vectordb", "vectordb"),
#     rerank: Annotated[bool, typer.Option(help="whether or not rerank the retrieved contexts")] = False,
#     rebuild: Annotated[bool, typer.Option(help="if true, rebuild the database from docs_jsonl and embed_jsonl")] = False,
#     top_k: Annotated[int, typer.Option(help="number of contexts to retrieve")] = cfg.get("top_k", 1),
#     llm: Annotated[str, typer.Option(help="which LLM to use")] = cfg.get("llm"),
#     logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
# ):
#     cfg['llm'] = llm
#     # initialize logging
#     log.setLevel(logging_level)
#     log.debug(f"Logging initialized at level {logging_level}")
#     # create database
#     db = VectorDB(vectordb, rerank=rerank)
#     if rebuild:
#         assert docs_jsonl, f"Input docs_jsonl ({docs_jsonl}) doesn't exist."
#         assert embed_jsonl, f"Input embed_jsonl ({embed_jsonl}) doesn't exist."
#         db.rebuild(docs_jsonl, embed_jsonl)
#     chat = Chat(db, top_k=top_k)
#     while question := Prompt.ask(
#         "Enter your question (quit by stroking [bold yellow]q[/] with [bold yellow]enter[/]):",
#         default="perform raytracing at munich. make sure the code is runable without modifications."
#         ):
#         if question == 'q': break
#         chat.send(question)

@app.command("demo")
def demo(
        docs_jsonl_sionna: Annotated[Path, typer.Argument(help="A JSONL file for Sionna documents")],
        embed_jsonl_sionna: Annotated[Path, typer.Argument(help="A JSONL file for Sionna embeddings")],
        docs_jsonl_ns3: Annotated[Path, typer.Argument(help="A JSONL file for NS-3 documents")],
        embed_jsonl_ns3: Annotated[Path, typer.Argument(help="A JSONL file for NS-3 embeddings")],
        vectordb_sionna: Annotated[str, typer.Option(help="Name of the Sionna database")] = "sionna_db",
        vectordb_ns3: Annotated[str, typer.Option(help="Name of the NS-3 database")] = "ns-3_db",
        rerank: Annotated[bool, typer.Option(help="Whether or not to rerank retrieved contexts")] = False,
        rebuild: Annotated[bool, typer.Option(help="If true, rebuild databases from JSONL files")] = False,
        top_k: Annotated[int, typer.Option(help="Number of contexts to retrieve")] = cfg.get("top_k", 1),
        llm: Annotated[str, typer.Option(help="Which LLM to use")] = cfg.get("llm"),
        logging_level: Annotated[int, typer.Option(help=LOGGING_HELP)] = logging.INFO,
):
    """Demo command for querying Sionna or NS-3 related questions."""

    # 初始化日志
    cfg['llm'] = llm
    log.setLevel(logging_level)
    log.debug(f"Logging initialized at level {logging_level}")

    # 创建数据库
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

    # 开始处理用户输入
    while question := Prompt.ask(
            "Enter your question (quit by stroking [bold yellow]q[/] with [bold yellow]enter[/]):",
            default="List the supported features of the mesh module, such as Peering Management Protocol, Hybrid Wireless Mesh Protocol, and 802.11e compatible airtime link metric.",
    ):
        if question == 'q':
            break

        # 判断领域
        domain = determine_domain(question, llm)

        # 根据领域调用数据库并返回响应
        if domain == "sionna" or domain == 'sionna':
            log.info("Sionna domain detected. Querying Sionna database...")
            sionna_chat.send(question)
        elif domain == "ns-3" or domain == 'ns-3':
            log.info("NS-3 domain detected. Querying NS-3 database...")
            ns3_chat.send(question)
        elif domain == "both" or domain == 'both' or domain =="'both'":
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


if __name__ == "__main__":
    app()
