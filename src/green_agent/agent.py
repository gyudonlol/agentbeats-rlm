"""Green agent implementation - manages assessment and evaluation."""

from typing import Dict
import uvicorn
import tomllib
import json
import random
import re
import time
from argparse import Namespace

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts
import dotenv



from src.my_util import parse_tags, my_a2a, rlm as rlm_utils
from src.my_util.repl import REPLEnv

dotenv.load_dotenv()


REPL_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {{header}} section: {{info}}")
    buffers.append(f"{{header}}: {{summary}}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""

# Prompt at every step to query root LM to make a decision
USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original query: \"{query}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:""" 
def next_action_prompt(query: str, iteration: int = 0, final_answer: bool = False) -> Dict[str, str]:
    if final_answer:
        return {"role": "user", "content": "Based on all the information you have, provide a final answer to the user's query."}
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your context yet. Your next action should be to look through, don't just provide a final answer yet.\n\n"
        return {"role": "user", "content": safeguard + USER_PROMPT.format(query=query)}
    else:
        return {"role": "user", "content": "The history before is your previous interactions with the REPL environment. " + USER_PROMPT.format(query=query)}


def load_agent_card_toml(agent_name):
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


def extract_first_number(text: str) -> str:
    """
    Extract the first instance of consecutive numbers from a string.

    Args:
        text: The string to search for numbers

    Returns:
        The first consecutive number found as a string, or None if no numbers found
    """
    match = re.search(r'\d+', text)
    if match:
        return match.group(0)
    return ""



def generate_massive_context(num_lines: int = 1_000_000, answer: str = "1298418") -> str:
    print("Generating massive context with 1M lines...")
    
    # Set of random words to use
    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
    
    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))
    
    # Insert the magic number at a random position (somewhere in the middle)
    magic_position = random.randint(400000, 600000)
    lines[magic_position] = f"The magic number is {answer}"
    
    print(f"Magic number inserted at position {magic_position}")
    
    return "\n".join(lines)


async def ask_agent_to_solve(white_agent_url, task, max_num_steps=30):
    total_cost = 0.0
    info = {}
    reward = 0.0
    max_character_length = 100000

    answer = str(random.randint(1000000, 9999999))
    context = generate_massive_context(num_lines=1_000_000, answer=answer)
    context_data, context_str = rlm_utils.convert_context_for_repl(context)
    repl_env = REPLEnv(
        context_json=context_data, 
        context_str=context_str,
    )


    next_green_message = REPL_SYSTEM_PROMPT + "\n\n" + next_action_prompt(task)["content"]
    context_id = None
    for step in range(max_num_steps):
        print(
            f"@@@ Green agent: Sending message to white agent{'ctx_id=' + str(context_id) if context_id else ''}... -->\n{next_green_message}"
        )
        white_agent_response = await my_a2a.send_message(
            white_agent_url, next_green_message, context_id=context_id
        )
        res_root = white_agent_response.root
        assert isinstance(res_root, SendMessageSuccessResponse)
        res_result = res_root.result
        assert isinstance(
            res_result, Message
        )  # though, a robust design should also support Task
        if context_id is None:
            context_id = res_result.context_id
        else:
            assert context_id == res_result.context_id, (
                "Context ID should remain the same in a conversation"
            )

        text_parts = get_text_parts(res_result.parts)
        assert len(text_parts) == 1, (
            "Expecting exactly one text part from the white agent"
        )
        white_text = text_parts[0]
        print(f"@@@ White agent response:\n{white_text}")
        # parse the action out
        code_blocks = rlm_utils.find_code_blocks(white_text)
        result_messages = []
        if code_blocks:
            for code in code_blocks:
                result = repl_env.code_execution(code)
                result_message = rlm_utils.format_execution_result(result.stdout, result.stderr, result.locals)
                if len(result_message) > max_character_length:
                    result_message = result_message[:max_character_length] + "..."
                result_messages.append(f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result_message}")
            next_green_message = "\n\n".join(result_messages) + "\n\n" + next_action_prompt(task, iteration=step + 1)["content"]
        else:
            next_green_message = next_action_prompt(task, iteration=step + 1)["content"]

        final_tuple = rlm_utils.find_final_answer(white_text)
        if final_tuple:
            final_type, final_content = final_tuple
            if final_type == "FINAL":
                final_content = extract_first_number(final_content)
                reward = str(final_content) == answer
                break
            elif final_type == "FINAL_VAR":
                final_content = repl_env.final_var(final_content.strip())
                final_content = extract_first_number(final_content)
                reward = str(final_content) == answer
                break


    return Namespace(
        reward=reward,
        info=info,
        messages=[],  # incompatible, thus removed
        total_cost=total_cost,
    )


class RlmGreenAgentExecutor(AgentExecutor):
    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # parse the task
        print("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]
        env_config_str = tags["env_config"]
        env_config = json.loads(env_config_str)

        # set up the environment
        # migrate from https://github.com/sierra-research/tau-bench/blob/4754e6b406507dbcbce8e8b3855dcf80aaec18ac/tau_bench/run.py#L20
        print("Green agent: Setting up the environment...")
        assert len(env_config["task_ids"]) == 1, (
            "Only single task supported for demo purpose"
        )
        env_config["task_ids"][0]
        metrics = {}

        print("Green agent: Starting evaluation...")
        timestamp_started = time.time()
        res = await ask_agent_to_solve(white_agent_url, "I'm looking for a magic number. What is it?")

        metrics["time_used"] = time.time() - timestamp_started
        result_bool = metrics["success"] = res.reward == 1
        result_emoji = "✅" if result_bool else "❌"

        print("Green agent: Evaluation complete.")
        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Finished. White agent success: {result_emoji}\nMetrics: {metrics}\n"
            )
        )  # alternative, impl as a task-generating agent

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="rlm_green_agent", host="localhost", port=9001):
    print("Starting green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url  # complete all required card fields

    request_handler = DefaultRequestHandler(
        agent_executor=RlmGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
