"""CLI entry point for agentify-example-rlm."""

from typing import Annotated
import typer
import asyncio

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent
from src.launcher import launch_evaluation

app = typer.Typer(help="Agentified RLM - Standardized agent assessment framework")


@app.command()
def green(
    host: Annotated[str, typer.Option()] = "localhost",
    port: Annotated[int, typer.Option()] = 9001,
    card_url: Annotated[str, typer.Option()] = "",
):
    """Start the green agent (assessment manager)."""
    del card_url
    start_green_agent(agent_name="green_agent", host=host, port=port)


@app.command()
def white(
    host: Annotated[str, typer.Option()] = "localhost",
    port: Annotated[int, typer.Option()] = 9002,
    card_url: Annotated[str, typer.Option()] = "",
):
    """Start the white agent (target being tested)."""
    del card_url
    start_white_agent(agent_name="repl_user", host=host, port=port  )


@app.command()
def launch():
    """Launch the complete evaluation workflow."""
    asyncio.run(launch_evaluation())


if __name__ == "__main__":
    app()
