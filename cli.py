"""Interactive CLI for the Multi-Agent Platform."""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
import typer

from agents import create_compiled_graph, AgentState
from langchain_core.messages import HumanMessage

console = Console()
app = typer.Typer()


async def run_interactive(task: str, max_cycles: int = 3):
    session_id = f"cli-{os.urandom(4).hex()}"
    graph, checkpointer = await create_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state: AgentState = {
        "task": task,
        "session_id": session_id,
        "messages": [HumanMessage(content=task)],
        "research_output": "",
        "code_output": "",
        "review_output": "",
        "current_agent": "researcher",
        "review_cycle": 0,
        "max_review_cycles": max_cycles,
        "review_decision": "",
        "awaiting_human": False,
        "human_feedback": "",
        "final_output": "",
        "error": "",
    }

    console.print(Panel(f"[bold cyan]Task:[/bold cyan] {task}", title="Multi-Agent Platform"))

    while True:
        state = await graph.ainvoke(initial_state if initial_state else None, config=config)
        initial_state = None  # Only pass on first invoke

        # Display agent outputs
        if state.get("research_output") and state.get("current_agent") in ("coder", "human"):
            console.print(Panel(
                Markdown(state["research_output"]),
                title="[green]Researcher Output[/green]",
                border_style="green",
            ))

        if state.get("code_output") and state.get("current_agent") in ("reviewer", "human"):
            console.print(Panel(
                Markdown(state["code_output"]),
                title="[blue]Coder Output[/blue]",
                border_style="blue",
            ))

        if state.get("review_output"):
            decision = state.get("review_decision", "")
            color = "green" if decision == "approve" else "yellow" if decision == "revise" else "red"
            console.print(Panel(
                Markdown(state["review_output"]),
                title=f"[{color}]Reviewer Output — {decision.upper()}[/{color}]",
                border_style=color,
            ))

        # Human checkpoint
        if state.get("awaiting_human"):
            console.print("\n[bold yellow]Human checkpoint reached. Review the outputs above.[/bold yellow]")
            feedback = Prompt.ask(
                "[yellow]Enter feedback (or press Enter to approve)[/yellow]",
                default="",
            )

            await graph.aupdate_state(
                config,
                {"human_feedback": feedback, "awaiting_human": False},
                as_node="human_checkpoint",
            )
            continue  # Re-invoke to resume

        # Final output
        if state.get("final_output") or state.get("current_agent") == "end":
            if state.get("final_output"):
                console.print(Panel(
                    Markdown(state["final_output"]),
                    title="[bold green]Final Output[/bold green]",
                    border_style="green",
                ))
            break

    console.print(f"\n[dim]Session ID: {session_id}[/dim]")


@app.command()
def run(
    task: str = typer.Argument(..., help="The task to execute"),
    max_cycles: int = typer.Option(3, "--max-cycles", "-m", help="Max review cycles"),
):
    """Run a task through the multi-agent pipeline."""
    asyncio.run(run_interactive(task, max_cycles))


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File path or text to ingest"),
    is_file: bool = typer.Option(False, "--file", "-f", help="Treat input as file path"),
):
    """Ingest documents into the RAG knowledge base."""
    from rag import VectorStore
    vs = VectorStore()
    if is_file:
        ids = vs.ingest_file(path)
        console.print(f"[green]Ingested {len(ids)} chunks from {path}[/green]")
    else:
        ids = vs.ingest_text(path)
        console.print(f"[green]Ingested {len(ids)} chunks[/green]")


if __name__ == "__main__":
    app()
