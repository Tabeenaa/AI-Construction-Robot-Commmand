import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from db import get_decisions, get_stats, export_to_json, DB_PATH

console = Console()

def display_stats():
    """Display overall safety checker audit statistics."""
    stats = get_stats()
    
    if stats["total"] == 0:
        console.print("[yellow]No safety checker audit logs found in the database.[/yellow]")
        return
        
    refuse_pct = stats["refuse_rate_pct"]
    if refuse_pct > 50:
        refuse_color = "bold red"
    elif refuse_pct > 20:
        refuse_color = "bold yellow"
    else:
        refuse_color = "bold green"
        
    stats_text = (
        f"[bold white]Total Decisions Logged :[/bold white] [cyan]{stats['total']}[/cyan]\n"
        f"[bold white]Refusal Count          :[/bold white] [red]{stats['refuse_count']}[/red]\n"
        f"[bold white]Refusal Rate %         :[/bold white] [{refuse_color}]{stats['refuse_rate_pct']}%[/{refuse_color}]\n"
        f"[bold white]Average Latency (ms)   :[/bold white] [yellow]{stats['avg_inference_time_ms']:.1f} ms[/yellow] ({stats['avg_inference_time_ms']/1000.0:.2f} s)\n"
        f"[bold white]Maximum Latency (ms)   :[/bold white] [magenta]{stats['max_inference_time_ms']:.1f} ms[/magenta] ({stats['max_inference_time_ms']/1000.0:.2f} s)\n"
        f"[bold white]Total Tokens Generated :[/bold white] [green]{stats['total_tokens']}[/green]"
    )
    
    console.print(Panel(
        stats_text,
        title="[bold cyan]Constitutional AI Audit Statistics[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        expand=False
    ))

def display_decisions(decisions, title="Audit Trail"):
    """Display safety decisions in a clean table format."""
    if not decisions:
        console.print("[yellow]No matching records found.[/yellow]")
        return
        
    table = Table(
        title=f"[bold]{title}[/bold] (Showing {len(decisions)} records)",
        box=box.ROUNDED,
        header_style="bold magenta"
    )
    
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Timestamp", style="dim")
    table.add_column("Command", min_width=25)
    table.add_column("Verdict", justify="center")
    table.add_column("Violated Principles", style="yellow")
    table.add_column("Latency", justify="right", style="cyan")
    table.add_column("Tokens", justify="right", style="green")
    
    verdict_colors = {
        "ALLOW": "bold green",
        "MODIFY": "bold yellow",
        "REFUSE": "bold red"
    }
    
    for dec in decisions:
        try:
            dt = datetime.fromisoformat(dec["timestamp"])
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_str = dec["timestamp"]
            
        verdict = dec["decision"]
        v_styled = f"[{verdict_colors.get(verdict, 'white')}]{verdict}[/{verdict_colors.get(verdict, 'white')}]"
        
        principles = dec["principle_violated"]
        p_str = ", ".join(principles) if principles else "-"
        
        cmd_str = dec["raw_command"]
        if verdict == "MODIFY" and dec["modified_command"]:
            cmd_str += f"\n[dim]↳ MOD:[/dim] [cyan]{dec['modified_command']}[/cyan]"
            
        latency_s = dec["inference_time_ms"] / 1000.0
        
        table.add_row(
            str(dec["id"]),
            ts_str,
            cmd_str,
            v_styled,
            p_str,
            f"{latency_s:.2f}s",
            str(dec["tokens_generated"])
        )
        
    console.print(table)

def display_detailed_decisions(decisions):
    """Display safety decisions with detailed step-by-step reasoning panels."""
    if not decisions:
        console.print("[yellow]No matching records found.[/yellow]")
        return
        
    verdict_colors = {
        "ALLOW": "green",
        "MODIFY": "yellow",
        "REFUSE": "red"
    }
    
    for i, dec in enumerate(decisions, 1):
        verdict = dec["decision"]
        color = verdict_colors.get(verdict, "white")
        
        try:
            dt = datetime.fromisoformat(dec["timestamp"])
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_str = dec["timestamp"]
            
        principles = dec["principle_violated"]
        p_str = ", ".join(principles) if principles else "None"
        
        content = (
            f"[bold]Timestamp:[/bold] {ts_str}\n"
            f"[bold]Raw Command:[/bold] {dec['raw_command']}\n"
            f"[bold]Verdict:[/bold] [{color}]{verdict}[/{color}]\n"
            f"[bold]Violated Principles:[/bold] [yellow]{p_str}[/yellow]\n"
            f"[bold]Reason:[/bold]\n{dec['reason']}\n"
        )
        if dec["modified_command"]:
            content += f"[bold]Modified Command:[/bold] [cyan]{dec['modified_command']}[/cyan]\n"
            
        content += f"[dim]Latency: {dec['inference_time_ms']/1000.0:.2f}s | Tokens generated: {dec['tokens_generated']}[/dim]"
        
        console.print(Panel(
            content,
            title=f"[bold {color}]Decision #{dec['id']}[/bold {color}]",
            border_style=color,
            box=box.ROUNDED,
            expand=False
        ))
        if i < len(decisions):
            console.print()

def main():
    parser = argparse.ArgumentParser(
        description="Constitutional AI Robot Safety Checker - SQLite Audit Query Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-l", "--list-refusals",
        action="store_true",
        help="List all refused commands"
    )
    parser.add_argument(
        "-p", "--principle",
        type=str,
        help="Filter decisions by a specific safety principle number or name"
    )
    parser.add_argument(
        "-n", "--last",
        type=int,
        help="Show the last N decisions"
    )
    parser.add_argument(
        "-s", "--stats",
        action="store_true",
        help="Compute and display safety audit stats"
    )
    parser.add_argument(
        "-e", "--export",
        type=str,
        help="Export query results to a JSON file"
    )
    parser.add_argument(
        "-d", "--detail",
        action="store_true",
        help="Show detailed verdict analysis with reasoning"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all decisions"
    )
    
    args = parser.parse_args()
    
    # Check database exists
    if not DB_PATH.exists():
        console.print(f"[bold red]Error:[/bold red] Database file not found at {DB_PATH}.")
        console.print("Please run the safety checker first to log some decisions.")
        sys.exit(1)
        
    # Default view if no arguments passed
    if len(sys.argv) == 1:
        console.print(Panel.fit(
            "[bold cyan]Robot Safety Checker Audit Terminal[/bold cyan]\n"
            "[dim]SQLite-backed Constitutional AI Decision Auditing[/dim]",
            border_style="cyan"
        ))
        display_stats()
        console.print("\n[bold]Last 5 Decisions Logged:[/bold]")
        decisions = get_decisions(limit=5)
        display_decisions(decisions)
        console.print("\n[dim]Run with -h or --help for available query options (filtering, listing refusals, exports).[/dim]")
        sys.exit(0)
        
    # Determine filters
    decision_filter = "REFUSE" if args.list_refusals else None
    
    # Query decisions
    # If -n isn't set but --all is NOT set, default to None (which returns all matching rows in DESC order)
    limit = args.last
    
    decisions = get_decisions(
        limit=limit,
        decision_filter=decision_filter,
        principle_filter=args.principle
    )
    
    # Run exports
    if args.export:
        count = export_to_json(
            args.export,
            limit=limit,
            decision_filter=decision_filter,
            principle_filter=args.principle
        )
        console.print(f"[bold green]Success:[/bold green] Exported {count} decisions to {args.export}")
        sys.exit(0)
        
    # Display Stats if requested
    if args.stats:
        display_stats()
        # If no other action, exit
        if not (args.list_refusals or args.principle or args.last or args.all):
            sys.exit(0)
            
    # Build title
    title_parts = ["Safety Checker Decisions"]
    if args.list_refusals:
        title_parts.append("[Refusals]")
    if args.principle:
        title_parts.append(f"[Principle Filter: '{args.principle}']")
    if limit:
        title_parts.append(f"[Last {limit}]")
        
    title = " ".join(title_parts)
    
    # Render results
    if args.detail:
        display_detailed_decisions(decisions)
    else:
        display_decisions(decisions, title=title)

if __name__ == "__main__":
    main()
