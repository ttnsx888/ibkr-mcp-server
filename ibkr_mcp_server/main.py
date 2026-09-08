"""Main entry point for IBKR MCP Server."""

import asyncio
import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import click
from mcp.server.stdio import stdio_server
from rich.console import Console
from rich.logging import RichHandler

from .client import ibkr_client
from .config import settings
from .tools import server


console = Console()

# Diagnostics (2026-09-08 PLTR incident): the server previously logged
# nothing to disk at all, so an IBKR rejection reason that scrolled past in
# stderr was gone for good. Default destination for the rotating diagnostics
# log; IBKR_MCP_LOG_FILE overrides it, and an explicit empty value disables
# file logging entirely (stderr-only).
DEFAULT_LOG_DIR = Path.home() / ".trader" / "logs"
DEFAULT_LOG_FILE = str(DEFAULT_LOG_DIR / "ibkr_mcp_server.log")
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def _resolve_log_file(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the diagnostics log file path.

    `explicit` (e.g. an explicit --log-file flag) wins when given. Otherwise
    IBKR_MCP_LOG_FILE wins when set — including an explicit empty string,
    which disables file logging entirely. With neither, use the default
    path under ~/.trader/logs.
    """
    if explicit is not None:
        return explicit
    if "IBKR_MCP_LOG_FILE" in os.environ:
        val = os.environ["IBKR_MCP_LOG_FILE"].strip()
        return val or None
    return DEFAULT_LOG_FILE


class GracefulKiller:
    """Handle shutdown signals gracefully."""
    
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        # Only log to stderr when running as MCP server
        logger = logging.getLogger(__name__)
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.kill_now = True


def setup_logging(level: str = "INFO", log_file: Optional[str] = None, mcp_mode: bool = False):
    """Setup logging configuration."""
    handlers = []

    # Always add file handler if specified. Rotating (5MB x 3) so a chatty
    # session can't grow this unbounded; the parent dir is created on demand
    # (~/.trader/logs doesn't exist until the first run).
    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(file_handler)
    
    # Only add console handler if NOT in MCP mode
    if not mcp_mode:
        handlers.append(RichHandler(console=console, show_time=True, show_path=False))
    else:
        # In MCP mode, log to stderr instead of stdout
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(stderr_handler)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Reduce noise from ib_async
    logging.getLogger('ib_async').setLevel(logging.WARNING)


async def test_connection():
    """Test IBKR connection and basic functionality."""
    console.print("[bold blue]🧪 Testing IBKR MCP Server...[/bold blue]")
    
    try:
        # Test connection
        console.print("📡 Testing IBKR connection...")
        await ibkr_client.connect()
        console.print("✅ Connection successful!")
        
        # Test basic functionality
        console.print("🔍 Testing basic functionality...")
        accounts = await ibkr_client.get_accounts()
        console.print(f"📊 Found {len(accounts)} accounts")
        
        # Test tools
        console.print("🛠️ Testing MCP tools...")
        tools = server.list_tools()
        console.print(f"⚙️ Loaded {len(tools)} tools")
        
        console.print("[bold green]✅ All tests passed![/bold green]")
        return True
        
    except Exception as e:
        console.print(f"[bold red]❌ Test failed: {e}[/bold red]")
        return False
    finally:
        await ibkr_client.disconnect()


async def run_server():
    """Run the MCP server with connection management."""
    logger = logging.getLogger(__name__)
    
    # Note: No console.print() calls here as they interfere with MCP protocol
    logger.info("Starting IBKR MCP Server...")
    
    try:
        # Start MCP server immediately - connection will be established on demand
        logger.info("Starting MCP server...")
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        try:
            await ibkr_client.disconnect()
        except:
            pass
        logger.info("Server shutdown complete")


@click.command()
@click.option('--test', is_flag=True, help='Test connection and exit')
@click.option('--log-level', default=settings.log_level, help='Logging level')
@click.option('--log-file', default=None,
              help=f'Log file path (default: {DEFAULT_LOG_FILE}; overridden by '
                   'env IBKR_MCP_LOG_FILE, empty string disables file logging)')
def cli(test: bool, log_level: str, log_file: Optional[str]):
    """IBKR MCP Server - Interactive Brokers integration for Claude."""
    setup_logging(log_level, _resolve_log_file(log_file), mcp_mode=not test)
    
    if test:
        # Run connection test
        success = asyncio.run(test_connection())
        sys.exit(0 if success else 1)
    else:
        # Run the server
        asyncio.run(run_server())


async def main():
    """Main entry point when called as module."""
    setup_logging(settings.log_level, _resolve_log_file(), mcp_mode=True)
    await run_server()


if __name__ == "__main__":
    cli()
