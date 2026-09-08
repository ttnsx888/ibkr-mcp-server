# IBKR MCP Server

A professional Model Context Protocol (MCP) server for Interactive Brokers API integration, designed for use with Claude Desktop and Claude Code.

[![CI](https://github.com/yourusername/ibkr-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/ibkr-mcp-server/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- ✅ **Multi-Account Support** - Switch between multiple IBKR accounts
- ✅ **Short Selling Analysis** - Shortable shares, borrow rates, margin requirements
- ✅ **Real-time Market Data** - Live quotes, historical data, options chains
- ✅ **Portfolio Management** - Positions, P&L, account summaries
- ✅ **Trading Operations** - Place, modify, cancel orders (with safety checks)
- ✅ **Auto-Reconnection** - Handles TWS/Gateway restarts gracefully
- ✅ **Production Ready** - Proper error handling, logging, and monitoring

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Interactive Brokers account with TWS or IB Gateway
- Claude Desktop or Claude Code

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ibkr-mcp-server.git
   cd ibkr-mcp-server
   ```

2. **Run the setup script:**
   ```bash
   # macOS/Linux
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   
   # Windows
   scripts\setup.bat
   ```

3. **Configure your settings:**
   ```bash
   cp .env.example .env
   # Edit .env with your IBKR settings
   ```

4. **Start TWS/IB Gateway** and enable API connections

5. **Test the server:**
   ```bash
   python -m ibkr_mcp_server.main --test
   ```

### Claude Integration

**Claude Desktop:**
Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "ibkr": {
      "command": "python",
      "args": ["-m", "ibkr_mcp_server.main"],
      "cwd": "/path/to/ibkr-mcp-server"
    }
  }
}
```

**Claude Code:**
```bash
claude mcp add ibkr 'python -m ibkr_mcp_server.main' --cwd /path/to/ibkr-mcp-server
```

## Usage Examples

### Basic Operations
```python
# Get portfolio across all accounts
"Show me my current portfolio"

# Switch accounts
"Switch to account DU7654321"

# Market data
"Get real-time quotes for AAPL, TSLA, MSFT"
```

### Short Selling Analysis
```python
# Complete short selling analysis
"Analyze short selling for GME, AMC, BBBY - show availability, borrow costs, and margin requirements"

# Check specific account
"Check shortable shares for TSLA in my paper trading account"
```

### Trading Operations
```python
# Place orders (paper trading recommended)
"Place a limit order to buy 100 shares of AAPL at $150"

# Check margin requirements
"What are the margin requirements for shorting 200 shares of TSLA?"
```

## Available Tools

| Tool | Description |
|------|-------------|
| `get_portfolio` | Current portfolio positions and P&L |
| `get_account_summary` | Account balances and key metrics |
| `switch_account` | Switch between IBKR accounts |
| `check_shortable_shares` | Short selling availability |
| `get_margin_requirements` | Margin requirements for securities |
| `get_borrow_rates` | Stock borrow rates (short selling costs) |
| `short_selling_analysis` | Complete short selling analysis |
| `get_market_data` | Real-time market quotes |
| `get_historical_data` | Historical price data |
| `place_order` | Place trading orders (with safety checks) |
| `get_connection_status` | Check IBKR connection status |

## Configuration

### Environment Variables
```env
# IBKR Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497=TWS Paper, 7496=TWS Live, 4001=Gateway
IBKR_CLIENT_ID=1
IBKR_IS_PAPER=true

# Logging
LOG_LEVEL=INFO
IBKR_MCP_LOG_FILE=  # default: ~/.trader/logs/ibkr_mcp_server.log; set "" to disable file logging

# Safety
ENABLE_LIVE_TRADING=false  # Set to true for live trading
MAX_ORDER_SIZE=1000  # Maximum order size
```

## Diagnostics

The server writes a rotating diagnostics log (INFO+, 5 MB x 3 backups) to
`~/.trader/logs/ibkr_mcp_server.log` by default. Set `IBKR_MCP_LOG_FILE` to
point it elsewhere, or to an empty string to disable file logging entirely
(the server always logs to stderr too — never stdout, which the stdio MCP
transport reserves for protocol frames). The log records every order
placement, every orderStatus transition, and every IBKR `errorEvent`
(reqId/code/message) at WARNING.

Because IBKR reports order rejections asynchronously via `errorEvent`
rather than as an exception from `placeOrder`, a placement can come back
with a bare status like `Cancelled` or `Inactive` and no attached reason.
To fix that, every tool result that reports an order status —
`stage_order`/`confirm_order`, `stage_stop_order`, `stage_bracket_order`,
`modify_live_order`, and order cancellation — carries two additional keys
alongside `status`:

- `ibkr_errors`: a list of `{code, message, ts}` objects — every
  `errorEvent` IBKR fired for that specific order (empty list if none).
- `last_error`: `"<code>: <message>"` for the most recent one, or `null`.

`get_connection_status` also exposes `recent_ibkr_errors`: the last 20
`errorEvent`s across all orders/requests, for an at-a-glance check without
tailing the log file.

### TWS/Gateway Setup
1. Start TWS or IB Gateway
2. Go to Configuration → API → Settings
3. Enable "ActiveX and Socket Clients"
4. Set socket port (7497 for paper, 7496 for live)
5. Add 127.0.0.1 to "Trusted IPs"
6. Check "Download open orders on connection"

## Development

### Setup Development Environment
```bash
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
black ibkr_mcp_server/
isort ibkr_mcp_server/
mypy ibkr_mcp_server/
```

## Deployment

### Auto-start on Boot (macOS)
```bash
python scripts/install_service.py --platform macos
```

### Auto-start on Boot (Linux)
```bash
python scripts/install_service.py --platform linux
```

### Docker Deployment
```bash
docker build -t ibkr-mcp-server .
docker run -d --name ibkr-mcp -p 8080:8080 ibkr-mcp-server
```

## Documentation

- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)  
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Safety & Disclaimers

⚠️ **Important Safety Notes:**
- Always test with paper trading first
- Verify all data in TWS before making trading decisions
- This software is for educational purposes
- Use at your own risk
- No warranty provided

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- [Issues](https://github.com/yourusername/ibkr-mcp-server/issues)
- [Discussions](https://github.com/yourusername/ibkr-mcp-server/discussions)
- [Wiki](https://github.com/yourusername/ibkr-mcp-server/wiki)
