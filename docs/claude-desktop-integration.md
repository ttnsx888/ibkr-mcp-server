# Claude Desktop Integration Guide

## Overview

This guide shows how to integrate the IBKR MCP Server with Claude Desktop, allowing you to interact with Interactive Brokers directly through Claude.

## Installation Steps

### 1. Ensure IBKR MCP Server is Installed

First, make sure the IBKR MCP server is properly installed:

```bash
cd /path/to/ibkr-mcp-server
source venv/bin/activate
pip install -e .
```

Test the installation:
```bash
ibkr-mcp-server --test
```

### 2. Add to Claude Desktop Configuration

The IBKR MCP server has been added to your Claude Desktop configuration at:
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

Configuration entry:
```json
{
  "mcpServers": {
    "ibkr-mcp-server": {
      "command": "/Users/macbook2024/Dropbox/AAA Backup/A Complete/IBKR MCP/ibkr-mcp-server/venv/bin/python3",
      "args": ["-m", "ibkr_mcp_server"],
      "env": {
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "7497",
        "IBKR_CLIENT_ID": "1",
        "IBKR_IS_PAPER": "true",
        "ENABLE_LIVE_TRADING": "false",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

After adding the configuration:
1. Quit Claude Desktop completely
2. Restart Claude Desktop
3. The IBKR MCP server will automatically start when Claude Desktop loads

## Available Commands

Once integrated, you can use these commands in Claude Desktop:

### Portfolio Management
- **"Show my portfolio"** - Get current positions and P&L
- **"Get my account summary"** - View account balances and metrics
- **"What accounts do I have?"** - List available IBKR accounts
- **"Switch to account [ID]"** - Change active account

### Short Selling Analysis
- **"Check if AAPL is shortable"** - Check short availability for a stock
- **"What are the margin requirements for TSLA?"** - Get margin info
- **"Analyze short selling for NVDA"** - Complete short selling analysis

### Connection & Status
- **"Check my IBKR connection"** - Verify TWS/Gateway connection
- **"What's my connection status?"** - Get detailed connection info

## IBKR Setup Requirements

### TWS (Trader Workstation) Setup

1. **Download and Install TWS**
   - Go to [IBKR Client Portal](https://www.interactivebrokers.com/en/trading/trader-workstation.php)
   - Download and install TWS

2. **Enable API Access**
   - Open TWS
   - Go to File → Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Set Socket port to `7497` (paper) or `7496` (live)
   - Check "Read-Only API"

3. **Configure API Settings**
   - Master API client ID: `0`
   - Read-Only API: `Yes` (recommended for safety)
   - Download open orders on connection: `Yes`

### IB Gateway Setup (Alternative)

For lighter resource usage, you can use IB Gateway instead of TWS:

1. **Download IB Gateway**
   - Download from IBKR website
   - Install and configure similar to TWS

2. **API Configuration**
   - Same API settings as TWS
   - Port 7497 for paper trading
   - Port 7496 for live trading

## Environment Variables

You can customize the server behavior using environment variables:

```bash
# Connection Settings
IBKR_HOST=127.0.0.1          # IBKR host (default: 127.0.0.1)
IBKR_PORT=7497               # IBKR port (7497=paper, 7496=live)
IBKR_CLIENT_ID=1             # Client ID (default: 1)
IBKR_IS_PAPER=true           # Paper trading mode (default: true)

# Safety Settings
ENABLE_LIVE_TRADING=false    # Enable live trading (default: false)
MAX_ORDER_SIZE=1000          # Maximum order size

# Logging
LOG_LEVEL=INFO               # Logging level
LOG_FILE=/tmp/ibkr-mcp.log   # Log file path
```

## Safety Features

The IBKR MCP server includes several safety features:

- **Paper Trading Default**: Connects to paper trading by default
- **Live Trading Disabled**: Live trading is disabled by default
- **Read-Only Mode**: Focuses on data retrieval, not order placement
- **Input Validation**: Validates all symbols and parameters
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: Prevents API overload

## Testing the Integration

### Basic Connection Test

Ask Claude:
```
"Check my IBKR connection status"
```

Expected response when TWS is not running:
```json
{
  "connected": false,
  "host": "127.0.0.1",
  "port": 7497,
  "client_id": 1,
  "current_account": null,
  "available_accounts": [],
  "paper_trading": true
}
```

### Portfolio Test (requires TWS running)

Ask Claude:
```
"Show me my portfolio"
```

### Account Test (requires TWS running)

Ask Claude:
```
"What accounts do I have access to?"
```

## Troubleshooting

### Common Issues

1. **"Connection refused" error**
   - Make sure TWS or IB Gateway is running
   - Verify API is enabled in TWS settings
   - Check port number (7497 for paper, 7496 for live)

2. **"Module not found" error**
   - Verify the Python path in claude_desktop_config.json
   - Make sure the virtual environment is activated
   - Reinstall the package: `pip install -e .`

3. **"Permission denied" error**
   - Check file permissions on the Python executable
   - Verify Claude Desktop has necessary permissions

### Debug Mode

Enable debug logging by updating the environment variables:
```json
"env": {
  "LOG_LEVEL": "DEBUG"
}
```

### Log Files

Check logs for issues (default path; override with `IBKR_MCP_LOG_FILE`):
```bash
tail -f ~/.trader/logs/ibkr_mcp_server.log
```

## Configuration Backup

Your original configuration has been backed up to:
```
~/Library/Application Support/Claude/claude_desktop_config.json.backup
```

To restore the original configuration:
```bash
cp ~/Library/Application\ Support/Claude/claude_desktop_config.json.backup ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/ArjunDivecha/ibkr-mcp-server/issues
- Documentation: [README.md](../README.md)
- Troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## License

This integration is part of the IBKR MCP Server project and is licensed under the MIT License. 