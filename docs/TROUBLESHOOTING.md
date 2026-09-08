# Troubleshooting

*To be completed: Add troubleshooting steps and common issues for IBKR MCP Server.*

## Common Issues

### Connection Problems
- Check TWS/Gateway is running
- Verify API settings are enabled
- Check port configuration

### Installation Issues
- Ensure Python 3.10+ is installed
- Check virtual environment setup
- Verify dependencies

### MCP Integration
- Check Claude Desktop configuration
- Verify server is running
- Check logs for errors

### Order rejected / cancelled with no visible reason
IBKR reports rejections asynchronously via `errorEvent`, not as an
exception, so a placement can return a bare `status` of `Cancelled` or
`Inactive`. Check the `ibkr_errors` / `last_error` keys on the tool result
(`stage_order`/`confirm_order`, `stage_stop_order`, `stage_bracket_order`,
`modify_live_order`, cancel) first, or `get_connection_status`'s
`recent_ibkr_errors` for the last 20 errorEvents server-wide. For the full
history, tail the rotating diagnostics log — default
`~/.trader/logs/ibkr_mcp_server.log`, overridable via `IBKR_MCP_LOG_FILE`
(see README "Diagnostics"):
```
tail -f ~/.trader/logs/ibkr_mcp_server.log
```

*Full troubleshooting guide coming soon.*
