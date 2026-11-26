# Imagies MCP bridge

Set `IMAGIES_URL` to the base URL of your trusted server. Set `IMAGIES_AUTH_TOKEN` if the server requires a bearer token. The bridge appends `/v1/mcp`. Use HTTPS outside localhost.

The bridge launches pinned `mcp-remote` when invoked and sends MCP calls to that server. Stage images and download results over REST.
