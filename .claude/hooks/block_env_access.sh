#!/usr/bin/env bash
# block_env_access.sh
# Blocks Read/Edit/Write/Bash tool calls that target .env files.
# Receives the tool input as JSON on stdin.

set -euo pipefail

input=$(cat)
tool=$(echo "$input" | jq -r '.tool_name // ""')

deny() {
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"'"$1"'"}}'
  exit 0
}

MSG="Access to .env files is blocked by the project security policy (see AGENTS.md)."

case "$tool" in
  Read|Edit|Write)
    file=$(echo "$input" | jq -r '.tool_input.file_path // ""')
    if echo "$file" | grep -qE '(^|/)\.env'; then
      deny "$MSG"
    fi
    ;;
  Bash)
    cmd=$(echo "$input" | jq -r '.tool_input.command // ""')
    # Match .env when preceded by a space (or start of string).
    # Prefix-only check avoids BSD grep escaping issues with quotes,
    # and still catches: cat .env, source .env, source .env.local,
    # bash -c 'cat .env' — while allowing --flag=".env" style arguments.
    if echo "$cmd" | grep -qE '(^|[[:space:]])\.env'; then
      deny "$MSG"
    fi
    ;;
esac
