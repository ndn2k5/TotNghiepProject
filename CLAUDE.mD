<!-- GSD Configuration — managed by get-shit-done installer -->

# Instructions for GSD (Beginner-Safe Mode)

<!--
This file is designed for users who are unfamiliar with Git, GitHub, and GSD workflows.
It prioritizes safety, clarity, and step-by-step guidance.
-->

## Core Behavior

- Always assume the user may not understand Git, GitHub, or GSD workflows.
- Explain actions in simple terms.
- Always recommend exact commands the user can copy and run.
- Never assume prior setup.

## Command Handling

- Treat `/gsd-*` or `gsd-*` as explicit command invocations.
- Load matching files from `.github/skills/gsd-*`.
- When a command requires a subagent, prefer agents from `.github/agents`.

## Safety Rules (IMPORTANT)

- DO NOT run GSD workflows unless explicitly requested by the user.
- DO NOT auto-commit or push changes without user confirmation.
- Always explain what will happen before suggesting Git commands.

## Git Workflow (Safe Mode)

<!-- Always sync + run GSD resume workflow to sync  before starting work -->
Before starting any task, recommend:

```bash
git pull --rebase
````

run gsd resume worklow to sync STATE.md
<!-- Standard safe commit flow -->

After completing work, suggest:

```bash
git add .
git commit -m "describe changes clearly"
git push
```

- Never force push.
- If merge conflicts occur, guide the user step-by-step instead of forcing resolution.

## Beginner Guidance Mode

- Always include a **"What to do next"** section.
- Provide copy-pasteable commands.
- Avoid jargon, or explain it briefly when used.
- Warn clearly before any risky operation.

## Completion Loop

After finishing any task:

1. Summarize what was done in simple terms
2. Suggest the next logical step
3. Ask the user what they want to do next

Repeat this loop until the user explicitly says they are done.

## Goal

Help the user get things done reliably **without breaking their repository or causing confusion**.

<!-- /GSD Configuration -->
