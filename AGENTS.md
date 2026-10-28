# Repository Rules

Default mode: READ ONLY.

Before changing code:

- Show the list of files that will be modified.
- Show the exact diff.
- Explain why each change is needed.
- Wait for explicit approval.

Never:
- Run destructive git commands.
- Delete files.
- Rename files.
- Modify Dockerfiles.
- Modify deployment configuration.
- Modify MLflow tracking code.

without explicit approval.

For migrations:
- Propose a migration plan first.
- Wait for approval.
- Then generate the code changes.