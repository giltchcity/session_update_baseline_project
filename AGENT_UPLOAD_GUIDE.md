# Agent Upload Guide

This repository was uploaded by:

```text
1.1agent
```

The Git author configuration is repository-local, not global:

```bash
git config user.name "1.1agent"
git config user.email "1.1agent@local"
```

Do not use `--global` for agent names. Other agents should set their own name
inside their own project repository, for example:

```bash
git config user.name "1.2agent"
git config user.email "1.2agent@local"
```

To upload another agent's project:

```bash
git init -b main
git add .
git commit -m "Create agent project snapshot"
git remote add origin git@github.com:giltchcity/<repo-name>.git
git push -u origin main
```

SSH authentication is also per project when using a dedicated deploy key:

```bash
ssh-keygen -t ed25519 -C "<agent>@giltchcity" -f ~/.ssh/<repo-name>_ed25519 -N ""
git config core.sshCommand "ssh -i ~/.ssh/<repo-name>_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
```

Add the generated `.pub` key to that GitHub repository under:

```text
Settings -> Deploy keys -> Add deploy key -> Allow write access
```

This keeps agent identity and upload permission scoped to one project instead
of changing every repository on the machine.
