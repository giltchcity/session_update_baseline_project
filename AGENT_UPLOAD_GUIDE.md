# Agent Upload Guide

This repository was uploaded by:

```text
1.1agent
```

This is a shared baseline repository. Later work may be done by:

```text
1.2agent
1.3agent
...
```

For this shared repository, do not leave the repo permanently configured as one
agent. Use a per-commit author instead:

```bash
git -c user.name="1.2agent" -c user.email="1.2agent@local" \
  commit -m "Update Base1.2 cross-session method"
```

Another example:

```bash
git -c user.name="1.3agent" -c user.email="1.3agent@local" \
  commit -m "Add NSS dataset joint test notes"
```

Do not use `--global` for agent names. If a local repo default was set
temporarily, remove it after the commit:

```bash
git config --local --unset user.name
git config --local --unset user.email
```

Recommended branch names:

```text
agent/1.1-final-map
agent/1.2-cross-session
agent/1.3-nss
```

To upload a new agent's separate project:

```bash
git init -b main
git add .
git -c user.name="<agent>" -c user.email="<agent>@local" \
  commit -m "Create agent project snapshot"
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

This keeps agent identity explicit per commit and keeps upload permission scoped
to one project instead of changing every repository on the machine.
