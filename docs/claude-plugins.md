# Claude Code plugins in this repo

Plugins are declared in [`.claude/settings.json`](../.claude/settings.json) so every
session on this repo — local CLI, desktop, and cloud (claude.ai/code) — picks them up.
Cloud sessions have no interactive `/plugin` panel, so the settings file is the only
way to enable a plugin there.

## Currently enabled

| Marketplace | Plugin | Source |
| --- | --- | --- |
| `k12-teacher-skills` | `k12-education` | [anthropics/k12-teacher-skills](https://github.com/anthropics/k12-teacher-skills) |

`k12-education` (v0.9.0, Apache-2.0, by Anthropic + Learning Commons) adds four
K-12 teaching skills: `k12-lesson-plan-creation`, `k12-lesson-differentiation`,
`k12-lesson-prep`, and `k12-check-for-understanding`. Some of them expect the
Learning Commons Knowledge Graph MCP connector for US state academic standards;
without it the skills still run, just without standards lookup.

## Adding another plugin

Add its marketplace under `extraKnownMarketplaces` (`source: "github"` with
`repo: "owner/name"`, or a git URL / local path), then flip the plugin on under
`enabledPlugins` using the `plugin-name@marketplace-name` key. The plugin name is
the `name` field of the entry in the marketplace's `.claude-plugin/marketplace.json`,
which is not always the repo name.

## If a plugin doesn't load

The settings file registers the marketplace automatically once the folder is trusted,
but Claude Code may still report a plugin from an external source as "not installed"
until it is fetched once. In a terminal session run:

```bash
claude plugin marketplace add anthropics/k12-teacher-skills
claude plugin install k12-education@k12-teacher-skills --scope project
```

In an interactive session, `/plugin` → **Installed** shows the state, the **Errors**
tab shows load failures, and `/reload-plugins` applies changes without a restart.
