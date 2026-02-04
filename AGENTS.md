# Repository Guidelines

## Project Structure & Module Organization
This repository tracks the `langsplain/` static web app. Core files live in `langsplain/`:
- `index.html`: page structure and UI sections
- `style.css`: global styles, responsive rules, theme variables
- `app.js`: app bootstrap, section state, and event wiring
- `modules/`: focused feature modules (for example `attention-demo.js`, `training-diagram.js`, `kv-cache-demo.js`, `math-utils.js`)

When adding features, prefer a new module in `modules/` and connect it through `app.js` rather than expanding one large file.

## Build, Test, and Development Commands
Run from `langsplain/`:
- `python3 -m http.server 8000` - start a local server (recommended for ES modules)
- `npx serve .` - alternative static server

Then open `http://localhost:8000`. There is no build step; deployment is static-file hosting.

## Coding Style & Naming Conventions
- Use ES module syntax (`import`/`export`) and prefer `const` over `let`.
- Match existing JavaScript style: 4-space indentation, semicolons, descriptive identifiers.
- Module filenames are kebab-case (`section-switcher.js`); function names are camelCase; constructor/class names are PascalCase (for example `AttentionDemoUI`).
- Keep functions focused and add brief comments only for non-obvious math or visualization logic.
- Reuse existing CSS custom properties and component naming patterns in `style.css`.

## Testing Guidelines
There is currently no automated test framework in this repo; rely on manual validation before opening a PR.
- Verify all three sections (Architecture, Training, Inference) load and switch correctly.
- Exercise interactive demos and guided tour steps end-to-end.
- Check responsive behavior on mobile/tablet widths and confirm dark-mode/readability states.
- Include reproduction steps for bug fixes in PR notes.

## Commit & Pull Request Guidelines
- Write commit messages in short imperative form (example: `Fix centered final tour step on mobile`).
- Keep commits single-purpose; separate refactors from behavior changes.
- PRs should include: a clear summary, linked issue (if applicable), UI screenshots/GIFs for visual changes, and a brief manual test checklist.
- Call out any educational simplifications or terminology updates so reviewers can verify conceptual accuracy.
