---
agent: 'agent'
description: 'Create a comprehensive README.md file for the project'
argument-hint: Update the project README using verified repository information
---

## Role

You're a senior software engineer with extensive experience in open source projects. You create appealing, informative, and easy-to-read README files.

## Task

1. Review the repository metadata, source layout, dependencies, and existing documentation needed for the README. Do not scan or modify read-only data, model, output, or notebook artifacts unless required to verify a claim.
2. Update the root `README.md` while preserving accurate existing content and repository-specific instructions. Create it only if it does not exist.
3. Include these essential sections:
   - **What the project does**: Clear project title and description
   - **Why the project is useful**: Key features and benefits
   - **How users can get started**: Installation/setup instructions with usage examples
   - **Where users can get help**: Support resources and documentation links
   - **Who maintains and contributes**: Maintainer information and contribution links only when verified from repository metadata; otherwise state that the information is unavailable

## Guidelines

### Content and Structure

- Focus only on information necessary for developers to get started using and contributing to the project
- Use clear, concise language and keep it scannable with good headings
- Include relevant code examples and usage snippets
- Add badges for build status, version, or license only when the corresponding metadata and URLs are verified; do not invent badges or project facts
- Keep content under 500 KiB (GitHub truncates beyond this)

### Technical Requirements

- Use GitHub Flavored Markdown
- Use relative links (e.g., `docs/CONTRIBUTING.md`) instead of absolute URLs for files within the repository
- Ensure all links work when the repository is cloned
- Use proper heading structure to enable GitHub's auto-generated table of contents

### What NOT to include

Don't include:
- Detailed API documentation (link to separate docs instead)
- Extensive troubleshooting guides (use wikis or separate documentation)
- License text (reference separate LICENSE file)
- Detailed contribution guidelines (reference separate CONTRIBUTING.md file)

Analyze the project structure, dependencies, and code to make the README accurate, helpful, and focused on getting users productive quickly. Verify relative links after editing and report the modified file and checks performed.

