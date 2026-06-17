# apply_patch format

`apply_patch` applies a structured **diff envelope**. It is NOT a "write the
whole file" tool: every content line MUST carry a one-character prefix
(`+`, `-`, or a single space). The most common failure is pasting raw file
contents with no prefix (e.g. a line that is just `{`); that always fails with
"unexpected line" / "unexpected hunk line". Prefix every line.

## Envelope

The `patch` argument starts with `*** Begin Patch` and ends with
`*** End Patch`, with one or more file operations in between:

- `*** Add File: <path>` — create a new file. Every line of the new file is
  prefixed with `+`.
- `*** Delete File: <path>` — remove an existing file. No body.
- `*** Update File: <path>` — edit an existing file with one or more hunks
  (see below). May be followed by `*** Move to: <new path>` to rename.

Paths are relative to the workspace root: no leading `/`, no drive letter
(`C:`), no `..`.

## Hunks (only for Update File)

Begin each hunk with `@@`, or `@@ <anchor text>` to help locate the spot. Then:

- unchanged context lines are prefixed with a single space ` `
- removed lines are prefixed with `-`
- added lines are prefixed with `+`

Include a few surrounding context lines so the hunk can be located, and keep
hunks in top-to-bottom file order (do not jump back to an earlier anchor). Use
`*** End of File` on its own line inside a hunk to anchor at end-of-file.

## Examples

Create a new file — note the `+` on EVERY line, including `{` and `}`:

```
*** Begin Patch
*** Add File: package.json
+{
+  "name": "ts-fullstack-demo",
+  "version": "1.0.0"
+}
*** End Patch
```

Edit an existing file:

```
*** Begin Patch
*** Update File: src/server.ts
@@
 import express from "express";
-const port = 3000;
+const port = Number(process.env.PORT) || 3000;
*** End Patch
```

Rename and edit in one op:

```
*** Begin Patch
*** Update File: src/old.ts
*** Move to: src/new.ts
@@
-export const VERSION = "1";
+export const VERSION = "2";
*** End Patch
```

Delete a file:

```
*** Begin Patch
*** Delete File: obsolete.txt
*** End Patch
```

## Common mistakes to avoid

- Pasting file contents with no `+` prefix. Wrong: `{` — Right: `+{`.
- Using `*** Update File:` to create a NEW file. Use `*** Add File:` for new
  files; Update requires the file to already exist.
- Wrapping the patch in a `shell` command (`cat`, `echo`, heredoc). Call the
  `apply_patch` tool directly with the envelope as the `patch` argument.
- Absolute or `..` paths. Keep paths relative to the workspace root.
