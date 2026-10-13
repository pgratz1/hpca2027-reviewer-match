# Nudge templates

One file per message kind, used by `scripts/discussion_nudges.py` (`make nudges`).
Plain text, one HotCRP comment each. Placeholders, filled per paper:

| placeholder   | value                                                        |
|---------------|--------------------------------------------------------------|
| `{mentions}`  | `@Given Family` for every person the comment is aimed at     |
| `{names}`     | the same people without the `@`                              |
| `{n}`         | how many of them                                             |
| `{pid}`       | the paper number                                             |
| `{title}`     | the paper title                                              |
| `{lead}`      | the discussion lead's name, or `the discussion lead`         |
| `{due}`       | `--due` as given, or empty                                   |
| `{due_clause}`| ` by {due}` when `--due` was given, otherwise empty          |

`--template NAME` picks `NAME.txt` here (or a path). `--text` gives the whole
message on the command line instead. Nothing here may name a person.
