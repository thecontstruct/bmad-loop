# claude-trust-prompt-parked.pipe-pane.log

Evidence for DW-339: raw `tmux pipe-pane` bytes of a claude trust prompt left parked,
untouched, for 80 s.

- Claude Code 2.1.284, tmux 3.7c, Linux; captured 2026-09-28.
- Pane 120x40 on a private tmux server (`-f /dev/null`); plain `claude`, no flags,
  launched in a fresh untrusted directory; `pipe-pane -o 'cat >> log'`.
- Log size, sampled every 5 s from launch: 0 B at 0 s, then 1,451 B at every sample
  from 5 s through 80 s. The prompt renders once and is byte-static; no timer repaint.
- Redaction: the one workspace path line was replaced with `/tmp/dw339/untrusted-proj`
  (1,451 B raw -> 1,359 B committed). Nothing else was changed.
- The tail is terminal capability queries (DA1, OSC 11, XTVERSION, DECRQM 2026), not
  a repaint.
- Words are separated by cursor-position escapes (`Enter\e[8Gto\e[11Gconfirm`), so
  a raw-byte substring match does not find the footer text; the rendered pane does.
- Line endings are CR CR LF; `.gitattributes` marks `*.pipe-pane.log` `-text`, so no
  `autocrlf` checkout rewrites them.
