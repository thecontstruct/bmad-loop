## Unity scene-save discipline (bmad-loop)

You are driving a live Unity Editor over MCP. Unity-MCP scene-mutating tools mark
scenes dirty but never save them, and a dirty or unsaved-on-quit scene raises modal
Editor dialogs ("scene changed on disk", "save changes before closing") that freeze
the Editor and every subsequent MCP call. Follow these rules:

- After any scene-mutating MCP batch (`gameobject-*`, `*component*`, `tilemap-*`,
  `cinemachine-*`, prefab instantiate), call `scene-save` for **each modified open
  scene** BEFORE any of: running tests (`tests-run` aborts on dirty scenes), any
  `git` command, entering play mode (`editor-application-set-state`), or ending the
  session.
- Never rewrite an open scene's `.unity`/`.prefab` on disk directly — prefer the MCP
  tools. If a disk edit is unavoidable: `scene-save` first, make the edit, then
  `assets-refresh`.
- A `SceneAutoSaveGuard` auto-saves dirty scenes after ~5s idle, but do **not** rely
  on it for ordering — call `scene-save` explicitly at the boundaries above.
