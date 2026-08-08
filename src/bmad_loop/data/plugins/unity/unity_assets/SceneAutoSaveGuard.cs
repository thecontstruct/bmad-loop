// bmad-loop-scene-guard-version: 1.0.0
//
// Scene auto-save guard for bmad-loop-driven Unity projects.
//
// Unity-MCP GameObject tools (com.ivanmurzak.unity.mcp) call MarkSceneDirty and
// never save, so a project driven by the shared editor accumulates a chronically
// dirty scene. That dirty state is what makes Unity raise the two run-stalling
// modal dialogs:
//   * "Scene '<name>' has been changed on disk. Reload the scene?" — pops when git
//     or an agent rewrites the open dirty scene's .unity file on disk;
//   * "Do you want to save the changes you made ...?" — pops on editor quit.
// A modal dialog freezes EditorApplication.update, which the MCP plugin uses to
// dispatch tool calls, so every subsequent MCP call times out and the run stalls.
//
// This guard fixes the root cause: it keeps loaded, on-disk scenes clean by
// debounce-saving them shortly after they go dirty, and on quit it saves (or, for
// an unsaveable untitled scene, discards) so no modal ever appears. Untitled scenes
// are never saved automatically — that would pop a "Save As" modal, exactly what we
// are avoiding.
//
// bmad-loop seeds this file into consumer projects; story-finalize's `git add -A`
// then commits it into the project — that is intended (the guard travels with the
// repo so any editor opening it is protected). Toggle it from the BmadLoop menu.
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BmadLoop.Unity.Editor
{
    [InitializeOnLoad]
    internal static class SceneAutoSaveGuard
    {
        // EditorPrefs toggle (default ON) + its two menu items.
        private const string EnabledPref = "BmadLoop.SceneAutoSaveGuard.Enabled";
        private const string EnabledMenu = "BmadLoop/Scene Auto-Save Guard";
        private const string SaveNowMenu = "BmadLoop/Save Open Scenes Now";

        // Debounce window: each sceneDirtied event pushes the deadline out this far,
        // so a burst of MCP edits collapses into a single save once they settle.
        private const double DebounceSeconds = 5.0;

        // The timeSinceStartup deadline for the pending save; < 0 means "nothing
        // pending" — the cheap early-out the per-frame update handler checks first.
        private static double s_dueAt = -1.0;

        static SceneAutoSaveGuard()
        {
            EditorSceneManager.sceneDirtied += OnSceneDirtied;
            EditorApplication.update += OnUpdate;
            EditorApplication.wantsToQuit += OnWantsToQuit;
            EditorApplication.quitting += OnQuitting;
            // A scene can already be dirty from before this domain reload — an MCP
            // tool dirtied it, then a recompile reloaded the domain and reset our
            // pending state. Arm immediately so that dirty scene still gets flushed.
            if (HasDirtyPathedScene())
            {
                Arm();
            }
        }

        private static bool Enabled => EditorPrefs.GetBool(EnabledPref, true);

        private static void Arm()
        {
            s_dueAt = EditorApplication.timeSinceStartup + DebounceSeconds;
        }

        private static void OnSceneDirtied(Scene scene)
        {
            Arm();
        }

        private static void OnUpdate()
        {
            // Cheap early-out: nothing pending is the overwhelmingly common case, so
            // this handler must cost next to nothing when idle.
            if (s_dueAt < 0.0)
            {
                return;
            }
            if (EditorApplication.timeSinceStartup < s_dueAt)
            {
                return;
            }
            if (!Enabled)
            {
                s_dueAt = -1.0; // toggled off while a save was pending — drop it
                return;
            }
            // Never save mid-transition: entering/exiting play mode, compiling,
            // importing/refreshing assets, building a player, or editing a prefab in
            // isolation (the open scenes aren't the active editing target then).
            // Re-arm and try again on a later frame.
            if (EditorApplication.isPlayingOrWillChangePlaymode
                || EditorApplication.isCompiling
                || EditorApplication.isUpdating
                || BuildPipeline.isBuildingPlayer
                || PrefabStageUtility.GetCurrentPrefabStage() != null)
            {
                Arm();
                return;
            }
            s_dueAt = -1.0;
            SaveDirtyPathedScenes("auto-saved");
        }

        private static bool OnWantsToQuit()
        {
            // Even with the guard off, never block the quit — that would defeat the
            // whole point (a frozen editor). Just don't touch the scenes.
            if (Enabled)
            {
                SaveDirtyPathedScenes("auto-saved");
                // A dirty *untitled* scene has no path to save to; leaving it would
                // pop the "save changes before quitting?" modal we exist to prevent.
                // Discard it by swapping in a fresh empty single scene.
                string discarded = DirtyUntitledSceneNames();
                if (discarded.Length > 0)
                {
                    Debug.LogWarning(
                        "[SceneAutoSaveGuard] discarding unsaved untitled scene(s) on quit: "
                            + discarded);
                    EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                }
            }
            // ALWAYS allow the quit to proceed.
            return true;
        }

        private static void OnQuitting()
        {
            // Backstop: wantsToQuit already saved, but another handler in the quit
            // chain may have re-dirtied a pathed scene after it. Save once more.
            if (Enabled)
            {
                SaveDirtyPathedScenes("auto-saved");
            }
        }

        [MenuItem(EnabledMenu, priority = 200)]
        private static void ToggleEnabled()
        {
            bool now = !Enabled;
            EditorPrefs.SetBool(EnabledPref, now);
            Menu.SetChecked(EnabledMenu, now);
            if (!now)
            {
                s_dueAt = -1.0; // cancel any pending save when the guard is disabled
            }
        }

        [MenuItem(EnabledMenu, validate = true)]
        private static bool ToggleEnabledValidate()
        {
            // Reflect the persisted toggle in the menu's checkmark each time it opens.
            Menu.SetChecked(EnabledMenu, Enabled);
            return true;
        }

        [MenuItem(SaveNowMenu, priority = 201)]
        private static void SaveOpenScenesNow()
        {
            s_dueAt = -1.0; // immediate flush, bypassing the debounce
            SaveDirtyPathedScenes("auto-saved");
        }

        private static bool HasDirtyPathedScene()
        {
            for (int i = 0; i < SceneManager.sceneCount; i++)
            {
                Scene scene = SceneManager.GetSceneAt(i);
                if (scene.isDirty && !string.IsNullOrEmpty(scene.path))
                {
                    return true;
                }
            }
            return false;
        }

        // Save every loaded scene that is dirty and has a path on disk. Untitled
        // scenes (empty path) are always skipped so we never trigger a "Save As"
        // modal. Returns the number saved; logs the paths when at least one saved.
        private static int SaveDirtyPathedScenes(string verb)
        {
            var saved = new List<string>();
            for (int i = 0; i < SceneManager.sceneCount; i++)
            {
                Scene scene = SceneManager.GetSceneAt(i);
                if (scene.isDirty
                    && !string.IsNullOrEmpty(scene.path)
                    && EditorSceneManager.SaveScene(scene))
                {
                    saved.Add(scene.path);
                }
            }
            if (saved.Count > 0)
            {
                Debug.Log(
                    "[SceneAutoSaveGuard] " + verb + " " + saved.Count + " scene(s): "
                        + string.Join(", ", saved));
            }
            return saved.Count;
        }

        private static string DirtyUntitledSceneNames()
        {
            var names = new List<string>();
            for (int i = 0; i < SceneManager.sceneCount; i++)
            {
                Scene scene = SceneManager.GetSceneAt(i);
                if (scene.isDirty && string.IsNullOrEmpty(scene.path))
                {
                    names.Add(string.IsNullOrEmpty(scene.name) ? "<untitled>" : scene.name);
                }
            }
            return string.Join(", ", names);
        }
    }
}
