Set shell = CreateObject("WScript.Shell")
scriptPath = CreateObject("Scripting.FileSystemObject").BuildPath(CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName), "start_windows_gui.ps1")
cmd = "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & scriptPath & """"
shell.Run cmd, 0, False
