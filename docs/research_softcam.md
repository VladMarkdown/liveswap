# softcam Virtual Camera Research

## Recommendation: tshino/softcam (MIT)
- Repo: https://github.com/tshino/softcam
- License: MIT
- Latest: v1.8.1 (March 2025)
- CLSID: {AEF3B972-5FA5-4647-9571-358EB472BC9E}
- Default name: "DirectShow Softcam"

## IMPORTANT: pyvirtualcam is also MIT (not GPL)
pyvirtualcam (https://github.com/letmaik/pyvirtualcam) wraps softcam on Windows.
Simpler API, same underlying mechanism.

## C API
```c
scCamera scCreateCamera(int width, int height, float framerate);
void scDeleteCamera(scCamera camera);
void scSendFrame(scCamera camera, const void* image_bits);  // BGR, width*height*3 bytes
bool scWaitForConnection(scCamera camera, float timeout);
bool scIsConnected(scCamera camera);
```
- Frame format: BGR, 3 bytes/pixel, no row padding, width/height must be multiples of 4
- OpenCV frames are natively BGR — pass directly, zero conversion

## App Compatibility
All target apps use DirectShow Video Input Devices:
- Zoom: DirectShow ✅
- Google Meet (Chrome): MediaFoundation + DirectShow fallback ✅
- Teams: DirectShow ✅
- Discord: DirectShow ✅
- TikTok Live Studio: DirectShow ✅

## FriendlyName Rename
```python
SOFTCAM_CLSID = "{AEF3B972-5FA5-4647-9571-358EB472BC9E}"
VID_INPUT_CAT = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"

def rename_softcam_device(new_name="Logitech HD Pro Webcam C920"):
    """Requires admin. Restart target app after rename."""
    import winreg
    with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT,
                        f"CLSID\\{SOFTCAM_CLSID}",
                        access=winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, new_name)
    with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT,
                        f"CLSID\\{VID_INPUT_CAT}\\Instance\\{SOFTCAM_CLSID}",
                        access=winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, "FriendlyName", 0, winreg.REG_SZ, new_name)
```

## Setup
1. Build from source (VS2022, Release x64) OR download v1.8.1 release
2. Run RegisterSoftcam.bat as Administrator (one-time)
3. Place softcam.dll next to virtual_cam.py

## Unity Capture: SKIP
No Python frame injection path. Unity plugin only.

## OBS Virtual Camera: SKIP
GPL-2.0, requires OBS running.
