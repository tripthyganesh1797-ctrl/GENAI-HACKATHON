"""
builtin_knowledge.py — a small, hand-written generic troubleshooting
reference text for the offline fallback path, used ONLY when a caller
supplies no `siis_response` at all.

Why this exists: without this file, `offline_fallback.offline_extract()`
had nothing to extract a plan FROM when no reference text was supplied --
see its early-return for a blank `siis_response`. That meant the "zero
setup, zero API key" offline path could only ever answer the 20 official
evaluation queries (which come paired with real SIIS reference text) and
returned a bare "no match" for a real person just typing their own
complaint with nothing else, like "my battery is draining fast" -- even
though the underlying retrieval, validation, and deeplink-matching code
was all completely real. That gap is a real usability problem, not a demo
concern: a person asking "can I actually use this for my own phone right
now, or is it a fake UI" deserves a real answer, and for the offline path
the honest fix is to give offline_extract() *something* to ground a
reasonable, generic answer in.

This is NOT official Samsung support text, and is never presented as such
-- it's ordinary, widely-known Android/Samsung troubleshooting advice
(the same kind of steps you'd find in any phone's help center), written by
hand in the same "# Header" + imperative-sentence shape offline_fallback.py's
parse_sections()/extract_steps() already know how to read, so it flows
through the EXACT SAME relevance-gating, multi-issue-splitting, and
Stage 2 deeplink-matching logic as real reference text -- no special-cased
code path, no lowered bar for "is this actually relevant" (an out-of-scope
complaint like "how do I cook pasta" still correctly clears no section's
relevance threshold and comes back no_match, exactly as before this file
existed; see tests/test_builtin_knowledge.py).

When a real `siis_response` IS supplied (the official 20-query evaluation
shape, or a caller pasting their own reference text), this file is never
consulted -- it only fills the gap for a genuinely bare complaint on the
offline path. The LLM path never needs this at all: it has its own
knowledge to draw on already.
"""

BUILTIN_REFERENCE_TEXT = """
# Battery Draining Quickly
Open Settings and tap Battery to check which apps are using the most power. Enable Adaptive battery so the system limits background activity for apps you rarely use. Turn on Power saving mode to reduce performance and background data while the battery is low. Check for apps running in the background and close the ones you are not using. Set your screen brightness lower or enable adaptive brightness, since a bright screen is one of the biggest sources of battery drain. Disable Always On Display if you do not need it. Turn off location services for apps that do not need to know where you are. Update your apps, since an outdated app can drain the battery more than usual. Restart your phone to clear temporary background processes that may be consuming power.

# Battery Not Charging
Try a different charging cable and adapter to rule out a damaged one. Wipe the charging port gently to remove any lint or debris that may be blocking the connection. Try a different wall outlet to rule out a faulty power source. Turn off USB debugging mode in Developer options if it was accidentally left on, since it can sometimes interfere with charging behavior. Restart your phone and reconnect the charger. Check Battery settings to see the current charging status and estimated time remaining. Contact Samsung Support or visit a Samsung service center if the phone still does not charge after trying these steps.

# Device Overheating
Close background apps that may be using the processor heavily, such as games or camera apps left open. Remove the phone case while charging so heat can dissipate properly. Unplug the phone if it feels hot while charging and let it cool down before continuing to use it. Turn off mobile hotspot and other radios you are not using. Update your apps and software, since outdated software can run inefficiently and generate extra heat. Restart your phone to close any runaway background process. Contact Samsung Support if the phone continues to overheat during normal, everyday use.

# Camera App Crashing
Force stop the Camera app from Settings, then Apps, then Camera, and tap Force stop. Clear the Camera app cache from the same Apps screen to remove any corrupted temporary files. Update the Camera app to the latest version from the Galaxy Store. Restart your phone to free up memory that may be causing the crash. Check available storage space, since very low storage can cause an app to crash on launch. Uninstall recent Camera app updates in Settings if the crashing started right after an update.

# Camera Blurry Or Out Of Focus
Wipe the camera lens gently with a soft, dry cloth to remove smudges or dust. Remove any screen protector or case that may be blocking the lens. Tap the screen where you want the camera to focus before taking the photo. Turn off Scene optimizer if it is causing unexpected focus behavior. Restart the Camera app and try again. Update the Camera app to the latest version from the Galaxy Store.

# Screen Flickering
Update your software in Settings, then Software update, since a display driver bug is a common cause of flickering. Turn off Adaptive brightness and set the brightness manually to check if the flickering stops. Disable Blue light filter and Dark mode one at a time to check whether either is causing the issue. Restart your phone to rule out a temporary display glitch. Restart your phone in Safe mode to check whether a third-party app is causing the flickering. Contact Samsung Support or visit a Samsung service center if the flickering continues in Safe mode, since that points to a hardware issue.

# Black Or Blank Screen
Press and hold the Power button and Volume down button together for about seven seconds to force restart the phone. Charge your phone for a few minutes before trying to turn it on again, since a black screen can simply mean the device has no charge left. Restart your phone in Safe mode to check whether a third-party app is causing the black screen. Contact Samsung Support or visit a Samsung service center if the screen stays black after a force restart.

# Screen Cracked Or Physically Damaged
Install a temporary screen protector over the cracked area to prevent further damage and reduce the risk of injury from loose glass. Back up your data as soon as possible in case the screen stops responding completely. Visit a Samsung service center or an authorized repair shop for a screen replacement, since a cracked screen needs a hardware fix, not a software one.

# Touchscreen Unresponsive
Remove the screen protector and case, since a poor-quality screen protector can interfere with touch sensitivity. Wipe the screen with a soft, dry cloth to remove moisture or dirt. Restart your phone to clear a temporary software glitch, if the touchscreen is not responding to touch at all. Restart your phone in Safe mode to check whether a third-party app is causing the unresponsive touch. Contact Samsung Support or visit a Samsung service center if the touchscreen remains unresponsive or stops responding after these steps.

# Wifi And Bluetooth Connectivity
Toggle Wi-Fi off and back on in Settings, then Connections, then Wi-Fi. Forget the Wi-Fi network and reconnect by entering the password again. Restart your phone if the Wi-Fi connection keeps disconnecting or randomly dropping throughout the day. Toggle Bluetooth off and back on in Settings, then Connections, then Bluetooth. Restart your router if other devices are also having trouble connecting to it. Turn on Airplane mode for about ten seconds and then turn it back off to reset the radios. Update your software in Settings, then Software update, since connectivity bugs are often fixed in updates. Reset network settings in Settings, then General management, then Reset, if the problem continues.

# Storage Full
Check Settings, then Battery and device care, then Storage, to see what is taking up space. Uninstall unused apps and clear the cache of the ones you keep. Back up photos and videos to Google Photos or another cloud service and then remove them from local storage. Clear the Downloads folder of files you no longer need. Try the built-in storage cleanup tool to remove duplicate and unnecessary files automatically.

# App Crashing
Force stop the app from Settings, then Apps, select the app, and tap Force stop. Clear the app's cache and app data from the same Apps screen. Update the app to the latest version from the Galaxy Store or Play Store. Restart your phone to free up memory. Uninstall and reinstall the app if it continues to crash after updating.

# Software Update Issue
Check your internet connection, since a weak or interrupted connection can cause an update to fail partway through. Clear extra storage space before starting the update, since updates need room to install. Restart your phone and try the update again from Settings, then Software update. Charge your phone to at least fifty percent before updating, since most updates require sufficient battery. Contact Samsung Support if the update continues to fail after these steps.

# Device Running Slow
Restart your phone to clear temporary files and close background processes. Update your apps and software to the latest versions, since outdated software can run inefficiently. Clear the cache of apps you use often from Settings, then Apps. Check available storage space, since a nearly full device can slow down significantly. Uninstall apps you no longer use.

# Speaker And Audio Issues
Check the volume settings and confirm Do not disturb mode is turned off. Remove any case or screen protector that may be covering the speaker grille. Wipe the speaker grille gently with a soft, dry brush to remove dust or lint. Restart your phone to rule out a temporary audio glitch. Restart your phone in Safe mode to check whether a third-party app is muting or interfering with the audio. Contact Samsung Support or visit a Samsung service center if the speaker still does not work.

# Microphone Issue
Remove any case that may be covering the microphone opening. Check app permissions in Settings, then Apps, to confirm the app actually has microphone access. Wipe the microphone opening gently with a soft, dry brush. Restart your phone to rule out a temporary software glitch. Try the microphone in a different app, such as Voice Recorder, to check whether the problem is app-specific. Contact Samsung Support if the microphone does not work in any app.

# Network And Signal Issues
Toggle Airplane mode on for about ten seconds and then turn it back off to reset the connection. Reinsert your SIM card to make sure it is seated correctly. Reset network settings in Settings, then General management, then Reset. Update your software in Settings, then Software update, since network bugs are sometimes fixed in updates. Contact Samsung Support or your carrier if the signal problem continues across multiple locations.
""".strip()


def get_builtin_reference_text() -> str:
    """Returns the built-in reference text used as a substitute
    `siis_response` on the offline path when the caller supplied none.
    A function (not a bare module constant import) so callers/tests have
    one obvious place to monkeypatch or extend this in the future."""
    return BUILTIN_REFERENCE_TEXT
