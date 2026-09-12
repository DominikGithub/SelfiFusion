# SelfieFusion

<table>
  <tbody>
    <tr>
      <td width="600" valign="top">
        <a href="docs/concept.svg"><img src="docs/concept.svg" width="100%" alt="SelfieFusion concept: while you face the view, the front camera captures you and the back camera captures the scene in front of you — both shots fuse into one photo; the person's position and size adapt to one-finger drag gestures"></a>
      </td>
      <td valign="top">
        <strong>You don't have to turn your back to the background you actually want in the picture.</strong><br><br>See yourself and what you're looking at — in one picture. A normal selfie has a built-in blind spot: the front camera faces <em>you</em>, so the view you're actually looking at never makes it into the photo — the only workaround is turning around, which means the camera no longer faces you. SelfieFusion removes that blind spot. While you take a selfie, <strong>both phone cameras shoot at the same time</strong> — the front camera captures you, the back camera captures the scene in front of you — and the app fuses both into a single full-resolution photo.<br><br>The result looks like a picture taken <em>of</em> you standing in that scene: you in the foreground, the view you were facing as the backdrop — as if someone else had photographed you.<br><br><em>Arrange it live: one-finger drag gestures adapt the person's position and size — what you arrange is what you get.</em>
      </td>
    </tr>
  </tbody>
</table>


Under the hood, SelfieFusion is an Android app that streams the front and back cameras concurrently (on devices with concurrent-camera support), segments the person from the front-camera feed entirely on-device, and composites them live over the rear-camera scene — you watch the fused selfie take shape in real time before you press the shutter.

## Features

- **Dual-camera live preview** — front + back cameras stream simultaneously via CameraX `ConcurrentCamera`
- **On-device segmentation** — ML Kit Selfie Segmentation, works fully offline, no cloud
- **Grounded person gestures** — in Composite mode the person always stays realistic: its bottom edge is locked to the bottom of the frame, so it never floats in empty space. One finger moves it sideways, a vertical drag resizes it (top edge follows your finger), two fingers zoom, double-tap resets.
- **What you arrange is what you get** — the saved fused still reproduces the exact position and size of the person from the live preview, including the grounded bottom edge.
- **Optional stats overlay** — FPS, mask resolution/format, sensor rotation and camera mode; hidden by default, toggleable via the menu (*Show stats (FPS)*)
- **Full-resolution fused stills** — one shutter press saves the fused JPG plus, optionally, the source files (menu toggle)
- **Color match** — before fusing, the person's lighting statistics (brightness, contrast, white balance) adapt toward the back scene's, so the composite stops looking like two photos taken by two different cameras. The white-balance adoption is damped and capped, and the brightness adoption is damped with a subject-brighten bias — the person follows the scene's tonal direction but stays bright as the photo's subject instead of being dragged down to the scene average. Toggleable via the menu (*Color match person to scene*, default on)
- **Generative seam blend (AI, prototype)** — where your body was cut off by the front camera's frame, the edge always gets a smooth fade (guaranteed, never a hard "camera image limit" rectangle), and a LaMa inpainting model additionally *paints in* the seam band between you and the back scene from the surrounding person and scene textures — a generated continuation instead of a mere dissolve. Runs fully on-device (ONNX Runtime), once per saved still (the live preview keeps the fade; per-frame AI would take seconds). The model ships inside the APK (downloaded once into the git-ignored `models/` folder for the build — see *Generative seam blend* under [Building](#building)); a build without it keeps the fade alone. Toggleable via the menu (*Generative seam blend (AI)*, default on)
- **Graceful fallback** — devices without concurrent front+back streaming automatically run in front-camera segmentation mode

## Requirements

- Android 8.0+ (API 26)
- Dual-camera mode needs a device that reports concurrent front+back streaming (Android 11+ / API 30)
- Camera permission only — no internet access required
- Developed and tested on a Google Pixel 9

## Tech stack

- Java 
- [CameraX](https://developer.android.com/training/camerax) 1.3.4 (`ConcurrentCamera`) 
- [ML Kit](https://developers.google.com/ml-kit/vision/selfie-segmentation) Selfie Segmentation 
- single-activity app, no UI framework 

## Models

SelfieFusion uses two on-device ML models — no cloud, no API keys, everything runs offline on your phone:

- **Selfie segmentation** (ML Kit Selfie Segmentation, ~256×256, bundled in the APK) — a small vision model that classifies every pixel of the front-camera image as *person* or *background*. The resulting confidence mask is what cuts you out of the selfie: it drives the live person overlay in the preview and the feathered alpha matte of the full-resolution still.
- **Inpainting** (LaMa-class inpainting network, optional prototype) — a generative model that synthesizes missing image content from the patterns surrounding it. It powers the *generative seam blend*: where your body was cut off by the front camera's frame, the seam band between the front and back photo is filled in ("painted") from the adjacent person and scene textures instead of being simply faded, so the merge looks continuous. Runs once per saved still via ONNX Runtime; the model is bundled into the APK (see *Generative seam blend* under [Building](#building)).

## Building

You need a JDK, the Android SDK command-line tools, and `unzip`/`wget`. No IDE required — every step runs in a normal shell.

### 1. Install a JDK (17 or newer; 21 tested)

```bash
sudo apt install openjdk-21-jdk-headless   # Debian/Ubuntu; use your distro's equivalent
java -version
```

### 2. Get the Android SDK command-line tools

```bash
wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -P /tmp
mkdir -p ~/android-sdk/cmdline-tools/latest
unzip -q /tmp/commandlinetools-linux-11076708_latest.zip -d ~/android-sdk/cmdline-tools/latest
mv ~/android-sdk/cmdline-tools/latest/cmdline-tools/* ~/android-sdk/cmdline-tools/latest
rmdir ~/android-sdk/cmdline-tools/latest/cmdline-tools
```

(If that URL has retired, grab the current *"Command line tools only"* zip from [developer.android.com/studio](https://developer.android.com/studio#command-line-tools-only).)

### 3. Point the project at the SDK

In the repository root, create a `local.properties` file (git-ignored) containing:

```properties
sdk.dir=/home/<you>/android-sdk
```

Alternatively, export `ANDROID_HOME=~/android-sdk` in your shell.

### 4. Install the platform + build tools and accept the licenses

```bash
SDKM=~/android-sdk/cmdline-tools/latest/bin/sdkmanager
yes | $SDKM --licenses > /dev/null
$SDKM "platforms;android-34" "build-tools;34.0.0"
```

### 5. Build the APK

```bash
git clone <repository-url> && cd selfie-fusion
./gradlew assembleDebug        # first run downloads Gradle 8.7 via the wrapper
```

The debug APK lands in `app/build/outputs/apk/debug/SelfiFusion.apk`.

### 6. (Optional) install on a device

With USB debugging enabled:

```bash
adb install -r app/build/outputs/apk/debug/SelfiFusion.apk
```

Or over WiFi:

```bash
adb pair <phone-ip>:<pairing-port>        # "Pair device with pairing code" on the phone, enter the 6-digit code
adb connect <phone-ip>:<main-port>        # port from the main Wireless debugging screen (NOT the pairing port)
adb devices -l                            # should show "device"
adb install -r app/build/outputs/apk/debug/SelfiFusion.apk
```

Needs `adb` ≥ 30 (`adb version`). If `adb pair` times out, the WiFi likely blocks device-to-device traffic — use USB instead.

### 7. Generative seam blend model

The AI seam blend uses a LaMa inpainting model that ships **inside the APK** (`models/inpaint.onnx`, ~93 MB — the APK grows accordingly; it is sideloaded via `adb`, so Play Store base-module size limits don't apply). One-time build setup:

1. Download the model once and place it at `models/inpaint.onnx` (the folder is git-ignored — ~100 MB blobs are never committed):
   [huggingface.co/opencv/inpainting_lama — `inpainting_lama_2025jan.onnx`](https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx)
2. Build and install as usual — **no device-side copy steps**. The first capture with *Generative seam blend (AI)* extracts the model from the APK into the app's private storage once (shown as *Preparing AI blend*, a few seconds); afterwards it loads instantly.
3. Done — the menu toggle *Generative seam blend (AI)* is on by default and activates automatically when the model is present. A checkout built without `models/inpaint.onnx` still works — without it (or if loading or the blend itself fails) the saved photo still always gets the smooth edge fade (never a hard seam); `adb logcat -s SelfieFusion` tells you exactly which treatment ran (`extracted inpainting model from APK asset`, `inpainting model loaded`, `no inpainting model`, `window(s) for n seam band(s) inpainted`, or `no seam bands`).

Notes: the blend runs once per saved still — one model inference per window, and long seam bands are split into several overlapping near-full-resolution windows whose pastes crossfade (the progress card shows *AI edge blend n/total*, one step per window; a long band therefore takes a few inferences, visible as several steps). The live preview always shows the edge-fade look. The prototype build targets arm64 devices (e.g. Pixel 9).

### Note for ARM64 (aarch64) machines

Google ships `aapt2` from Maven as an **x86_64 binary** only. On aarch64 hosts (ARM Chromebook Crostini, Raspberry Pi 64-bit, Asahi Linux, …) install your distro's native `aapt` package and redirect AGP to it via the **user-level** `~/.gradle/gradle.properties` (not the project file, so the repo stays portable):

```bash
sudo apt install aapt
echo 'android.aapt2FromMavenOverride=/usr/bin/aapt2' >> ~/.gradle/gradle.properties
```

The committed `gradle.properties` is tuned for low-RAM machines (`-Xmx900m`); raise that value on normal hardware if you like. Android Studio also opens this project fine if you have it — the steps above are the pure-terminal path.

## How it works

1. Both cameras are opened through CameraX's `ConcurrentCamera` (front: `Preview + ImageAnalysis + ImageCapture`, back: `Preview + ImageCapture`), with an automatic step-down ladder if the platform rejects a combination.
2. The ML Kit selfie segmenter refreshes the person mask at a fixed cadence (the model is the expensive part under dual-camera load), while the person video itself is re-rendered from every camera frame using the cached mask — that keeps the live composite smooth even when segmentation is slow.
3. The confidence mask is temporally smoothed (EMA), spatially blurred and converted into a smoothstep alpha matte, which kills edge flicker and gives soft hair edges. Where the person was cut by the camera frame (sides or top), the matte fades smoothly into the scene instead of showing a rectangular seam; only the bottom edge stays hard — it is glued to the frame bottom, where a real photo crops it.
4. In Composite mode the person cut-out is drawn over the rear-camera preview; touch gestures resize and move it, with its bottom edge always locked to the frame so the composition stays photorealistic.
5. On shutter press both cameras capture full-resolution JPEGs. Segmentation runs again on the full-resolution front photo, and the cut-out is placed onto the back photo exactly as arranged in the live preview — same position, same size, grounded bottom edge.
6. Before compositing, the cut-out's luma/chroma statistics adapt toward the back photo's (Reinhard-style color match, applied as a single native `ColorMatrix` pass), so the person adopts the scene's tonal direction and white balance instead of keeping the front camera's — damped, so the person keeps subject brightness rather than being darkened down to the scene average.
7. (Prototype, optional model) Where the person was cut by the front camera's frame, the edge first gets the smooth frame-crop fade (always, so a scaled-down person never shows a hard "camera image limit" line — the fade keeps a minimum width in the placed photo), and then — if the APK bundles the model — a LaMa inpainting model (ONNX Runtime, one scaled 512×512 window per seam band covering the band plus its surrounding context) re-synthesizes the band from the person and scene textures on both sides: a generated, continuous merge painted over the fade.

## Notes

- The stats overlay (menu: *Show stats (FPS)*) shows the active view mode, FPS, mask resolution/format, sensor rotation and camera configuration. It is hidden by default; in *Overlay off* mode its FPS number shows the raw camera delivery rate, which helps separate camera-side from pipeline-side slowness.
- The color match (menu: *Color match person to scene*, default on) runs in the save pipeline only — the live preview still shows the person in the front camera's raw colors; the saved JPG contains the scene-matched person. The chroma adaptation is damped and capped, so even strongly colored scenes (sunset, forest) tint the face only moderately.
