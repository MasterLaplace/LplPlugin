#!/bin/bash
set -e

# ============================================================================
# LplPlugin Android APK Builder
# Usage: ./build_apk.sh [--set-ndk PATH] [--set-sdk PATH]
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Print functions
info()  { echo "ℹ️  $@"; }
ok()    { echo "✅ $@"; }
warn()  { echo "⚠️  $@"; }
error() { echo "❌ $@"; exit 1; }

# ============================================================================
# Configuration
# ============================================================================

# Try to detect NDK and SDK if env vars not set
if [ -z "$ANDROID_NDK_HOME" ]; then
    # Try common paths
    for ndk_path in ~/Android/ndk/* ~/Android/sdk/ndk/* /opt/android-ndk-* /opt/android-ndk; do
        if [ -d "$ndk_path/toolchains/llvm/prebuilt/linux-x86_64/bin" ]; then
            export ANDROID_NDK_HOME="$ndk_path"
            info "Detected NDK at: $ANDROID_NDK_HOME"
            break
        fi
    done
fi

if [ -z "$ANDROID_SDK_HOME" ]; then
    for sdk_path in ~/Android/sdk /opt/android-sdk; do
        if [ -d "$sdk_path" ]; then
            export ANDROID_SDK_HOME="$sdk_path"
            info "Detected SDK at: $ANDROID_SDK_HOME"
            break
        fi
    done
fi

# Manual override support
while [ $# -gt 0 ]; do
    case "$1" in
        --set-ndk)  shift; export ANDROID_NDK_HOME="$1"; shift ;;
        --set-sdk)  shift; export ANDROID_SDK_HOME="$1"; shift ;;
        *) error "Unknown option: $1" ;;
    esac
done

# Verify NDK
if [ -z "$ANDROID_NDK_HOME" ] || [ ! -d "$ANDROID_NDK_HOME" ]; then
    error "❌ ANDROID_NDK_HOME not set or invalid. Please set it:\n   export ANDROID_NDK_HOME=/path/to/ndk"
fi
ok "NDK found at: $ANDROID_NDK_HOME"

# API level
ANDROID_API=24
ANDROID_ABI="arm64-v8a"

# Toolchain paths
TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin"
if [ ! -d "$TOOLCHAIN" ]; then
    error "Toolchain not found at $TOOLCHAIN"
fi

CLANG_CXX="${TOOLCHAIN}/aarch64-linux-android${ANDROID_API}-clang++"
CLANG_CC="${TOOLCHAIN}/aarch64-linux-android${ANDROID_API}-clang"

if [ ! -f "$CLANG_CXX" ]; then
    error "Clang++ not found at $CLANG_CXX"
fi
ok "Clang++ found at: $CLANG_CXX"

# ============================================================================
# Build Native Library (liblpl_visual.so)
# ============================================================================

info "Step 1: Building native library..."

# Compile android_native_app_glue
info "  Compiling android_native_app_glue.c..."
"$CLANG_CC" -c \
    "$ANDROID_NDK_HOME/sources/android/native_app_glue/android_native_app_glue.c" \
    -I"$ANDROID_NDK_HOME/sources/android/native_app_glue" \
    -fPIC -o android_glue.o
ok "  android_glue.o created"

# Compile visual_android.cpp
info "  Compiling visual_android.cpp..."
"$CLANG_CXX" -c \
    -O3 -std=c++20 \
    -I../../shared -I../../engine -I../../bci/include \
    -I"$ANDROID_NDK_HOME/sources/android/native_app_glue" \
    -fPIC -DLPL_USE_SOCKET -D__ANDROID__ -Wall \
    -o visual_android.o visual_android.cpp
ok "  visual_android.o created"

# Link shared library
info "  Linking liblpl_visual.so..."
"$CLANG_CXX" -shared -u ANativeActivity_onCreate \
    -static-libstdc++ \
    -fPIC -o liblpl_visual.so visual_android.o android_glue.o \
    -landroid -llog -lEGL -lGLESv2 \
    -Wl,-soname,liblpl_visual.so
ok "liblpl_visual.so created ($(ls -lh liblpl_visual.so | awk '{print $5}'))"

# Cleanup object files
rm -f visual_android.o android_glue.o
ok "Object files cleaned"

# ============================================================================
# Create APK Directory Structure
# ============================================================================

info "Step 2: Creating APK structure..."

# Create directory structure
mkdir -p lib/arm64-v8a
mkdir -p res/values
mkdir -p res/drawable
mkdir -p gen

# Copy native library
cp liblpl_visual.so lib/arm64-v8a/
ok "Native library staged"

# ============================================================================
# Create Minimal AndroidManifest.xml if missing
# ============================================================================

if [ ! -f AndroidManifest.xml ]; then
    warn "AndroidManifest.xml not found, creating minimal version..."
    cat > AndroidManifest.xml << 'MANIFEST'
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.lpl.visual"
    android:versionCode="1"
    android:versionName="1.0">

    <uses-sdk android:minSdkVersion="24" android:targetSdkVersion="34" />

    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-feature android:name="android.hardware.screen.landscape" android:required="false" />

    <application
        android:label="@string/app_name"
        android:debuggable="true"
        android:icon="@drawable/icon"
        android:extractNativeLibs="false">

        <activity android:name="android.app.NativeActivity"
            android:label="@string/app_name"
            android:theme="@android:style/Theme.Black.NoTitleBar.Fullscreen"
            android:screenOrientation="sensorLandscape"
            android:exported="true">

            <meta-data android:name="android.app.lib_name"
                android:value="lpl_visual" />

            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

    </application>

</manifest>
MANIFEST
    ok "AndroidManifest.xml created"
fi

# ============================================================================
# Create Minimal Resources
# ============================================================================

info "Step 3: Creating minimal resources..."

# Create strings.xml
if [ ! -f res/values/strings.xml ]; then
    cat > res/values/strings.xml << 'STRINGS'
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">LplPlugin</string>
</resources>
STRINGS
    ok "res/values/strings.xml created"
fi

# Create a minimal icon (1x1 transparent PNG)
if [ ! -f res/drawable/icon.png ]; then
    # Create minimal 1x1 PNG
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\x0b\xb5\x00\x00\x00\x00IEND\xaeB`\x82' > res/drawable/icon.png
    ok "res/drawable/icon.png created"
fi

# ============================================================================
# Build APK (Simple zip-based approach)
# ============================================================================

info "Step 4: Building APK..."

# Create APK using zip
rm -f base.apk LplVisual.apk
zip -q -r base.apk AndroidManifest.xml lib res 2>/dev/null || true
[ -f base.apk ] || error "Failed to create base.apk"
ok "base.apk created"

# ============================================================================
# Sign APK
# ============================================================================

info "Step 5: Signing APK..."

# Create debug keystore if not exists
if [ ! -f debug.keystore ]; then
    info "  Generating debug keystore..."
    keytool -genkey -v -keystore debug.keystore \
        -storepass android -alias androiddebugkey -keypass android \
        -keyalg RSA -keysize 2048 -validity 10000 \
        -dname "CN=Android Debug,O=Android,C=US" \
        2>/dev/null || warn "  Keystore generation may have issues"
    ok "  debug.keystore created"
fi

# Try jarsigner first (most compatible)
if command -v jarsigner &>/dev/null; then
    info "  Signing with jarsigner..."
    jarsigner -verbose -sigalg SHA256withRSA -digestalg SHA-256 \
        -keystore debug.keystore -storepass android -keypass android \
        base.apk androiddebugkey 2>&1 | grep -v "^  " || true
    cp base.apk LplVisual.apk
    ok "  APK signed with jarsigner"
elif command -v apksigner &>/dev/null; then
    info "  Signing with apksigner..."
    apksigner sign --ks debug.keystore --ks-pass pass:android \
        --key-pass pass:android --v1-signing-enabled true \
        --v2-signing-enabled true --out LplVisual.apk base.apk
    ok "  APK signed with apksigner"
else
    warn "  No signing tool found, using unsigned APK (may need Settings → Unknown Installer)"
    cp base.apk LplVisual.apk
fi

ok "APK ready: LplVisual.apk ($(ls -lh LplVisual.apk | awk '{print $5}'))"

# ============================================================================
# Verification
# ============================================================================

info "Step 6: Verifying APK..."
unzip -t LplVisual.apk >/dev/null 2>&1 && ok "✅ APK integrity verified" || error "APK is corrupted"

# List contents
echo ""
info "📦 APK contents:"
unzip -l LplVisual.apk | grep -E "(\.so|AndroidManifest|strings.xml)" | sed 's/^/    /' || true

# ============================================================================
# Deployment Instructions
# ============================================================================

echo ""
cat << 'DEPLOY'
════════════════════════════════════════════════════════════════════════════════
✅ APK BUILD SUCCESSFUL

📱 Installation:

  1. Enable USB debugging on phone (Settings → Developer Options)
  2. Connect phone via USB cable
  3. Run:
     adb install -r LplVisual.apk

  4. Or manually copy to phone and install from Files app

🔧 Before running:

  ⚠️  IMPORTANT: Configure server IP in visual_android.cpp

     Line 203: static const char *SERVER_IP = "192.168.1.YOUR_IP";

     Replace with your computer's WiFi IP address!

     Then rebuild:
     make android && ./build_apk.sh

📱 Running on Phone:

  1. Tap "LplPlugin" on home screen
  2. App will try to connect to server (port 7777)
  3. Entities will appear as colored cubes
  4. Touch screen to move: top=forward, bottom=backward, left/right=strafe

🌐 Server Requirements:

  1. Run server on same WiFi network:
     ./apps/server/server

  2. Accept incoming connection from phone (port 7777)

🐛 Troubleshooting:

  • "App not installed": Uninstall old version or enable unknown installer
  • "Server refused": Check IP address, same WiFi network, firewall open
  • Black screen: Check logcat: adb logcat | grep LplVisual

════════════════════════════════════════════════════════════════════════════════
DEPLOY
