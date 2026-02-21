#!/bin/bash
set -e

# Paths
SDK="/tmp/lpl_sdk_real"
BUILD_TOOLS="$SDK/build-tools/34.0.0"
PLATFORM="$SDK/platforms/android-34/android.jar"
AAPT2="$BUILD_TOOLS/aapt2"
ZIPALIGN="$BUILD_TOOLS/zipalign"
APKSIGNER="$BUILD_TOOLS/apksigner"

# Clean
rm -f compiled_res.zip base.apk aligned.apk LplVisual.apk

echo "--- 1. Compile Resources ---"
# We need at least one resource for AAPT2 to work happily usually, or valid structure
$AAPT2 compile --dir res -o compiled_res.zip

echo "--- 2. Link APK ---"
$AAPT2 link -o base.apk \
    -I "$PLATFORM" \
    --manifest AndroidManifest.xml \
    --java gen \
    compiled_res.zip \
    --auto-add-overlay

echo "--- 3. Add Native Lib ---"
# Ensure lib is in correct folder structure inside the zip
# Zip command updates the apk in-place
# We need to add 'lib/arm64-v8a/liblpl_visual.so' to the root of the APK
# CRITICAL: Use -0 (store) for native libraries to ensure alignment works and modern Android accepts it
mkdir -p lib/arm64-v8a
cp liblpl_visual.so lib/arm64-v8a/
zip -u -0 base.apk lib/arm64-v8a/liblpl_visual.so

mkdir -p lib/armeabi-v7a
cp liblpl_visual_v7.so lib/armeabi-v7a/liblpl_visual.so
zip -u -0 base.apk lib/armeabi-v7a/liblpl_visual.so

echo "--- 4. Zipalign ---"
# -v for verbose, -p for memory alignment of .so
$ZIPALIGN -f -v 4 base.apk aligned.apk

echo "--- 5. Sign APK ---"
if [ ! -f debug.keystore ]; then
    echo "Generating debug keystore..."
    keytool -genkey -v -keystore debug.keystore -storepass android -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Android Debug,O=Android,C=US"
fi

# Explicitly enable v1 and v2 signing for max compatibility
$APKSIGNER sign --ks debug.keystore --ks-pass pass:android --key-pass pass:android \
    --v1-signing-enabled true --v2-signing-enabled true \
    --out LplVisual.apk aligned.apk

echo "=========================================="
echo "SUCCESS: LplVisual.apk created!"
echo "=========================================="
