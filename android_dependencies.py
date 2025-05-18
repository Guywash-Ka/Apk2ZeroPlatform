ANDROID_PACKAGES = {
    "android.",
    "androidx.",
    "com.google.android.",
    "android.app.",
    "android.os.",
    "android.view.",
    "android.widget.",
    "android.content.",
}

ANDROID_COMPONENTS = {
    "Activity",
    "Fragment",
    "Service",
    "BroadcastReceiver",
    "ContentProvider",
    "Intent",
    "Context",
    "View",
}

POTENTIALLY_REPLACEABLE_APIS = {
    "android.util.Log": "java.util.logging.Logger",
    "android.os.AsyncTask": "java.util.concurrent.ExecutorService",
    "android.os.Handler": "java.util.Timer",
}


def is_android_import(import_line):
    for pkg in ANDROID_PACKAGES:
        if import_line.startswith(f"import {pkg}"):
            return True
    return False


def is_android_component(class_name):
    for component in ANDROID_COMPONENTS:
        if component in class_name:
            return True
    return False
