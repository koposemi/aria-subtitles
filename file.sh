#!/data/data/com.termux/files/usr/bin/bash
# 영상 파일에 자막 만들기:  aria-file /sdcard/Movies/Telegram/영상.mp4
if [ -z "$1" ]; then echo "사용법: aria-file <영상 파일 경로> [--no-burn | --burn-only]"; exit 1; fi
APP="$(cd "$(dirname "$0")" && pwd)"
F="$(realpath "$1")"; shift
cd "$APP/src" || exit 1
termux-wake-lock 2>/dev/null
export ARIA_LAN=0 ARIA_NO_ADB=1
exec python -m realtime_subtitles.youtube_subs.file_subs "$F" "$@"
