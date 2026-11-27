#!/data/data/com.termux/files/usr/bin/bash
# 이 기기에서 ARIA 자막 서버를 켭니다 (이 기기의 브라우저만 접속 가능).
APP="$(cd "$(dirname "$0")" && pwd)"
cd "$APP/src" || exit 1
termux-wake-lock 2>/dev/null     # 화면이 꺼져도 CPU가 자지 않게
export ARIA_LAN=0 ARIA_NO_ADB=1
echo "ARIA 자막 서버를 시작합니다 (끄려면 Ctrl+C, 또는 Termux 알림에서 Exit)"
exec python -m realtime_subtitles.youtube_subs.server
