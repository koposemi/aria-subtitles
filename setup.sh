#!/data/data/com.termux/files/usr/bin/bash
# ARIA 자막 서버 설치 (안드로이드 Termux).  저장소를 ~/aria 로 받은 뒤 한 번만 실행:
#     cd ~/aria && bash setup.sh
# 다시 실행해도 안전합니다 (업데이트 후 패키지만 다시 맞추고 싶을 때).
set -e
APP="$(cd "$(dirname "$0")" && pwd)"

echo "[1/4] 필요한 프로그램 설치 (python, numpy, ffmpeg, nodejs, git) - 처음엔 몇 분 걸립니다"
pkg update -y
pkg upgrade -y -o Dpkg::Options::=--force-confnew   # 새로 깐 Termux는 이걸 먼저 해야 ffmpeg가 실행됨
pkg install -y python python-numpy ffmpeg nodejs git

echo "[2/4] yt-dlp 설치"
pip install -U yt-dlp || pkg install -y yt-dlp

echo "[3/4] Groq API 키"
KEYDIR="$HOME/.config/realtime-subtitles"
KEYFILE="$KEYDIR/groq_key.txt"
mkdir -p "$KEYDIR"
FOUND=""
for f in "$APP/groq_key.txt" "$HOME/storage/downloads/groq_key.txt" "/sdcard/Download/groq_key.txt"; do
  if [ -f "$f" ]; then FOUND="$f"; break; fi
done
if [ -n "$FOUND" ]; then
  cp "$FOUND" "$KEYFILE"
  chmod 600 "$KEYFILE"
  rm -f "$FOUND"        # 키 파일 사본이 다운로드 폴더 등에 남지 않게 지움
  echo "  키 $(grep -c gsk_ "$KEYFILE")개 복사 완료 ($FOUND 는 지웠습니다)"
elif [ -f "$KEYFILE" ]; then
  echo "  기존 키 파일을 그대로 씁니다 ($(grep -c gsk_ "$KEYFILE")개)"
else
  echo "  !! 키 파일이 없습니다. 아래 파일에 Groq 키를 한 줄에 하나씩 넣어 주세요:"
  echo "     $KEYFILE"
fi

echo "[4/4] 'aria' 명령 만들기 + Termux를 열면 자동 시작"
printf '#!/data/data/com.termux/files/usr/bin/bash\nexec bash "%s/start.sh" "$@"\n' "$APP" > "$PREFIX/bin/aria"
chmod +x "$PREFIX/bin/aria"
touch "$HOME/.bashrc"
grep -qx 'aria' "$HOME/.bashrc" || echo aria >> "$HOME/.bashrc"

echo
echo "설치 끝. Termux를 완전히 닫았다 다시 열면 서버가 켜집니다 (지금 바로 켜려면:  aria)"
echo "그 다음 Firefox에서 http://127.0.0.1:8765/aria.user.js 를 열어 스크립트를 한 번 설치하세요."
