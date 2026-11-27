# ARIA 모바일 자막

안드로이드(Termux)에서 영상 소리를 Groq로 인식·번역해 영상 위에 한국어 자막을 띄웁니다.

## 설치

`groq_key.txt`(Groq 키 한 줄)를 다운로드 폴더에 넣고 Termux에서:

```bash
termux-setup-storage
pkg install -y git
git clone https://github.com/koposemi/aria-subtitles ~/aria
cd ~/aria && bash setup.sh
```

Termux를 다시 열고 Firefox(Tampermonkey)에서 `http://127.0.0.1:8765/aria.user.js` 설치.

## 업데이트

```bash
cd ~/aria && git pull
```

## 영상 파일에 자막

```bash
aria-file <영상 파일 경로>
```
